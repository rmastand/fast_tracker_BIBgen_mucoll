from copy import deepcopy
import torch
import os
import numpy as np
import zero
from tab_ddpm import GaussianMultinomialDiffusion
from utils_train import get_model, make_dataset, update_ema
import lib
import pandas as pd

class Trainer:
    def __init__(self, diffusion, train_iter, lr, weight_decay, steps, device=torch.device('cuda:1'),
                 val_loader=None, val_every=1000, parent_dir=None):
        self.diffusion = diffusion
        self.ema_model = deepcopy(self.diffusion._denoise_fn)
        for param in self.ema_model.parameters():
            param.detach_()

        self.train_iter = train_iter
        self.val_loader = val_loader
        self.steps = steps
        self.init_lr = lr
        self.optimizer = torch.optim.AdamW(self.diffusion.parameters(), lr=lr, weight_decay=weight_decay)
        self.device = device
        self.loss_history = pd.DataFrame(columns=['step', 'mloss', 'gloss', 'loss', 'val_mloss', 'val_gloss', 'val_loss'])
        self.log_every = 100
        self.print_every = 500
        self.ema_every = 1000
        self.val_every = val_every
        self.parent_dir = parent_dir
        self.best_val_loss = float('inf')

    def _anneal_lr(self, step):
        frac_done = step / self.steps
        lr = self.init_lr * (1 - frac_done)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _run_step(self, x, out_dict):
        x = x.to(self.device)
        for k in out_dict:
            out_dict[k] = out_dict[k].long().to(self.device)
        self.optimizer.zero_grad()
        loss_multi, loss_gauss = self.diffusion.mixed_loss(x, out_dict)
        loss = loss_multi + loss_gauss
        loss.backward()
        self.optimizer.step()

        return loss_multi, loss_gauss

    def _compute_val_loss(self):
        self.diffusion.eval()
        total_multi = 0.0
        total_gauss = 0.0
        total_count = 0
        with torch.no_grad():
            for x, y in self.val_loader:
                x = x.to(self.device)
                out_dict = {'y': y.long().to(self.device)}
                loss_multi, loss_gauss = self.diffusion.mixed_loss(x, out_dict)
                n = len(x)
                total_multi += loss_multi.item() * n
                total_gauss += loss_gauss.item() * n
                total_count += n
        self.diffusion.train()
        val_mloss = np.around(total_multi / total_count, 4)
        val_gloss = np.around(total_gauss / total_count, 4)
        return val_mloss, val_gloss

    def run_loop(self):
        step = 0
        curr_loss_multi = 0.0
        curr_loss_gauss = 0.0

        curr_count = 0
        val_mloss = float('nan')
        val_gloss = float('nan')

        while step < self.steps:
            x, out_dict = next(self.train_iter)
            out_dict = {'y': out_dict}
            batch_loss_multi, batch_loss_gauss = self._run_step(x, out_dict)

            self._anneal_lr(step)

            curr_count += len(x)
            curr_loss_multi += batch_loss_multi.item() * len(x)
            curr_loss_gauss += batch_loss_gauss.item() * len(x)

            if (step + 1) % self.log_every == 0:
                mloss = np.around(curr_loss_multi / curr_count, 4)
                gloss = np.around(curr_loss_gauss / curr_count, 4)

                # Compute validation loss at val_every intervals
                if self.val_loader is not None and (step + 1) % self.val_every == 0:
                    val_mloss, val_gloss = self._compute_val_loss()
                    val_total = val_mloss + val_gloss
                    if val_total < self.best_val_loss:
                        self.best_val_loss = val_total
                        if self.parent_dir is not None:
                            torch.save(
                                self.diffusion._denoise_fn.state_dict(),
                                os.path.join(self.parent_dir, 'model_best.pt')
                            )
                    print(f'Step {(step + 1)}/{self.steps} MLoss: {mloss} GLoss: {gloss} Sum: {mloss + gloss} | Val MLoss: {val_mloss} Val GLoss: {val_gloss} Val Sum: {val_total}')
                elif (step + 1) % self.print_every == 0:
                    print(f'Step {(step + 1)}/{self.steps} MLoss: {mloss} GLoss: {gloss} Sum: {mloss + gloss}')

                self.loss_history.loc[len(self.loss_history)] = [step + 1, mloss, gloss, mloss + gloss, val_mloss, val_gloss, val_mloss + val_gloss if not (np.isnan(val_mloss) or np.isnan(val_gloss)) else float('nan')]
                curr_count = 0
                curr_loss_gauss = 0.0
                curr_loss_multi = 0.0

            update_ema(self.ema_model.parameters(), self.diffusion._denoise_fn.parameters())

            step += 1

def train(
    parent_dir,
    real_data_path = 'data/higgs-small',
    steps = 1000,
    lr = 0.002,
    weight_decay = 1e-4,
    batch_size = 1024,
    model_type = 'mlp',
    model_params = None,
    num_timesteps = 1000,
    gaussian_loss_type = 'mse',
    scheduler = 'cosine',
    T_dict = None,
    num_numerical_features = 0,
    device = torch.device('cuda:1'),
    seed = 0,
    change_val = False
):
    real_data_path = os.path.normpath(real_data_path)
    parent_dir = os.path.normpath(parent_dir)

    zero.improve_reproducibility(seed)

    T = lib.Transformations(**T_dict)

    dataset = make_dataset(
        real_data_path,
        T,
        num_classes=model_params['num_classes'],
        is_y_cond=model_params['is_y_cond'],
        change_val=change_val
    )

    K = np.array(dataset.get_category_sizes('train'))
    if len(K) == 0 or T_dict['cat_encoding'] == 'one-hot':
        K = np.array([0])
    print(K)

    num_numerical_features = dataset.X_num['train'].shape[1] if dataset.X_num is not None else 0
    d_in = np.sum(K) + num_numerical_features
    model_params['d_in'] = d_in
    print(d_in)
    
    print(model_params)
    model = get_model(
        model_type,
        model_params,
        num_numerical_features,
        category_sizes=dataset.get_category_sizes('train')
    )
    model.to(device)

    # train_loader = lib.prepare_beton_loader(dataset, split='train', batch_size=batch_size)
    train_loader = lib.prepare_fast_dataloader(dataset, split='train', batch_size=batch_size)
    val_loader = lib.prepare_fast_torch_dataloader(dataset, split='val', batch_size=batch_size)

    diffusion = GaussianMultinomialDiffusion(
        num_classes=K,
        num_numerical_features=num_numerical_features,
        denoise_fn=model,
        gaussian_loss_type=gaussian_loss_type,
        num_timesteps=num_timesteps,
        scheduler=scheduler,
        device=device
    )
    diffusion.to(device)
    diffusion.train()

    trainer = Trainer(
        diffusion,
        train_loader,
        lr=lr,
        weight_decay=weight_decay,
        steps=steps,
        device=device,
        val_loader=val_loader,
        val_every=1000,
        parent_dir=parent_dir,
    )
    trainer.run_loop()

    trainer.loss_history.to_csv(os.path.join(parent_dir, 'loss.csv'), index=False)
    torch.save(diffusion._denoise_fn.state_dict(), os.path.join(parent_dir, 'model.pt'))
    torch.save(trainer.ema_model.state_dict(), os.path.join(parent_dir, 'model_ema.pt'))

    # If no validation checkpoint was saved (e.g. val_every never fired), fall back to final model
    best_path = os.path.join(parent_dir, 'model_best.pt')
    if not os.path.exists(best_path):
        torch.save(diffusion._denoise_fn.state_dict(), best_path)