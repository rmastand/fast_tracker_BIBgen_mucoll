# TabDDPM integration

## 1. Code

- Official repository: <https://github.com/yandex-research/tab-ddpm>
- Pinned commit: `b476257dd460b778ba09eb97f7a51d6490fa17f8`
- License: MIT
- Official source: `diffusion/tabddpm_official/`

Project-specific data loading, training, sampling, and evaluation are
implemented in `02_generate_samples.py` and `helpers/`. The official source is
included in full, with the changes documented below.

## 2. Environment

Reference environment: Linux, Python 3.13.5, PyTorch 2.10.0+cu128, and CUDA 12.8.

Create and activate an isolated environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install PyTorch and the remaining packages:

```bash
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install \
  numpy==2.4.2 pandas==3.0.2 scipy==1.17.1 scikit-learn==1.8.0 \
  matplotlib==3.10.8 pyyaml==6.0.3 tqdm==4.67.3 xgboost==3.2.0 \
  numba==0.64.0 zuko==1.6.0 wandb==0.25.1 \
  category-encoders==2.3.0 icecream==2.1.2 \
  tomli==1.2.2 tomli-w==0.4.0 pynvml==11.5.3
python -m pip install --no-deps libzero==0.0.8 rtdl==0.0.9
```

The integration is compatible with the versions above; dependency warnings
from `libzero` and `rtdl` can be ignored.

The original TabDDPM environment used Python 3.9.7, PyTorch 1.10.1+cu111, and
scikit-learn 1.0.2. Two official-source lines are changed:

- `diffusion/tabddpm_official/lib/data.py:221`:
  `subsample=1e9` -> `subsample=int(1e9)` for modern scikit-learn.
- `diffusion/tabddpm_official/scripts/sample.py:79`:
  `D.y['train']` -> `np.concatenate([D.y['train'], D.y['val']])` so Flow and
  TabDDPM sample from the same full context distribution.

All other official source code remains unmodified.
