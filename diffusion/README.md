# TabDDPM integration

## 1. Code

- Official repository: <https://github.com/yandex-research/tab-ddpm>
- Pinned commit: `b476257dd460b778ba09eb97f7a51d6490fa17f8`
- License: MIT
- Official source: `diffusion/tabddpm_official/`

Project-specific data loading, training, sampling, and evaluation are
implemented in `run_zuko.py` and `helpers/`. The official source is included
in full, with one compatibility change documented below.

## 2. Environment

Reference environment: Linux, Python 3.13.5, and PyTorch 2.11.0+cu128.

Create and activate an isolated environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install PyTorch 2.11.0 with a CUDA build supported by the target machine, then
install the remaining packages:

```bash
python -m pip install \
  numpy==2.4.3 pandas==3.0.2 scipy==1.17.1 scikit-learn==1.8.0 \
  matplotlib==3.10.8 pyyaml==6.0.3 tqdm==4.67.3 xgboost==3.2.0 \
  numba==0.65.0 zuko==1.6.0 wandb==0.26.0 \
  category-encoders==2.9.0 icecream==2.2.0 libzero==0.0.8 \
  rtdl==0.0.13 tomli==2.4.1 tomli-w==1.2.0
```

The original TabDDPM environment used Python 3.9.7, PyTorch 1.10.1+cu111, and
scikit-learn 1.0.2. For modern scikit-learn,
`diffusion/tabddpm_official/lib/data.py:221` changes `subsample=1e9` to
`subsample=int(1e9)`. The value is unchanged, and all other official source
code remains unmodified. The original dependency pins are preserved in
`tabddpm_official/requirements.txt`.
