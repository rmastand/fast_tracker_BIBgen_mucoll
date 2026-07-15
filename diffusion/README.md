# TabDDPM integration

This project uses the official implementation of **TabDDPM: Modelling
Tabular Data with Diffusion Models**.

- Upstream repository: https://github.com/yandex-research/tab-ddpm
- Pinned upstream commit: `b476257dd460b778ba09eb97f7a51d6490fa17f8`
- Upstream license: MIT

The complete upstream repository is included under `tabddpm_official/`. Its
contents are kept unchanged from the pinned commit. Muon-collider-specific
data preparation, configuration, training orchestration, sampling, and
evaluation must be implemented outside `tabddpm_official/`.

Users do not need to download TabDDPM separately. Training and evaluation for
this project should be launched through the project-level entry point:

```bash
python run_zuko.py [options]
```

The exact TabDDPM command-line options and reproducibility configurations will
be documented here after the integration is complete.
