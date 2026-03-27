# Physics-Informed Neural Networks for Continuum Robots: Towards Fast Approximation of Static Cosserat Rod Theory

Code to the ICRA 2024 publication: DOI 10.1109/ICRA57147.2024.10610742

```bibtex 
  @INPROCEEDINGS{10610742, 
author={Bensch, Martin and Job, Tim-David and Habich, Tim-Lukas and Seel, Thomas and Schappler, Moritz},
booktitle={2024 IEEE International Conference on Robotics and Automation (ICRA)},
title={Physics-Informed Neural Networks for Continuum Robots: Towards Fast Approximation of Static Cosserat Rod Theory},
year={2024},
pages={17293-17299},
doi={10.1109/ICRA57147.2024.10610742}
}
```


This code uses the Cosserat rod statics C++ implementation from 
https://github.com/SvenLilge/tdcr-modeling
through a pybind11 binding.

---

## Installation from a fresh clone

This repository depends on a C++/pybind11 package located in
`Required/tdcr-lilge-binding` (`pytdcrsv` + `tdrpyb` extension).

### Prerequisites

- Python `>=3.9`
- A C++ compiler (`clang` or `gcc`)
- `gsl` and `eigen3` installed on your system

macOS (Homebrew):

```bash
brew install gsl eigen
```

Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install -y libgsl-dev libeigen3-dev
```

### Quickstart

```bash
git clone <your-fork-or-upstream-url>
cd tdcr-pinn
bash scripts/bootstrap.sh
```

The bootstrap script installs:
1. `pytdcrsv` from `Required/tdcr-lilge-binding`
2. `tdcrpinn` in editable mode with development extras

### Verify installation

```bash
make smoke
```

Expected output includes `smoke-ok`.

## Alternative manual install

```bash
python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install -e ./Required/tdcr-lilge-binding
python3 -m pip install -e ".[dev]"
```

## Troubleshooting

- If linker errors mention `gsl`/`gslcblas`, verify the library is installed.
- If headers for `Eigen` are not found, install `eigen3`.
- You can provide custom locations with:
  - `EIGEN_INCLUDE_DIRS=/path/one:/path/two`
  - `GSL_LIBRARY_DIRS=/path/one:/path/two`
