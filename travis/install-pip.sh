#!/usr/bin/env bash

set -e

# Alternative to "conda init bash"
source "$HOME/miniconda/etc/profile.d/conda.sh"

conda activate test
hash -r

if [[ "$TRAVIS_OS_NAME" == 'linux' ]]; then
  echo "install and upgrade PyTorch"
  conda install --yes pytorch cpuonly -c pytorch
  echo "pip installing required python packages"
  pip install -r requirements/dev.txt
  python -c "import torch; print(f'PyTorch version = {torch.__version__}')"
elif [[ "$TRAVIS_OS_NAME" == 'osx' ]]; then
  echo "install XGboost via conda-forge due to univieCUBE/phenotrex #23"
  conda install -c conda-forge xgboost
  echo "pip installing required python packages"
  pip install -r requirements/dev.txt
fi

python --version
