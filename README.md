# PINN-LB
This repository contains codes to reproduce the experiments from the paper:
- Estimation of the Characteristic Wavelength Parameter in 1D Leray-Burgers Equation with Physics-Informed Neural Network (https://arxiv.org/abs/2310.08874)

### Requirements
- The file `env-TF2-ubuntu-cpu.yaml` lists all the packages and dependencies needed to run the code on a CPU under Ubuntu 24.04.1 LTS.
- The file `env-TF2-ubuntu-gpu.yaml` lists all the packages and dependencies needed to run the code on a GPU under Ubuntu 22.04.5 LTS.
- In addition to the Python packages, you need to install TeX Live Fonts. Run the following command outside of the Python virtual environment:
  <pre> >> sudo apt-get -qq install texlive-fonts-recommended texlive-fonts-extra cm-super dvipng </pre>
