# PINN-LB: Physics-Informed Neural Networks for the Leray-Burgers Equation

This repository contains the implementation of Physics-Informed Neural Networks (PINNs) for parameter estimation and adaptive solution of the Leray-Burgers equation, as presented in our research paper.

## 📄 Publication

**Parameter estimation and adaptive solution of the Leray-Burgers equation using physics-informed neural networks**

*Authors:* DooSeok Lee, Yuncherl Choi, Bong-Sik Kim

*Journal:* Results in Applied Mathematics, Volume 27 (2025), Article 100619

**📖 [Read the full paper](https://www.sciencedirect.com/science/article/pii/S2590037425000834?via%3Dihub)**

## 🔬 Abstract

This study presents a unified framework that integrates physics-informed neural networks (PINNs) to address both the inverse and forward problems of the one-dimensional Leray-Burgers equation. We investigate the inverse problem by empirically determining a physically consistent range of the characteristic wavelength parameter α, and solve the forward problem using a PINN architecture where α is dynamically optimized during training via our dedicated Alpha2Net subnetwork.

## ✨ Key Features

- **Inverse Problem Solution**: Systematic determination of physically consistent α parameter ranges
- **Alpha2Net Architecture**: Novel subnetwork for dynamic α optimization with physical constraints  
- **Unified Framework**: Seamless integration of inverse and forward problem solutions
- **Traffic State Estimation**: Real-world application demonstrating practical utility
- **Comprehensive Analysis**: Comparison with viscous Burgers equation and convergence studies

## 🛠️ Requirements

### System Dependencies
The code has been tested on **Ubuntu 24.04.1 LTS**. You must install TeX Live fonts outside of the Python environment:

```bash
sudo apt-get -qq install texlive-fonts-recommended texlive-fonts-extra cm-super dvipng
```

### Python Environment
All required Python packages and dependencies are specified in the `env-TF2N-ubuntu.yaml` file. Create the conda environment using:

```bash
conda env create -f env-TF2N-ubuntu.yaml
conda activate pinn-lb
```

## 🚀 Quick Start

1. **Clone the repository:**
   ```bash
   git clone https://github.com/bkimo/PINN-LB.git
   cd PINN-LB
   ```

2. **Set up the environment:**
   ```bash
   # Install system dependencies
   sudo apt-get -qq install texlive-fonts-recommended texlive-fonts-extra cm-super dvipng
   
   # Create conda environment
   conda env create -f env-TF2N-ubuntu.yaml
   conda activate pinn-lb
   ```

3. **Run the experiments:**
   ```bash
   # Add specific commands to run your main experiments here
   python main_experiment.py
   ```

## 📁 Repository Structure

```
PINN-LB/
├── README.md
├── env-TF2N-ubuntu.yaml    # Environment configuration
├── src/                    # Source code
│   ├── inverse_problem/    # Inverse problem implementation
│   ├── forward_problem/    # Forward problem with Alpha2Net
│   └── traffic_estimation/ # Traffic state estimation application
├── experiments/            # Experimental scripts and notebooks
├── data/                   # Dataset files
└── results/                # Output results and figures
```

## 🎯 Main Contributions

1. **Parameter Range Identification**: Empirically determined α ranges (0.01-0.05 for continuous profiles, 0.01-0.03 for discontinuous profiles)

2. **Alpha2Net Innovation**: Novel subnetwork architecture that dynamically learns optimal α(t) while maintaining physical constraints

3. **Unified Methodology**: Seamless integration of inverse problem findings into forward problem solutions

4. **Practical Application**: Demonstrated efficiency in Traffic State Estimation with 2x speed improvement over traditional methods

## 🔗 Related Resources

- **Paper URL**: https://doi.org/10.1016/j.rinam.2025.100619
- **Traffic Data**: [PISE Dataset](https://github.com/arjhuang/pise)
- **Journal**: [Results in Applied Mathematics](https://www.journals.elsevier.com/results-in-applied-mathematics)

## 📊 Results Highlights

- Successfully captures shock and rarefaction waves in the Leray-Burgers equation
- Achieves L₂ errors on the order of 10⁻² to 10⁻³ for various initial conditions
- Demonstrates superior computational efficiency in traffic state estimation applications
- Validates the physical consistency of learned α parameters across different scenarios

## 📝 Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{kim2025parameter,
    title={Parameter estimation and adaptive solution of the Leray-Burgers equation using physics-informed neural networks},
    author={Lee, DooSeok and Choi, Yuncherl and Kim, Bong-Sik},
    journal={Results in Applied Mathematics},
    volume={27},
    pages={100619},
    year={2025},
    publisher={Elsevier},
    doi={10.1016/j.rinam.2025.100619}
}
```

## 👥 Authors

- **DooSeok Lee** - Daegu Gyeongbuk Institute of Science and Technology
- **Yuncherl Choi** - Kwangwoon University  
- **Bong-Sik Kim** - American University of Ras Al Khaimah

## 🤝 Contributing

We welcome contributions! Please feel free to submit issues or pull requests.

## 📄 License

This project is licensed under the CC BY-NC License - see the paper's license terms for details.

---

*For questions about the implementation or paper, please open an issue or contact the corresponding authors.*