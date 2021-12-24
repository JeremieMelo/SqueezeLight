# SqueezeLight

By [Jiaqi Gu](https://github.com/JeremieMelo), [Chenghao Feng](https://github.com/Fengchenghao1996), Zheng Zhao, Zhoufeng Ying, Mingjie Liu, Ray T. Chen and David Z. Pan.

This repo is the official implementation of "[SqueezeLight: Towards Scalable Optical Neural Networks with Multi-Operand Ring Resonators](https://doi.org/10.23919/DATE51398.2021.9474147)".

# Introduction
**SqueezeLight** is a scalable optical neural network architecture based on multi-operand micro-ring resonators (MORRs), first experimentally demonstrated [here](https://aip.scitation.org/doi/abs/10.1063/1.5126517).
**SqueezeLight** is constructed with a nonlinear optical neuron based on multi-operand ring resonators to squeeze vector dot-product into a single device with low wavelength usage and built-in nonlinearity.
A block-level squeezing technique with structured sparsity is exploited to support higher scalability via a fine-grained structured pruning technique.
**SqueezeLight** achieves one-order-of-magnitude improvement in compactness and efficiency over previous MRR-based ONN designs with high fidelity, trainability, and robustness.

![flow](figures/SqueezeLightArch.png)
<!-- ![teaser](figures/teaser.png) -->
# Dependencies
* Python >= 3.6
* pyutils >= 0.0.1. See [pyutils](https://github.com/JeremieMelo/pyutility) for installation.
* pytorch-onn >= 0.0.2. See [pytorch-onn](https://github.com/JeremieMelo/pytorch-onn) for installation.
* Python libraries listed in `requirements.txt`
* NVIDIA GPUs and CUDA >= 10.2

# Structures
* core/
    * models/
        * layers/
            * morr_conv2d and morr_linear layers
            * activation.py: activation function
        * morr_\*.py: SqueezeLight model definition
        * morr_base.py: base SqueezeLight model definition
    * builder.py: build training utilities
* script/: contains experiment scripts
* train.py: training logic

# Usage
* Train SqueezeLight CNN model on MNIST dataset.\
`> python3 train.py config/mnist/cnn/train/train.yml`
or
`> nohup python3 script/mnist/cnn/launch_train.py &`

* Prune MORR in SqueezeLight CNN model on MNIST dataset.\
`> python3 train.py config/mnist/cnn/train/prune.yml`
or
`> nohup python3 script/mnist/cnn/launch_prune.py &`

# Citing SqueezeLight
```
@inproceedings{gu2021SqueezeLight,
  title={SqueezeLight: Towards Scalable Optical Neural Networks with Multi-Operand Ring Resonators},
  author={Jiaqi Gu and Chenghao Feng and Zheng Zhao and Zhoufeng Ying and Ray T. Chen and David Z. Pan},
  booktitle={Design, Automation and Test in Europe Conference (DATE)},
  year={2021}
}
```

# Related Papers
 * Jiaqi Gu, Chenghao Feng, Zheng Zhao, Zhoufeng Ying, Mingjie Liu, Ray T. Chen, David Z. Pan, "SqueezeLight: SqueezeLight: Towards Scalable Optical Neural Networks with Multi-Operand Ring Resonators," IEEE Design, Automation & Test in Europe Conference & Exhibition (DATE), 2021. \[[paper](https://doi.org/10.23919/DATE51398.2021.9474147)\]
 * Yunfei Sun, "Design and optimization of multi-operand ring resonator based efficient optical recurrent neural network", Master Thesis, 2021. \[[paper](https://repositories.lib.utexas.edu/handle/2152/87414)\]
 * Zhoufeng Ying, Chenghao Feng, Zheng Zhao, Richard Soref, David Z. Pan, Ray T. Chen, "Integrated multi-operand electro-optic logic gates for optical computing," Applied Physics Letters (APL), 2019. \[[paper](https://aip.scitation.org/doi/abs/10.1063/1.5126517)\]
 * Zhoufeng Ying, Chenghao Feng, Zheng Zhao, David Z. Pan, Ray T. Chen, "Multi-Operand Directed Logic-Based Electro-Optic Gates for Ultracompact Optical Computing,"  IEEE Photonics Conference (IPC), 2019. \[[paper](https://ieeexplore.ieee.org/document/8908348)\]
