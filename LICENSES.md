# Third-Party Assets and Licenses

This repository builds on the following third-party assets. We thank the authors
and maintainers.

## Datasets

### CIFAR-100
- **Source:** Krizhevsky, A. (2009). *Learning Multiple Layers of Features from
  Tiny Images.* Technical Report, University of Toronto.
- **URL:** https://www.cs.toronto.edu/~kriz/cifar.html
- **License:** Distributed for research use; no formal license file. Cite the
  technical report when using.

## Model Architectures (re-implemented from published papers)

### ResNet-56
- **Source:** He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual
  Learning for Image Recognition.* CVPR.
- **Notes:** Our implementation in `models/resnet56.py` follows the original
  CIFAR variant (3 stages × 9 residual blocks) from the paper.

### VGG-16-BN
- **Source:** Simonyan, K., & Zisserman, A. (2015). *Very Deep Convolutional
  Networks for Large-Scale Image Recognition.* ICLR.

### MobileNetV1
- **Source:** Howard, A. G., Zhu, M., Chen, B., Kalenichenko, D., Wang, W.,
  Weyand, T., Andreetto, M., & Adam, H. (2017). *MobileNets: Efficient
  Convolutional Neural Networks for Mobile Vision Applications.* arXiv:1704.04861.

## Baseline Methods (re-implemented for comparison)

### Shallow-Deep Networks (SDN)
- **Source:** Kaya, Y., Hong, S., & Dumitras, T. (2019). *Shallow-Deep Networks:
  Understanding and Mitigating Network Overthinking.* ICML, PMLR 97:3301–3310.

### Zero-Time Waste (ZTW)
- **Source:** Wołczyk, M., Wójcik, B., Bałazy, K., Podolak, I. T., Tabor, J.,
  Śmieja, M., & Trzciński, T. (2021). *Zero Time Waste: Recycling Predictions
  in Early Exit Neural Networks.* NeurIPS 34, 2516–2528.

## Frameworks and Tools

### PyTorch / torchvision
- **License:** BSD 3-Clause License
- **URL:** https://github.com/pytorch/pytorch

### NVIDIA TensorRT
- **License:** Used under the NVIDIA TensorRT SDK License Agreement (standard
  developer terms).
- **URL:** https://developer.nvidia.com/tensorrt

## This Repository

The original code in this repository is released under the MIT License
(see `LICENSE`).
