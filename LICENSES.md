# Third-Party Assets and Licenses

This document lists the external datasets, model architectures, and software
libraries used in this repository, along with their respective licenses.

## Datasets

- **CIFAR-100** (Krizhevsky, 2009).
  Downloaded automatically through `torchvision.datasets.CIFAR100`.
  Source: <https://www.cs.toronto.edu/~kriz/cifar.html>

## Model Architectures

- **ResNet-56** (He et al., "Deep Residual Learning for Image Recognition",
  CVPR 2016). The implementation in `slienet/models/resnet56.py` is an
  independent reimplementation written for this project.

## Software Libraries

| Library      | License                                       | Notes                |
|--------------|-----------------------------------------------|----------------------|
| PyTorch      | BSD-3-Clause                                  | Workstation training |
| torchvision  | BSD-3-Clause                                  | Datasets, transforms |
| NumPy        | BSD-3-Clause                                  | Numerical utilities  |
| ONNX         | Apache-2.0                                    | Deployment export    |
| TensorRT     | NVIDIA Software License Agreement             | Deployment only      |

## License of This Repository

The original code in this repository is released under the MIT License; see
`LICENSE` for the full text. Datasets and third-party libraries listed above
retain their respective licenses.
