# Auto-Prox: Training-Free Vision Transformer Architecture Search via Automatic Proxy Discovery

This repository contains the code implementation of Auto-Prox, a training-free vision transformer architecture search algorithm. Auto-Prox is designed to automatically discover the optimal architecture for vision transformers without the need for training.

## Overview
Auto-Prox utilizes a proxy-based approach to efficiently search through a large space of architectures and find the best one for a given task. It eliminates the need for manual architecture design and time-consuming training, making it a fast and efficient solution for vision transformer architecture search.

## Features
- Training-free architecture search for vision transformers
- Automatic proxy discovery for efficient architecture exploration
- Python implementation with functions for data preparation, model selection, and architecture search

## Usage
To use Auto-Prox, follow these steps:



**Step 1:** download datasets from their official websites. Imagenet is also supported.

- [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html)
- [Oxford Flowers](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)
- [Chaoyang](https://bupt-ai-cz.github.io/HSA-NRL/)

**Step 2:** move or link the datasets to `data/` directory. We show the layout of `data/` directory as follow:

```text
data
└── cifar-100-python
|   ├── meta
|   ├── test
|   └── train
└── flowers
|   ├── jpg
|   ├── imagelabels.mat
|   └── setid.mat
└── chaoyang
    ├── test
    ├── train
    ├── test.json
    └── train.json
```

## 

**Step 3:**  Run the `run_net.py` script to execute the Auto-Prox algorithm. This script contains examples and commands for running the architecture search.

## Citation

If you find Auto-Prox useful in your research, please consider citing the following paper:

```
@inproceedings{wei2024auto,
  title={Auto-prox: Training-free vision transformer architecture search via automatic proxy discovery},
  author={Wei, Zimian and Dong, Peijie and Hui, Zheng and Li, Anggeng and Li, Lujun and Lu, Menglong and Pan, Hengyue and Li, Dongsheng},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2024}
}
```

## License

This project is licensed under the [MIT License](LICENSE).

For more details and the source code, please visit the [Auto-Prox GitHub repository](https://github.com/username/repo).
