# LoGo-Net
The codes are for the work "A Local and Global Feature Disentangled Network: Toward Classification of Benign-malignant Thyroid Nodules from Ultrasound Image" and are also available for download on our homepage https://www.neuro.uestc.edu.cn/vccl/home.html.

## Introduction
In this study, inspired by the domain knowledge of sonographers when diagnosing ultrasound images, a local and global feature disentangled network (LoGo-Net) is proposed to classify benign and malignant thyroid nodules. This model imitates the dual-pathway structure of human vision and establishes a new feature extraction method to improve the recognition performance of nodules. We use the tissue-anatomy disentangled (TAD) block to connect the dual pathways, which decouples the clues of local and global features based on the self-attention mechanism.
<p align="center">
  <img src="https://github.com/Phanzsx/LoGo-Net/blob/main/graph/model4.png" width="900" />
</p>
<p align="center">
  <img src="https://github.com/Phanzsx/LoGo-Net/blob/main/graph/module3.png" width="900" />
</p>

## Pre-requirements
The codebase is tested on the following setting.
* Python>=3.7
* PyTorch>=1.6.0
* torchvision>=0.7

## Train
* For easier use of LoGo-Net, this project provides a simple example framework. There are three scale models can choose from, which are logonet18, 34, and 50.
```
python train.py
```

## Citation
If you use this codes in your research, please cite the paper:
```BibTex
@ARTICLE{logonet,
  author={Zhao, Shi-Xuan and Chen, Yang and Yang, Kai-Fu and Yang, Kai-Fu and Luo, Yan and Ma, Bu-Yun and Li, Yong-Jie},
  journal={IEEE Transactions on Medical Imaging}, 
  title={A Local and Global Feature Disentangled Network: Toward Classification of Benign-malignant Thyroid Nodules from Ultrasound Image}, 
  year={2022},
  volume={4},
  number={6},
  pages={1497--1509},
  doi={10.1109/TMI.2022.3140797}}
```
