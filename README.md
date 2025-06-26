# Trusted Multi-view Learning under Noisy Supervision

This work is the journal extension of our paper, "Trusted Multi-view Learning with label noise," which will be published at IJCAI 2024. In this version, we have improved and extended the original method and conducted more comprehensive experimental validation. The manuscript is currently under submission and peer review.

**Journal Version**: This repository contains the complete implementation code for multi-modal/multi-view learning for the journal submission version.

**Conference Version**: If you are interested in our prior work from the conference, please visit its corresponding code repository: [https://github.com/YilinZhang107/TMNR](https://github.com/YilinZhang107/TMNR)

## Repository Structure

Please download the UPMC-Food-101 multi-modal dataset from this address: [https://www.kaggle.com/datasets/gianmarco96/upmcfood101/data](https://www.kaggle.com/datasets/gianmarco96/upmcfood101/data)

Other files required for training can be downloaded from the following link. Please place them according to the structure below: [https://drive.google.com/drive/folders/1sIgbPFMjvmITzHdwDLCyd9XUYwb3TM5i?usp=sharing](https://drive.google.com/drive/folders/1sIgbPFMjvmITzHdwDLCyd9XUYwb3TM5i?usp=sharing)

```
TMNRR/
├── Multi-modal_Released/
│   ├── datasets/
│   │   ├── food101/        # noisy labels
│   │   └── UPMC_Food101/   # img and text dataset
│   ├── checkpoint/
│   │   ├── train_img_encoder.pth
│   │   └── train_text_encoder.pth
│   ├── utils/
│   │   ├── dataset.py
│   │   ├── helpers.py
│   │   ├── util.py
│   │   └── vocab.py
│   ├── encoder.py
│   ├── model.py
│   ├── loss.py
│   ├── extractFeature.py
│   ├── findAndCalibration.py
│   └── main.py
│
└── Multi-view_Released/
    └── ......
```
Multi-view datasets can be found in the conference version of the repository


# Trusted Multi-view Learning under Noisy Supervision

本文是我们将发表在 IJCAI 2024上的论文 Trusted Multi-view Learning with label noise 的期刊扩展版本。我们对原有方法进行了改进和扩展，并进行了更全面的实验验证。稿件仍在投稿和评审过程中。

**期刊版本**：本仓库包含的是期刊投稿版本的完整多模态/视角学习实现代码。

**会议版本**：如果您对我们前期的会议版本工作感兴趣，请访问其对应的代码仓库：https://github.com/YilinZhang107/TMNR



# Repository Structure

请在该地址下载UPMC_Food101多模态数据集:https://www.kaggle.com/datasets/gianmarco96/upmcfood101/data

训练所需的其他文件请在该链接下载后按如下结构放置：https://drive.google.com/drive/folders/1sIgbPFMjvmITzHdwDLCyd9XUYwb3TM5i?usp=sharing 

```
TMNRR/
├── Multi-modal_Released/         
│   ├── datasets/      
│   │   ├── food101/     # noisy labels
│   │   └── UPMC_Food101/ # img and text dataset
│   ├── checkpoint/               
│   │   ├── train_img_encoder.pth
│   │   └── train_text_encoder.pth
│   ├── utils/                 
│   │   ├── dataset.py          
│   │   ├── helpers.py           
│   │   ├── util.py             
│   │   └── vocab.py           
│   ├── encoder.py               
│   ├── model.py                
│   ├── loss.py                 
│   ├── extractFeature.py       
│   ├── findAndCalibration.py    
│   └── main.py                 
│
├── Multi-view_Released/          
│   └── ......
             
```

多视角数据集请见会议版本仓库


