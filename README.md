# TASIF
The source code for our Paper [**"Time-Aware Adaptive Side Information Fusion for Sequential Recommendation"**]

## Overview
Overall framework of TASIF.

![avatar](TASIF.png)

## Preparation

We train and evaluate our TASIF using a NVIDIA GeForce RTX 4090 GPU with 24 GB memory, where the CUDA version is 12.5 and python version is 3.8.20. Our code is based on PyTorch 2.4.1, and requires the following python packages:

> + numpy==1.24.4
> + scipy==1.10.1
> + torch==2.4.1
> + tensorboard==2.14.0


## Usage

Download datasets from [RecSysDatasets](https://github.com/RUCAIBox/RecSysDatasets) or their [Google Drive](https://drive.google.com/drive/folders/1ahiLmzU7cGRPXf5qGMqtAChte2eYp9gI). And put the files in `./dataset/` like the following.

```
$ tree
.
├── Amazon_Beauty
│   ├── Amazon_Beauty.inter
│   └── Amazon_Beauty.item
├── Amazon_Toys_and_Games
│   ├── Amazon_Toys_and_Games.inter
│   └── Amazon_Toys_and_Games.item
├── Amazon_Sports_and_Outdoors
│   ├── Amazon_Sports_and_Outdoors.inter
│   └── Amazon_Sports_and_Outdoors.item
└── yelp
    ├── README.md
    ├── yelp.inter
    ├── yelp.item
    └── yelp.user

```

Run `TASIF.sh`.

## Acknowledgement
This repository is based on [RecBole](https://github.com/RUCAIBox/RecBole) and [DIF-SR](https://github.com/AIM-SE/DIF-SR).