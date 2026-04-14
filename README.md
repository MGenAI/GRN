# GRN: Generative Refinement Networks

[![arXiv](https://img.shields.io/badge/arXiv%20paper-xxxx-b31b1b.svg)](https://arxiv.org/abs/xxxx)&nbsp;

This is a PyTorch implementation of the paper [Generative Refinement Networks](https://arxiv.org/abs/xxxx):

<!-- ```
@article{li2025GRN,
  title={Back to Basics: Let Denoising Generative Models Denoise},
  author={Li, Tianhong and He, Kaiming},
  journal={arXiv preprint arXiv:2511.13720},
  year={2025}
}
``` -->
### Introduction
<p align="center">
  <img src="demo/framework.jpg" width="100%">
</p>


<p align="center">
  <img src="demo/c2i_examples.jpg" width="100%">
</p>

<p align="center">
  <img src="demo/t2i_examples.jpg" width="100%">
</p>



## 📑 Open-Source Plan
GRN adopts a minimalist and self-contained design. This implementation is in PyTorch+GPU.
  - [] GRN T2V Checkpoints
  - [] GRN T2V Inference Code
  - [x] GRN T2V Training Code
  - [] GRN T2I Checkpoints
  - [] GRN T2I Inference Code
  - [x] GRN T2I Training Code
  - [] GRN C2I Checkpoints
  - [x] GRN C2I Inference Code
  - [x] GRN C2I Training Code



## Installation

Download the code: 
```
git clone https://github.com/MGenAI/GRN
cd GRN
```

A suitable [conda](https://conda.io/) environment named `GRN` can be created and activated with:

```
conda env create -f environment.yaml
conda activate GRN
```

If you get ```undefined symbol: iJIT_NotifyEvent``` when importing ```torch```, simply
```
pip uninstall torch
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
```
Check this [issue](https://github.com/conda/conda/issues/13812#issuecomment-2071445372) for more details.


## class-to-image
### Dataset
Download [ImageNet](http://image-net.org/download) dataset, and place it in your `IMAGENET_PATH`.

### Training
Example script for training GRN_ind_B on ImageNet 256x256 for 600 epochs, we suggest using 8x80GB GPUs:
```
bash scripts/c2i/train_GRN_ind_B.sh
```
We suggest using 8x80GB GPUs for training.

Example script for training GRN_bit_B on ImageNet 256x256 for 600 epochs, we suggest using 8x80GB vRAM GPUs:
```
bash scripts/c2i/train_GRN_bit_B.sh
```
We suggest using 8x80GB GPUs for training.

Example script for training GRN_ind_L on ImageNet 256x256 for 600 epochs, we suggest using 8x80GB  vRAM GPUs:
```
bash scripts/c2i/train_GRN_ind_L.sh
```

Example script for training GRN_ind_H on ImageNet 256x256 for 600 epochs, we suggest using 16x80GB  vRAM GPUs:
```
bash scripts/c2i/train_GRN_ind_H.sh
```

Example script for training GRN_ind_G on ImageNet 256x256 for 600 epochs, we suggest using 32x80GB  vRAM GPUs:
```
bash scripts/c2i/train_GRN_ind_G.sh
```



### Evaluation

PyTorch pre-trained models are available [here]().

Evaluate pre-trained GRN-ind-B, we suggest using 8x80GB vRAM GPUs:
```
bash scripts/c2i/eval_GRN_ind_B.sh
```

Evaluate pre-trained GRN-bit-B, we suggest using 8x80GB vRAM GPUs:
```
bash scripts/c2i/eval_GRN_bit_B.sh
```

Evaluate pre-trained GRN-ind-L, we suggest using 8x80GB vRAM GPUs:
```
bash scripts/c2i/eval_GRN_ind_L.sh
```

Evaluate pre-trained GRN-ind-H, we suggest using 8x80GB vRAM GPUs:
```
bash scripts/c2i/eval_GRN_ind_H.sh
```

Evaluate pre-trained GRN-ind-G, we suggest using 8x80GB vRAM GPUs:
```
bash scripts/c2i/eval_GRN_ind_G.sh
```

We use [```torch-fidelity```](https://github.com/LTH14/torch-fidelity)
to evaluate FID and IS against a reference image folder or statistics. We use the JiT's pre-computed reference stats under ```grn/utils_c2i/fid_stats```.
