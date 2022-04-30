# OpenOOD: Benchmarking Generalized OOD Detection

This repository reproduces representative methods within the [`Generalized Out-of-Distribution Detection Framework`](https://arxiv.org/abs/2110.11334),
aiming to make a fair comparison across methods that initially developed for anomaly detection, novelty detection, open set recognition, and out-of-distribution detection.
This codebase is still under construction.
Comments, issues, contributions, and collaborations are all welcomed!

| ![timeline.jpg](assets/timeline.jpg) |
|:--:|
| <b>Image from [Fig.3 in our survey](https://arxiv.org/abs/2110.11334) - Timeline for representative methodologies.</b>|


## Updates
- **12 April, 2022**: Primary release to support [Full-Spectrum OOD Detection](https://arxiv.org/abs/2204.05306).

## Get Started


To setup the environment, we use `conda` to manage our dependencies.

Our developers use `CUDA 10.1` to do experiments.

You can specify the appropriate `cudatoolkit` version to install on your machine in the `environment.yml` file, and then run the following to create the `conda` environment:
```bash
conda env create -f environment.yml
conda activate openood
```

Datasets are provided [here](https://entuedu-my.sharepoint.com/:f:/g/personal/jingkang001_e_ntu_edu_sg/Eso7IDKUKQ9AoY7hm9IU2gIBMWNnWGCYPwClpH0TASRLmg?e=iEYhXO).
Our codebase accesses the datasets from `./data/` by default.
```
├── ...
├── data
│   ├── images
│   └── imglist
├── openood
├── scripts
├── main.py
├── ...
```

The easiest hands-on script is to train LeNet-5 on MNIST and evaluate its OOD or FS-OOD performance with MSP baseline.
```bash
sh scripts/0_basics/mnist_train.sh
sh scripts/c_ood/0_mnist_test_ood_msp.sh
sh scripts/c_ood/0_mnist_test_fsood_msp.sh
```


[More tutorials](https://github.com/Jingkang50/OpenOOD/wiki/Get-Started) are provided in our [wiki](https://github.com/Jingkang50/OpenOOD/wiki) pages.


---
## Supported Benchmarks
This part lists all the benchmarks we


<details open>
<summary><b>Anomaly Detection (1)</b></summary>

> - [x] [MVTec-AD](https://www.mvtec.com/company/research/datasets/mvtec-ad)
</details>

<details open>
<summary><b>Open Set Recognition (2)</b></summary>

> - [x] [TinyImageNet-20/180]()
> - [x] [ImageNet1K / ImageNet21K-P]()
</details>

<details open>
<summary><b>Out-of-Distribution Detection (4)</b></summary>

> - [x] [COVID]() <br>
>      > ID: `BIMCV`; (CS-ID: `ActMed`, `Hannover`;)<br>
>      > Near-OOD: `CT-SCAN`, `X-Ray-Bones`; <br>
>      > Far-OOD: `MNIST`, `CIFAR-10`, `Texture`, `Tiny-ImageNet`;
> - [x] [DIGITS]()
>      > ID: `MNIST`; (CS-ID: `USPS`, `SVHN`;)<br>
>      > Near-OOD: `NotMNIST`, `FashionMNIST`;<br>
>      > Far-OOD: `Texture`, `CIFAR-10`, `Tiny-ImageNet`, `Places-365`;
> - [x] [OBJECTS]()
>      > ID: `CIFAR-10`; (CS-ID: `CIFAR-10-C`, `ImageNet-10`;)<br>
>      > Near-OOD: `CIFAR-100`, `Tiny-ImageNet`;<br>
>      > Far-OOD: `MNIST`, `FashionMNIST`, `Texture`, `CIFAR-100-C`;
> - [x] [IMAGENET]()
>      > ID: `ImageNet-1K`; (CS-ID: `ImageNet-C`, `ImageNet-v2`)<br>
>      > OOD: `Species`, `iNaturalist`, `ImageNet-O`, `OpenImage-O`, `Texture`;
</details>



---
## Supported Backbones
This part lists all the backbones we will support in our codebase, including CNN-based and Transformer-based models. Backbones like ResNet-50 and Transformer have ImageNet-1K/22K pretrained models.

<details open>
<summary><b>CNN-based Backbones (4)</b></summary>

> - [x] [LeNet-5](http://yann.lecun.com/exdb/lenet/)
> - [x] [ResNet-18](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html)
> - [x] [WideResNet-28](https://arxiv.org/abs/1605.07146)
> - [x] [ResNet-50](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html) ([BiT](https://github.com/google-research/big_transfer))
</details>


<details open>
<summary><b>Transformer-based Architectures (3)</b></summary>

> - [x] [ViT](https://github.com/google-research/vision_transformer)
> - [x] [DeiT](https://github.com/facebookresearch/deit)
> - [x] [Swin Transformer](https://openaccess.thecvf.com/content/ICCV2021/html/Liu_Swin_Transformer_Hierarchical_Vision_Transformer_Using_Shifted_Windows_ICCV_2021_paper.html)
</details>

---
## Supported Methods
This part lists all the methods we include in this codebase. In `v0.5`, we totally support **more than 30 popular methods** for generalized OOD detection.


<details open>
<summary><b>Anomaly Detection (5)</b></summary>

> - [x] [DeepSVDD (ICML'18)](https://github.com/lukasruff/Deep-SVDD-PyTorch)
> - [x] [KDAD (arXiv'20)]()
> - [x] [CutPaste (CVPR'2021)]()
> - [x] [PatchCore (arXiv'2021)]()
> - [x] [DRÆM (ICCV'21)]()
</details>


<details open>
<summary><b>Open Set Recognition (3)</b></summary>

> - [x] [OpenMax (CVPR'16)](https://github.com/13952522076/Open-Set-Recognition)
> - [x] [ARPL (TPAMI'21)](https://github.com/iCGY96/ARPL)
> - [x] [OpenGAN (ICCV'21)](https://github.com/aimerykong/OpenGAN/tree/main/utils)
</details>


<details open>
<summary><b>Out-of-Distribution Detection (18)</b></summary>

> No Extra Data (15):
> - [x] [MSP (ICLR'17)]()
> - [x] [ODIN (ICLR'18)]()
> - [x] [MDS (NeurIPS'18)]()
> - [x] [ConfBranch (arXiv'18)](https://github.com/uoguelph-mlrg/confidence_estimation)
> - [x] [G-ODIN (CVPR'20)](https://github.com/guyera/Generalized-ODIN-Implementation)
> - [x] [Gram (ICML'20)](https://github.com/VectorInstitute/gram-ood-detection)
> - [ ] [DUQ (ICML'20)](https://github.com/y0ast/deterministic-uncertainty-quantification) (@Zzitang in progress)
> - [ ] [CSI (NeurIPS'20)](https://github.com/alinlab/CSI) (@Prophet-C in progress)
> - [x] [EBO (NeurIPS'20)](https://github.com/wetliu/energy_ood)
> - [ ] [MOS (CVPR'21)](https://github.com/deeplearning-wisc/large_scale_ood) (@OmegaDING in progress)
> - [x] [GradNorm (NeurIPS'21)](https://github.com/deeplearning-wisc/gradnorm_ood)
> - [x] [ReAct (NeurIPS'21)](https://github.com/deeplearning-wisc/react)
> - [ ] [VOS (ICLR'22)](https://github.com/deeplearning-wisc/vos) (@JediWarriorZou in progress)
> - [x] [VIM (CVPR'22)](https://ooddetection.github.io/)
> - [x] [SEM (arXiv'22)](https://arxiv.org/abs/2204.05306)
> - [x] [MLS (arXiv'22)](https://github.com/hendrycks/anomaly-seg)

> With Extra Data (3):
> - [x] [OE (ICLR'19)]()
> - [x] [MCD (ICCV'19)]()
> - [x] [UDG (ICCV'21)]()
</details>


<details open>
<summary><b>Other Methods on Robustness and Uncertainty (6)</b></summary>

> - [ ] [MCDropout (ICML'16)]()
> - [ ] [DeepEnsemble (NeurIPS'17)]()
> - [ ] [TempScale (ICML'17)]()
> - [ ] [Mixup (ICLR'18)]()
> - [ ] [AugMix (ICLR'20)]()
> - [ ] [PixMix (CVPR'21)]()
</details>

---
## Citation
If you find our repository useful for your research, please consider citing our paper:
```bibtex
@article{yang2022openood,
    author = {Yang, Jingkang and {\textit{et al.}}},
    title = {OpenOOD: Benchmarking Generalized Out-of-Distribution Detection},
    year = {2022}
}

@article{yang2022fsood,
    title = {Full-Spectrum Out-of-Distribution Detection},
    author = {Yang, Jingkang and Zhou, Kaiyang and Liu, Ziwei},
    journal={arXiv preprint arXiv:2204.05306},
    year = {2022}
}

@article{yang2021oodsurvey,
    title={Generalized Out-of-Distribution Detection: A Survey},
    author={Yang, Jingkang and Zhou, Kaiyang and Li, Yixuan and Liu, Ziwei},
    journal={arXiv preprint arXiv:2110.11334},
    year={2021}
}

@InProceedings{yang2021scood,
    author = {Yang, Jingkang and Wang, Haoqi and Feng, Litong and Yan, Xiaopeng and Zheng, Huabin and Zhang, Wayne and Liu, Ziwei},
    title = {Semantically Coherent Out-of-Distribution Detection},
    booktitle = {Proceedings of the IEEE International Conference on Computer Vision},
    year = {2021}
}
```
