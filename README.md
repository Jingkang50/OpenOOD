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


To setup the environment, we use `conda` to manage our dependencies, and CUDA 10.1 to run our experiments.

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
│   ├── covid_images
│   └── imglist
├── openood
├── scripts
├── main.py
├── ...
```

The easiest hands-on script is to train LeNet on MNIST and evaluate its OOD or FS-OOD performance with MSP baseline.
```bash
sh scripts/0_basics/mnist_train.sh
sh scripts/c_ood/0_mnist_test_ood_msp.sh
sh scripts/c_ood/0_mnist_test_fsood_msp.sh
```


[More tutorials](https://github.com/Jingkang50/OpenOOD/wiki/Get-Started) are provided in our wiki pages.

## Supported Methods
This part lists all the methods we include in this codebase.


<details open>
<summary><b>Anomaly Detection</b></summary>

> - [x] [DeepSVDD (ICML'18)](https://github.com/lukasruff/Deep-SVDD-PyTorch)
> - [x] [KDAD (arXiv'20)]()
> - [x] [CutPaste (CVPR'2021)]()
> - [x] [PatchCore (arXiv'2021)]()
> - [x] [DRÆM (ICCV'21)]()
</details>


<details open>
<summary><b>Open Set Recognition</b></summary>

> - [x] [OpenMax (CVPR'16)](https://github.com/13952522076/Open-Set-Recognition)
> - [ ] [CROSR (CVPR'19)](https://nae-lab.org/~rei/research/crosr/) (@OmegaDING in progress)
> - [ ] [ARPL (TPAMI'21)](https://github.com/iCGY96/ARPL)
> - [x] [OpenGAN (ICCV'21)](https://github.com/aimerykong/OpenGAN/tree/main/utils)
</details>


<details open>
<summary><b>Out-of-Distribution Detection</b></summary>

> No Extra Data:
> - [x] [MSP (ICLR'17)]()
> - [x] [ODIN (ICLR'18)]()
> - [x] [MDS (NeurIPS'18)]()
> - [ ] [CONF (arXiv'18)](https://github.com/uoguelph-mlrg/confidence_estimation) (@JediWarriorZou in progress)
> - [ ] [G-ODIN (CVPR'20)](https://github.com/guyera/Generalized-ODIN-Implementation) (@Prophet-C in progress)
> - [ ] [Gram (ICML'20)](https://github.com/VectorInstitute/gram-ood-detection) (@Zzitang in progress)
> - [ ] [DUQ (ICML'20)](https://github.com/y0ast/deterministic-uncertainty-quantification) (@Zzitang in progress)
> - [ ] [CSI (NeurIPS'20)](https://github.com/alinlab/CSI) (@Prophet-C in progress)
> - [x] [EBO (NeurIPS'20)](https://github.com/wetliu/energy_ood)
> - [ ] [MOS (CVPR'21)](https://github.com/deeplearning-wisc/large_scale_ood)
> - [ ] [MOOD (CVPR'21)](https://github.com/deeplearning-wisc/MOOD)
> - [ ] [GradNorm (NeurIPS'21)](https://github.com/deeplearning-wisc/gradnorm_ood) (@haoqiwang in progress)
> - [x] [ReAct (NeurIPS'21)](https://github.com/deeplearning-wisc/react)
> - [ ] [VOS (ICLR'22)](https://github.com/deeplearning-wisc/vos)
> - [ ] [VIM (CVPR'22)](https://ooddetection.github.io/) (@haoqiwang in progress)
> - [x] [SEM (arXiv'22)](https://arxiv.org/abs/2204.05306)

> With Extra Data:
> - [ ] [OE (ICLR'19)]()
> - [ ] [MCD (ICCV'19)]()
> - [ ] [UDG (ICCV'21)]()
</details>

## Citation
If you find our repository useful for your research, please consider citing our paper:
```bibtex
@article{yang2022openood,
    author = {Yang {\textit{et al.}}, Jingkang},
    title = {OpenOOD: Benchmarking Generalized Out-of-Distribution Detection},
    year = {2022}
}

@article{yang2022fsood,
    author = {Yang, Jingkang and Zhou, Kaiyang and Liu, Ziwei},
    title = {Full-Spectrum Out-of-Distribution Detection},
    journal={arXiv preprint arXiv:2204.05306},
    year = {2022}
}

@article{yang2021oodsurvey,
  title={Generalized Out-of-Distribution Detection: A Survey},
  author={Yang, Jingkang and Zhou, Kaiyang and Li, Yixuan and Liu, Ziwei},
  journal={arXiv preprint arXiv:2110.11334},
  year={2021}
}
```
