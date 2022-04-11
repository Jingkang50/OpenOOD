# OpenOOD: Benchmarking Generalized OOD Detection

This repository includes representative methods within the `Generalized Out-of-Distribution Detection Framework` that proposed
in [our survey paper](https://arxiv.org/abs/2110.11334).

Topics of anomaly detection, novelty detection, open set recognition,
and out-of-distribution detection
are within the scope of this codebase.

This codebase is still under construction. Comments, issues, and contributions are all welcomed!

| ![timeline.jpg](assets/timeline.jpg) |
|:--:|
| <b>Image from [Fig.3 in our survey](https://arxiv.org/abs/2110.11334) - Timeline for representative methodologies.</b>|


## Updates
- **12 April, 2022**: Primary release to support Full-Spectrum OOD Detection.

## Get Started

The easiest hands-on script is to train LeNet on MNIST and evaluate its OOD or FS-OOD performance with MSP baseline.
```bash
sh scripts/0_basics/mnist_train.sh
sh scripts/c_ood/0_mnist_test_ood_msp.sh
sh scripts/c_ood/0_mnist_test_fsood_msp.sh
```


[Tutorials](https://github.com/Jingkang50/OpenOOD/wiki) on understanding and contributing the codebase are provided in our wiki pages.

## Supported Methods
This part lists all the methods we reproduced in this codebase.
Organization/Indexing follows the structure of [our survey paper](https://arxiv.org/abs/2110.11334).


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
> - [ ] [CROSR (CVPR'19)](https://nae-lab.org/~rei/research/crosr/)
> - [ ] [ARPL (TPAMI'21)](https://github.com/iCGY96/ARPL)
> - [ ] [MOS (CVPR'21)](https://github.com/deeplearning-wisc/large_scale_ood)
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
> - [ ] [MOOD (CVPR'21)](https://github.com/deeplearning-wisc/MOOD)
> - [ ] [GradNorm (NeurIPS'21)](https://github.com/deeplearning-wisc/gradnorm_ood)
> - [x] [ReAct (NeurIPS'21)](https://github.com/deeplearning-wisc/react)
> - [ ] [VOS (ICLR'22)](https://github.com/deeplearning-wisc/vos)
> - [ ] [VIM (CVPR'22)](https://ooddetection.github.io/) (@haoqiwang in progress)
> - [x] [SEM (arXiv'22)]()

> With Extra Data:
> - [ ] [OE (ICLR'19)]()
> - [ ] [MCD (ICCV'19)]()
> - [ ] [UDG (ICCV'21)]()
</details>
