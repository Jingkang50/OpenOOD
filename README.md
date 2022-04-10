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


## Get Started

The easiest hands-on script is to train LeNet on MNIST.
```bash
sh scripts/_get_started/0_mnist_train.sh
```
Tutorials on understanding and contributing the codebase are provided in our documentation.

## Supported Methods
This part lists all the methods we reproduced in this codebase.
Organization/Indexing follows the structure of [our survey paper](https://arxiv.org/abs/2110.11334).
Results and models are available in the [model zoo](docs/model_zoo.md).


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
> - [x] [CROSR (CVPR'19)](https://nae-lab.org/~rei/research/crosr/)
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
> - [ ] [CONF (arXiv'18)](https://github.com/uoguelph-mlrg/confidence_estimation)
> - [ ] [G-ODIN (CVPR'20)](https://github.com/guyera/Generalized-ODIN-Implementation)
> - [ ] [Gram (ICML'20)](https://github.com/VectorInstitute/gram-ood-detection)
> - [ ] [DUQ (ICML'20)](https://github.com/y0ast/deterministic-uncertainty-quantification)
> - [ ] [CSI (NeurIPS'20)](https://github.com/alinlab/CSI)
> - [x] [EBO (NeurIPS'20)](https://github.com/wetliu/energy_ood)
> - [ ] [MOOD (CVPR'21)](https://github.com/deeplearning-wisc/MOOD)
> - [ ] [GradNorm (NeurIPS'21)](https://github.com/deeplearning-wisc/gradnorm_ood)
> - [ ] [ReAct (NeurIPS'21)](https://github.com/deeplearning-wisc/react)
> - [ ] [VOS (ICLR'22)](https://github.com/deeplearning-wisc/vos)

> With Extra Data:
> - [ ] [OE (ICLR'19)]()
> - [ ] [MCD (ICCV'19)]()
> - [ ] [UDG (ICCV'21)]()
</details>
