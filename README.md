# OpenOOD: Benchmarking Generalized OOD Detection

This repository includes representative methods within the `Generalized Out-of-Distribution Detection Framework` that proposed
in [our survey paper](https://arxiv.org/abs/2110.11334).

Topics of anomaly detection, novelty detection, open set recognition,
and out-of-distribution detection
are within the scope of this codebase.

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

> - [ ] [DeepSVDD (ICML'18)](https://github.com/lukasruff/Deep-SVDD-PyTorch)
> - [x] [KDAD (arXiv'20)]()
> - [x] [CutPaste (CVPR'2021)]()
> - [x] [PatchCore (arXiv'2021)]()
> - [x] [DRÆM (ICCV'21)]()
</details>


<details open>
<summary><b>Open Set Recognition</b></summary>

> - [ ] [OpenMax (CVPR'16)](https://github.com/13952522076/Open-Set-Recognition)
> - [ ] [G-OpenMax (BMVC'17)](https://github.com/lwneal/counterfactual-open-set/blob/master/generativeopenset/gen_openmax.py)
> - [ ] [OSRCI (ECCV'18)](https://github.com/lwneal/counterfactual-open-set/)
> - [ ] [CROSR (CVPR'19)](https://nae-lab.org/~rei/research/crosr/)
> - [ ] [OLTR (CVPR'19)](https://liuziwei7.github.io/projects/LongTail.html)
> - [x] [RPL (ECCV'20)]()
> - [x] [MOS (CVPR'21)]()
> - [x] [OpenGAN (ICCV'21)](https://github.com/aimerykong/OpenGAN/tree/main/utils)
</details>


<details open>
<summary><b>Out-of-Distribution Detection</b></summary>

> No Extra Data:
> - [x] [MSP (ICLR'17)]()
> - [x] [ODIN (ICLR'18)]()
> - [x] [MDS (NeurIPS'18)]()
> - [ ] [Likelihood Ratio (NeurIPS'19)]()
> - [ ] [G-ODIN (CVPR'20)]()
> - [ ] [Gram (ICML'20)]()
> - [ ] [DUQ (ICML'20)](https://github.com/y0ast/deterministic-uncertainty-quantification)
> - [ ] [CSI (NeurIPS'20)](https://github.com/alinlab/CSI)
> - [x] [EBO (NeurIPS'20)]()
> - [x] [SEM]()

> With Extra Data:
> - [x] [OE (ICLR'19)]()
> - [ ] [BGC (arXiv'21)]()
> - [x] [UDG (ICCV'21)]()
</details>
