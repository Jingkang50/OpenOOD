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

> - [ ] OCC ()
> - [ ] DeepSVDD ()
> - [x] KDAD (ArXiv'20)
> - [x] CutPaste (CVPR'2021)
> - [x] PatchCore (ArXiv'2021)
> - [x] DRÆM (ICCV'21)
</details>


<details open>
<summary><b>Open Set Recognition</b></summary>

> - [ ] OpenMax
> - [ ] G-OpenMax
> - [x] OSRCI
> - [x] C2AE
> - [ ] CROSR
> - [x] RPL
> - [x] MOS
> - [x] OpenGAN
</details>


<details open>
<summary><b>Out-of-Distribution Detection</b></summary>

> No Extra Data:
> - [x] MSP
> - [x] ODIN
> - [x] MDS
> - [ ] Likelihood Ratio
> - [ ] G-ODIN
> - [ ] Gram
> - [ ] DUQ
> - [x] EBO
> - [x] SEM

> With Extra Data:
> - [x] OE
> - [ ] BGC
> - [x] UDG
</details>
