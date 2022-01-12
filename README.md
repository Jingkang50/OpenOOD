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

### 3. Anomaly Detection

<details open>
<summary><b>3.1 Density-based Method:</b></summary>

> - [x] [KDAD (ArXiv'2020)](https://github.com/rohban-lab/Knowledge_Distillation_AD)
> - [x] [CutPaste (CVPR'2021)](https://arxiv.org/abs/2104.04015)
> - [x] [PatchCore (ArXiv'2021)](https://arxiv.org/pdf/2106.08265.pdf)
> - [x] [DRÆM (ICCV'2021)](https://openaccess.thecvf.com/content/ICCV2021/papers/Zavrtanik_DRAEM_-_A_Discriminatively_Trained_Reconstruction_Embedding_for_Surface_Anomaly_ICCV_2021_paper.pdf)
</details>

<details open>
<summary><b>3.2 Reconstruction-based Method:</b></summary>

> - [x] [GPND]()
> - [x] [CoRA]()
> - [x] [MemAE]()
> - [x] [SCADN]()
</details>

<details open>
<summary><b>3.3 Classification-based Method:</b></summary>

> - [x] [OCC]()
> - [x] [DeepSVDD]()
> - [x] [CSI]()
> - [x] [GOAD]()
</details>

<details open>
<summary><b>3.4 Distance-based Method:</b></summary>

> - [x] [DBSCAN]()
</details>

<details open>
<summary><b>3.5 Gradient-based Method:</b></summary>

> - [x] [GradAD]()
</details>


### 4. Open Set Recognition

<details open>
<summary><b>4.1 Classification-based Method:</b></summary>

> - **4.1.1 EVT-based Calibration**
>   - [x] [OpenMax (ArXiv'2020)](https://github.com/rohban-lab/Knowledge_Distillation_AD)
>
> - **4.1.2 EVT-free Calibration**
>   - [x] [OSRCI]()
>
> - **4.1.3 Unknown Generation**
>   - [x] [G-OpenMax (CVPR'2021)](https://arxiv.org/abs/2104.04015)
>
> - **4.1.4 Label Space Redesign**
>   - [x] [MOS]()
>
</details>


<details open>
<summary><b>4.2 Distance-based Method:</b></summary>

> - [x] [RPL](https://github.com/rohban-lab/Knowledge_Distillation_AD)
> - [x] [PEELER](https://arxiv.org/abs/2104.04015)
</details>

<details open>
<summary><b>4.3 Reconstruction-based Method:</b></summary>

> - [x] [OpenMax (ArXiv'2020)](https://github.com/rohban-lab/Knowledge_Distillation_AD)
> - [x] [G-OpenMax (CVPR'2021)](https://arxiv.org/abs/2104.04015)
</details>


### 5. Out-of-Distribution Detection

<details open>
<summary><b>5.1 Classification-based Method:</b></summary>

> - **5.1.1 Output-based Methods:**
>   - **5.1.1.a Post-hoc Calibration:**
>     - [x] [MSP (ArXiv'2020)](https://github.com/rohban-lab/Knowledge_Distillation_AD)
>     - [x] [ODIN (CVPR'2021)](https://arxiv.org/abs/2104.04015)
>     - [x] [Likelihood Ratio (ICCV'2021)]()
>   - **5.1.1.b Confidence Enhancement:**
>     - [x] [MDS (ArXiv'2021)](https://arxiv.org/pdf/2106.08265.pdf)
>   - **5.1.1.c Outlier Exposure:**
>     - [x] [OE (ICCV'2021)]()
>     - [x] [UDG (ICCV'2021)]()
> - **5.1.2 OOD Data Generation:**
>   - [x] []()
> - **5.1.3 Gradient-based Method:**
>   - [x] []()
> - **5.1.4 Bayesian Models:**
>   - [x] [Gram (ICCV'2021)]()
>   - [x] [DUQ (ICCV'2021)]()
> - **5.1.5 Large-scale OOD Detection:**
>   - [x] []()
</details>

<details open>
<summary><b>5.2 Density-based Methods:</b></summary>

> - [x] []()
</details>


<details open>
<summary><b>5.3 Distance-based Methods:</b></summary>

> - [x] [MDS (ArXiv'2021)](https://arxiv.org/pdf/2106.08265.pdf)
> - [x] [SEM (CVPR'2022)](https://arxiv.org/pdf/2106.08265.pdf)
</details>
