# OpenOOD leaderboard

**Leaderboard website**: https://zjysteven.github.io/OpenOOD/

**Paper:** [OpenOOD v1.5: Enhanced Benchmark for Out-of-Distribution Detection](https://arxiv.org/abs/2306.09301)


## Main idea
  
`OpenOOD` aims to provide accurate, standardized, and unified evaluation of OOD detection.
There are at least 100+ works on OOD detection in the past 6 years, but it is still unclear which approaches really work since the evaluation setup is highly inconsistent from paper to paper.
OpenOOD currently provides 6 benchmarks for OOD detection (4 for standard setting and 2 for full-spectrum setting) in the context of image classification and benchmarks 40 advanced methodologies within our framework.
We expect OpenOOD to foster collective efforts in the community towards advancing the state-of-the-art in OOD detection.

## How to add entries to the leaderboard?

1. Fork this repo
2. Add your paper (title and a link) to `utils/papers.csv`
3. Add your results to the corresponding result file `results/[ID Dataset]_[OOD scheme].csv`, e.g., `results/imagenet1k_ood.csv`
4. Open a Pull Request

For step 2 and 3, there are already plenty of examples. Please make sure to follow the format/style.

## Contributions
Contributions both to the leaderboard and our [codebase](https://github.com/Jingkang50/OpenOOD/) are very welcome, as well as any suggestions for improving the project! We would be happy to hear any feedback on how to make it better and more comprehensive.


## Acknowledgement
The leaderboard webpage is adapted from [RobustBench](https://robustbench.github.io/).
