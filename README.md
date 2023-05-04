# Website of RobustBench: a standardized adversarial robustness benchmark

**Leaderboard website**: [https://robustbench.github.io/](https://robustbench.github.io/)

**Model Zoo**: [https://github.com/RobustBench/robustbench](https://github.com/RobustBench/robustbench)

**Paper:** [https://arxiv.org/abs/2010.09670](https://arxiv.org/abs/2010.09670)


## Main idea
  
The goal of **`RobustBench`** is to systematically track the *real* progress in adversarial robustness. 
There are already [more than 3'000 papers](https://nicholas.carlini.com/writing/2019/all-adversarial-example-papers.html) 
on this topic, but it is still unclear which approaches really work and which only lead to [overestimated robustness](https://arxiv.org/abs/1802.00420).
We start from benchmarking the Linf, L2, and common corruption robustness since these are the most studied settings in the literature. 

Evaluation of the robustness to Lp perturbations *in general* is not straightforward and requires adaptive attacks ([Tramer et al., (2020)](https://arxiv.org/abs/2002.08347)).
Thus, in order to establish a reliable *standardized* benchmark, we need to impose some restrictions on the defenses we consider.
In particular, **we accept only defenses that are (1) have in general non-zero gradients wrt the inputs, (2) have a fully deterministic forward pass (i.e. no randomness) that
(3) does not have an optimization loop.** Often, defenses that violate these 3 principles only make gradient-based attacks 
harder but do not substantially improve robustness ([Carlini et al., (2019)](https://arxiv.org/abs/1902.06705)) except those
that can present concrete provable guarantees (e.g. [Cohen et al., (2019)](https://arxiv.org/abs/1902.02918)).

To prevent potential overadaptation of new defenses to AutoAttack, we also welcome external evaluations based on **adaptive attacks**, especially where AutoAttack [flags](https://github.com/fra31/auto-attack/blob/master/flags_doc.md) a potential overestimation of robustness. For each model, we are interested in the best known robust accuracy and see AutoAttack and adaptive attacks as complementary to each other.


## Contributions
Contributions both to the website and [model zoo](https://github.com/RobustBench/robustbench) are very welcome, as well as any suggestions for improving the project! We would be happy to hear any feedback on how to make it better and more comprehensive.
