---
name: Adding leaderboard entry
about: Like the title suggests
title: "[leaderboard]"
labels: ''
assignees: ''

---

Hi, thank you for adding your method to our OpenOOD leaderboard.

Please first make sure that you have evaluated your method with the latest OpenOOD benchmarks and evaluator, just so we can have a straight comparison between all methods.

Essentially, the leaderboard expects below information for each entry:

- **`Training`**: The training method of your model, e.g., `CrossEntropy`.
- **`Postprocessor`**: The postprocessor of your model, e.g., `MSP`, `ReAct`, etc.
- **`Near-OOD AUROC`**: The AUROC score of your model on the near-OOD split.
- **`Far-OOD AUROC`**: The AUROC score of your model on the far-OOD split.
- **`ID Accuracy`**: The accuracy of your model on the ID test data.
- **`Outlier Data`**: Whether your model uses the outlier data for training.
- **`Model Arch.`**: The architecture of your base classifier, e.g., `ResNet18`.
- **`Additional Description`**: Any additional description of your model, e.g., `100 epochs`, `torchvision pretrained`, etc.

The easiest steps to provide the information would be as follows:

1. Upload the output csv files that contain evaluation results in this issue. They should already include `Near/Far-OOD AUROC` and `ID Accuracy`. The `Model Arch` is currently predefined for each ID dataset (e.g., CIFAR-10 will have a ResNet-18), but let us know if you are using a different model backbone.
2. Let us know about `Training`, `Postprocessor`, `Outlier Data`, and `Additional Description`.

Then you are all set! Our maintainers will include your entries in the leaderboard as soon as possible.
