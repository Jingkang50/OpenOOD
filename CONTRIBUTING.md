## Contributing to OpenOOD

All kinds of contributions are welcome, including but not limited to the following.

- Integrate more methods under generalized OOD detection
- Fix typo or bugs
- Add new features and components

### Workflow

1. fork and pull the latest OpenOOD repository
2. checkout a new branch (do not use master branch for PRs)
3. commit your changes
4. create a PR

```{note}
If you plan to add some new features that involve large changes, it is encouraged to open an issue for discussion first.
```
### Code style

#### Python

We adopt [PEP8](https://www.python.org/dev/peps/pep-0008/) as the preferred code style.

We use the following tools for linting and formatting:

- [flake8](http://flake8.pycqa.org/en/latest/): A wrapper around some linter tools.
- [yapf](https://github.com/google/yapf): A formatter for Python files.
- [isort](https://github.com/timothycrosley/isort): A Python utility to sort imports.
- [markdownlint](https://github.com/markdownlint/markdownlint): A linter to check markdown files and flag style issues.
- [docformatter](https://github.com/myint/docformatter): A formatter to format docstring.

Style configurations of yapf and isort can be found in [setup.cfg](./setup.cfg).

We use [pre-commit hook](https://pre-commit.com/) that checks and formats for `flake8`, `yapf`, `isort`, `trailing whitespaces`, `markdown files`,
fixes `end-of-files`, `double-quoted-strings`, `python-encoding-pragma`, `mixed-line-ending`, sorts `requirments.txt` automatically on every commit.
The config for a pre-commit hook is stored in [.pre-commit-config](./.pre-commit-config.yaml).

After you clone the repository, you will need to install initialize pre-commit hook.

```shell
pip install -U pre-commit
```

From the repository folder

```shell
pre-commit install
```

## Contributing to OpenOOD leaderboard

We welcome new entries submitted to the leaderboard. Please follow the instructions below to submit your results.

1. Evaluate your model/method with OpenOOD's benchmark and evaluator such that the comparison is fair.

2. Report your new results by opening an issue. Remember to specify the following information:

- **`Training`**: The training method of your model, e.g., `CrossEntropy`.
- **`Postprocessor`**: The postprocessor of your model, e.g., `MSP`, `ReAct`, etc.
- **`Near-OOD AUROC`**: The AUROC score of your model on the near-OOD split.
- **`Far-OOD AUROC`**: The AUROC score of your model on the far-OOD split.
- **`ID Accuracy`**: The accuracy of your model on the ID test data.
- **`Outlier Data`**: Whether your model uses the outlier data for training.
- **`Model Arch.`**: The architecture of your base classifier, e.g., `ResNet18`.
- **`Additional Description`**: Any additional description of your model, e.g., `100 epochs`, `torchvision pretrained`, etc.

3. Ideally, send us a copy of your model checkpoint so that we can verify your results on our end. You can either upload the checkpoint to a cloud storage and share the link in the issue, or send us an email at [jz288@duke.edu](mailto:jz288@duke.edu).
