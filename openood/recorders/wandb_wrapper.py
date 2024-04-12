
from numbers import Real
from openood.recorders.recorder import RecorderProtocol
import torch.nn as nn


def get_metadata(model: nn.Module) -> dict:
    """
    Get metadata of a PyTorch model.

    Args:
        model (nn.Module): The PyTorch model.

    Returns:
        dict: A dictionary containing the metadata of the model.
    """
    return dict((k, getattr(model, k)) for k in vars(model) if not k.startswith('_'))


class WandbWrapper:
    def __init__(self, recorder: RecorderProtocol, args: dict | str | None = None):
        self._recorder = recorder
        self.output_dir = self._recorder.output_dir
        try:
            import wandb
            self._wandb = wandb
        except ImportError:
            raise ImportError(
                "To use the Weights and Biases Logger please install wandb."
                "Run `pip install wandb` to install it."
            )

        # Initialize a W&B run 
        if self._wandb.run is None:
            self._wandb.init(
                project=args.project,
                config=args
            )

    def report(self, train_metrics: dict[str, Real], val_metrics: dict[str, Real]):
        """Log train/val metrics onto W&B."""

        # Log current epoch
        for k, v in train_metrics.items():
            self._wandb.log({f'Train/{k}': v}, commit=False)

        for k, v in val_metrics.items():
            self._wandb.log({f'Val/{k}': v}, commit=False)
        
        self._wandb.log({})

        return self._recorder.report(train_metrics, val_metrics)

    def save_model(self, net: nn.Module, val_metrics: dict[str, Real]):
        output = self._recorder.save_model(net, val_metrics)      
        model_artifact = self._wandb.Artifact(
            self._wandb.run.id + "_models", 
            type="model",
            metadata=get_metadata(net)
        )
        model_artifact.add_dir(self.output_dir)
        self._wandb.log_artifact(model_artifact)
        return output

    def summary(self):
        return self._recorder.summary()
