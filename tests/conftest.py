"""Shared pytest fixtures and configuration for OpenOOD tests."""

import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List, Generator
import pytest
import torch
import numpy as np
import yaml
import json


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def mock_config() -> Dict[str, Any]:
    """Provide a mock configuration dictionary for testing."""
    return {
        "dataset": {
            "name": "cifar10",
            "data_dir": "/tmp/data",
            "batch_size": 32,
            "num_workers": 2,
        },
        "network": {
            "name": "resnet18",
            "pretrained": False,
            "num_classes": 10,
        },
        "optimizer": {
            "name": "sgd",
            "lr": 0.1,
            "momentum": 0.9,
            "weight_decay": 5e-4,
        },
        "scheduler": {
            "name": "cosine",
            "num_epochs": 100,
        },
        "trainer": {
            "name": "base",
            "save_dir": "/tmp/checkpoints",
            "save_epoch": 10,
        },
        "evaluator": {
            "name": "ood",
            "save_output": True,
        },
    }


@pytest.fixture
def sample_tensor() -> torch.Tensor:
    """Provide a sample tensor for testing."""
    return torch.randn(4, 3, 32, 32)


@pytest.fixture
def sample_labels() -> torch.Tensor:
    """Provide sample labels for testing."""
    return torch.randint(0, 10, (4,))


@pytest.fixture
def sample_features() -> torch.Tensor:
    """Provide sample feature vectors for testing."""
    return torch.randn(4, 512)


@pytest.fixture
def mock_model():
    """Provide a mock neural network model."""
    class MockModel(torch.nn.Module):
        def __init__(self, num_classes=10):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 64, 3, padding=1)
            self.pool = torch.nn.AdaptiveAvgPool2d(1)
            self.fc = torch.nn.Linear(64, num_classes)
            
        def forward(self, x):
            x = self.conv(x)
            x = torch.nn.functional.relu(x)
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)
    
    return MockModel()


@pytest.fixture
def sample_dataset_config(temp_dir: Path) -> Dict[str, Any]:
    """Create a sample dataset configuration."""
    data_dir = temp_dir / "data"
    data_dir.mkdir(exist_ok=True)
    
    # Create mock image list files
    train_list = data_dir / "train.txt"
    val_list = data_dir / "val.txt"
    test_list = data_dir / "test.txt"
    
    # Write sample image paths
    for file_path, num_samples in [(train_list, 100), (val_list, 20), (test_list, 20)]:
        with open(file_path, 'w') as f:
            for i in range(num_samples):
                f.write(f"image_{i}.jpg {i % 10}\n")
    
    return {
        "data_dir": str(data_dir),
        "train_list": str(train_list),
        "val_list": str(val_list),
        "test_list": str(test_list),
        "num_classes": 10,
    }


@pytest.fixture
def yaml_config_file(temp_dir: Path, mock_config: Dict[str, Any]) -> Path:
    """Create a temporary YAML configuration file."""
    config_path = temp_dir / "config.yml"
    with open(config_path, 'w') as f:
        yaml.dump(mock_config, f)
    return config_path


@pytest.fixture
def json_config_file(temp_dir: Path, mock_config: Dict[str, Any]) -> Path:
    """Create a temporary JSON configuration file."""
    config_path = temp_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(mock_config, f)
    return config_path


@pytest.fixture
def mock_checkpoint(temp_dir: Path, mock_model) -> Path:
    """Create a mock model checkpoint file."""
    checkpoint_path = temp_dir / "checkpoint.pth"
    torch.save({
        'epoch': 50,
        'model_state_dict': mock_model.state_dict(),
        'optimizer_state_dict': {},
        'loss': 0.123,
    }, checkpoint_path)
    return checkpoint_path


@pytest.fixture
def sample_metrics() -> Dict[str, float]:
    """Provide sample evaluation metrics."""
    return {
        "accuracy": 0.95,
        "auroc": 0.89,
        "aupr_in": 0.91,
        "aupr_out": 0.87,
        "fpr95": 0.15,
    }


@pytest.fixture
def mock_dataloader(sample_tensor: torch.Tensor, sample_labels: torch.Tensor):
    """Create a mock dataloader for testing."""
    from torch.utils.data import TensorDataset, DataLoader
    
    dataset = TensorDataset(sample_tensor, sample_labels)
    return DataLoader(dataset, batch_size=2, shuffle=False)


@pytest.fixture
def numpy_rng():
    """Provide a seeded numpy random number generator."""
    return np.random.RandomState(42)


@pytest.fixture
def torch_device():
    """Provide the appropriate torch device (CPU/GPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(autouse=True)
def reset_torch_seed():
    """Reset torch random seed before each test."""
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


@pytest.fixture
def sample_ood_scores() -> Dict[str, np.ndarray]:
    """Provide sample OOD detection scores."""
    num_id = 100
    num_ood = 100
    
    # Generate scores where ID samples have lower scores than OOD
    id_scores = np.random.normal(0, 1, num_id)
    ood_scores = np.random.normal(2, 1, num_ood)
    
    return {
        "id_scores": id_scores,
        "ood_scores": ood_scores,
        "id_labels": np.zeros(num_id),
        "ood_labels": np.ones(num_ood),
    }


@pytest.fixture
def mock_logger(temp_dir: Path):
    """Create a mock logger for testing."""
    import logging
    
    logger = logging.getLogger("test_logger")
    logger.setLevel(logging.DEBUG)
    
    # Add file handler
    log_file = temp_dir / "test.log"
    handler = logging.FileHandler(log_file)
    handler.setLevel(logging.DEBUG)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger


@pytest.fixture
def sample_transform():
    """Provide a sample image transformation pipeline."""
    from torchvision import transforms
    
    return transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


@pytest.fixture
def mock_args():
    """Provide mock command-line arguments."""
    class MockArgs:
        config = "configs/test_config.yml"
        dataset = "cifar10"
        network = "resnet18"
        batch_size = 32
        num_workers = 2
        gpu = "0"
        seed = 42
        output_dir = "/tmp/output"
        checkpoint = None
        verbose = False
        
    return MockArgs()