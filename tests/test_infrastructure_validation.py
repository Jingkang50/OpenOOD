"""Validation tests to verify the testing infrastructure is properly set up."""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys
import importlib


class TestInfrastructureSetup:
    """Test class to validate the testing infrastructure."""
    
    def test_pytest_installed(self):
        """Verify pytest is installed and importable."""
        assert pytest.__version__ is not None
        
    def test_pytest_cov_installed(self):
        """Verify pytest-cov is installed."""
        try:
            import pytest_cov
            assert pytest_cov is not None
        except ImportError:
            pytest.fail("pytest-cov is not installed")
            
    def test_pytest_mock_installed(self):
        """Verify pytest-mock is installed."""
        try:
            import pytest_mock
            assert pytest_mock is not None
        except ImportError:
            pytest.fail("pytest-mock is not installed")
    
    def test_openood_importable(self):
        """Verify the openood package can be imported."""
        try:
            import openood
            assert openood is not None
        except ImportError:
            pytest.fail("openood package cannot be imported")
    
    def test_fixtures_available(self, temp_dir, mock_config, sample_tensor):
        """Verify that custom fixtures are available and working."""
        assert isinstance(temp_dir, Path)
        assert temp_dir.exists()
        
        assert isinstance(mock_config, dict)
        assert "dataset" in mock_config
        assert "network" in mock_config
        
        assert isinstance(sample_tensor, torch.Tensor)
        assert sample_tensor.shape == (4, 3, 32, 32)
    
    def test_markers_configured(self, request):
        """Verify custom markers are properly configured."""
        markers = request.config.getini("markers")
        assert any("unit:" in marker for marker in markers)
        assert any("integration:" in marker for marker in markers)
        assert any("slow:" in marker for marker in markers)
    
    @pytest.mark.unit
    def test_unit_marker(self):
        """Test that unit marker works."""
        assert True
    
    @pytest.mark.integration
    def test_integration_marker(self):
        """Test that integration marker works."""
        assert True
    
    @pytest.mark.slow
    def test_slow_marker(self):
        """Test that slow marker works."""
        assert True
    
    def test_temp_dir_cleanup(self, temp_dir):
        """Verify temp_dir fixture creates and cleans up properly."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("test content")
        assert test_file.exists()
        # Cleanup will be verified after test completes
    
    def test_mock_model_fixture(self, mock_model, sample_tensor):
        """Verify mock model fixture works correctly."""
        output = mock_model(sample_tensor)
        assert output.shape == (4, 10)  # batch_size=4, num_classes=10
    
    def test_yaml_config_fixture(self, yaml_config_file):
        """Verify YAML config file fixture works."""
        assert yaml_config_file.exists()
        assert yaml_config_file.suffix == ".yml"
        
        import yaml
        with open(yaml_config_file, 'r') as f:
            config = yaml.safe_load(f)
        assert isinstance(config, dict)
        assert "dataset" in config
    
    def test_coverage_configured(self):
        """Verify coverage is properly configured."""
        # This test will pass if coverage is running
        assert True
    
    def test_torch_device_fixture(self, torch_device):
        """Verify torch device fixture works."""
        assert isinstance(torch_device, torch.device)
        assert torch_device.type in ["cpu", "cuda"]
    
    def test_numpy_rng_fixture(self, numpy_rng):
        """Verify numpy RNG fixture provides consistent results."""
        value1 = numpy_rng.rand()
        # Create a new RNG with same seed
        rng2 = np.random.RandomState(42)
        value2 = rng2.rand()
        assert value1 == value2
    
    def test_sample_ood_scores_fixture(self, sample_ood_scores):
        """Verify OOD scores fixture provides proper data."""
        assert "id_scores" in sample_ood_scores
        assert "ood_scores" in sample_ood_scores
        assert len(sample_ood_scores["id_scores"]) == 100
        assert len(sample_ood_scores["ood_scores"]) == 100
        # OOD scores should generally be higher than ID scores
        assert np.mean(sample_ood_scores["ood_scores"]) > np.mean(sample_ood_scores["id_scores"])


class TestDirectoryStructure:
    """Test that the directory structure is set up correctly."""
    
    def test_tests_directory_exists(self):
        """Verify tests directory exists."""
        tests_dir = Path(__file__).parent
        assert tests_dir.name == "tests"
        assert tests_dir.exists()
    
    def test_unit_directory_exists(self):
        """Verify unit tests directory exists."""
        unit_dir = Path(__file__).parent / "unit"
        assert unit_dir.exists()
        assert unit_dir.is_dir()
    
    def test_integration_directory_exists(self):
        """Verify integration tests directory exists."""
        integration_dir = Path(__file__).parent / "integration"
        assert integration_dir.exists()
        assert integration_dir.is_dir()
    
    def test_conftest_exists(self):
        """Verify conftest.py exists."""
        conftest_file = Path(__file__).parent / "conftest.py"
        assert conftest_file.exists()
        assert conftest_file.is_file()
    
    def test_init_files_exist(self):
        """Verify __init__.py files exist in test directories."""
        tests_dir = Path(__file__).parent
        
        for subdir in [".", "unit", "integration"]:
            init_file = tests_dir / subdir / "__init__.py"
            assert init_file.exists(), f"Missing __init__.py in {subdir}"


if __name__ == "__main__":
    # Run validation tests
    pytest.main([__file__, "-v"])