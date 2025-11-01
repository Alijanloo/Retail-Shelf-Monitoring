"""
Unit tests for PyTorchTensorRTModel.

Note: These tests require PyTorch and torch_tensorrt to be installed.
Some tests may be skipped if CUDA is not available.
"""

import unittest
from unittest.mock import Mock, patch

import numpy as np

from retail_shelf_monitoring.frameworks.inference_engines.pytorch_tensorrt import (
    PyTorchTensorRTModel,
)
from retail_shelf_monitoring.usecases.interfaces.inference_model import (  # noqa: E501
    InferenceModel,
)

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class TestPyTorchTensorRTModel(unittest.TestCase):
    """Test cases for PyTorchTensorRTModel class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")

        # Create a simple test model
        self.test_model = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(16, 10),
        )
        self.input_shape = (3, 32, 32)
        self.batch_size = 4

    @patch(
        "retail_shelf_monitoring.frameworks.inference_engines."
        "pytorch_tensorrt.torch_tensorrt"
    )
    def test_init_with_pytorch_model(self, mock_torch_tensorrt):
        """Test initialization with PyTorch model."""

        # Mock torch_tensorrt.compile to avoid actual compilation
        mock_torch_tensorrt.compile = Mock(return_value=self.test_model)
        mock_torch_tensorrt.Input = Mock()

        model = PyTorchTensorRTModel(
            pytorch_model=self.test_model,
            input_shape=self.input_shape,
            device="cpu",  # Use CPU to avoid CUDA requirements
            optimize_for_inference=True,
        )

        self.assertIsNotNone(model.model)
        self.assertEqual(model.input_shape, self.input_shape)
        self.assertEqual(str(model.device), "cpu")

    def test_init_without_model_raises_error(self):
        """Test that initialization without model raises ValueError."""

        with self.assertRaises(ValueError):
            PyTorchTensorRTModel()

    @patch(
        "retail_shelf_monitoring.frameworks.inference_engines."
        "pytorch_tensorrt.torch_tensorrt"
    )
    def test_infer_basic(self, mock_torch_tensorrt):
        """Test basic inference functionality."""

        # Mock torch_tensorrt.compile
        mock_torch_tensorrt.compile = Mock(return_value=self.test_model)
        mock_torch_tensorrt.Input = Mock()

        model = PyTorchTensorRTModel(
            pytorch_model=self.test_model,
            input_shape=self.input_shape,
            device="cpu",
            optimize_for_inference=False,  # Skip optimization for testing
        )

        # Create test input
        input_data = np.random.randn(self.batch_size, *self.input_shape).astype(
            np.float32
        )

        # Run inference
        output = model.infer(input_data)

        # Check output properties
        self.assertIsInstance(output, np.ndarray)
        self.assertEqual(output.shape[0], self.batch_size)
        self.assertEqual(output.shape[1], 10)  # Model outputs 10 classes

    @patch(
        "retail_shelf_monitoring.frameworks.inference_engines."
        "pytorch_tensorrt.torch_tensorrt"
    )
    def test_batch_infer(self, mock_torch_tensorrt):
        """Test batch inference functionality."""

        # Mock torch_tensorrt.compile
        mock_torch_tensorrt.compile = Mock(return_value=self.test_model)
        mock_torch_tensorrt.Input = Mock()

        model = PyTorchTensorRTModel(
            pytorch_model=self.test_model,
            input_shape=self.input_shape,
            device="cpu",
            optimize_for_inference=False,
        )

        # Create test data loader
        num_samples = 10
        data_loader = [
            np.random.randn(*self.input_shape).astype(np.float32)
            for _ in range(num_samples)
        ]

        # Run batch inference
        results = model.batch_infer(data_loader, batch_size=4)

        # Check results
        self.assertIsInstance(results, np.ndarray)
        self.assertEqual(results.shape[0], num_samples)
        self.assertEqual(results.shape[1], 10)

    @patch(
        "retail_shelf_monitoring.frameworks.inference_engines."
        "pytorch_tensorrt.torch_tensorrt"
    )
    def test_precision_modes(self, mock_torch_tensorrt):
        """Test different precision modes."""

        mock_torch_tensorrt.compile = Mock(return_value=self.test_model)
        mock_torch_tensorrt.Input = Mock()

        precisions = ["fp32", "fp16"]

        for precision in precisions:
            with self.subTest(precision=precision):
                model = PyTorchTensorRTModel(
                    pytorch_model=self.test_model,
                    input_shape=self.input_shape,
                    device="cpu",
                    precision=precision,
                    optimize_for_inference=False,
                )

                self.assertEqual(model.precision, precision)

    @patch(
        "retail_shelf_monitoring.frameworks.inference_engines."
        "pytorch_tensorrt.torch_tensorrt"
    )
    def test_get_model_info(self, mock_torch_tensorrt):
        """Test model information retrieval."""

        mock_torch_tensorrt.compile = Mock(return_value=self.test_model)
        mock_torch_tensorrt.Input = Mock()

        model = PyTorchTensorRTModel(
            pytorch_model=self.test_model,
            input_shape=self.input_shape,
            device="cpu",
            optimize_for_inference=False,
        )

        info = model.get_model_info()

        # Check info structure
        self.assertIn("device", info)
        self.assertIn("precision", info)
        self.assertIn("total_parameters", info)
        self.assertIn("trainable_parameters", info)
        self.assertIn("input_shape", info)
        self.assertIn("max_batch_size", info)

        # Check values
        self.assertEqual(info["input_shape"], self.input_shape)
        self.assertIsInstance(info["total_parameters"], int)
        self.assertGreater(info["total_parameters"], 0)

    @patch(
        "retail_shelf_monitoring.frameworks.inference_engines."
        "pytorch_tensorrt.torch_tensorrt"
    )
    @patch(
        "retail_shelf_monitoring.frameworks.inference_engines."
        "pytorch_tensorrt.torch.jit.save"
    )
    def test_save_model(self, mock_jit_save, mock_torch_tensorrt):
        """Test model saving functionality."""

        mock_torch_tensorrt.compile = Mock(return_value=self.test_model)
        mock_torch_tensorrt.Input = Mock()

        model = PyTorchTensorRTModel(
            pytorch_model=self.test_model,
            input_shape=self.input_shape,
            device="cpu",
            optimize_for_inference=False,
        )

        # Test saving
        save_path = "test_model.pt"
        model.save_model(save_path)

        # Verify save was called
        mock_jit_save.assert_called_once_with(model.model, save_path)

    @patch(
        "retail_shelf_monitoring.frameworks.inference_engines."
        "pytorch_tensorrt.torch_tensorrt"
    )
    def test_preprocess_input(self, mock_torch_tensorrt):
        """Test input preprocessing."""

        mock_torch_tensorrt.compile = Mock(return_value=self.test_model)
        mock_torch_tensorrt.Input = Mock()

        model = PyTorchTensorRTModel(
            pytorch_model=self.test_model,
            input_shape=self.input_shape,
            device="cpu",
            precision="fp32",
            optimize_for_inference=False,
        )

        # Test numpy array input
        numpy_input = np.random.randn(2, *self.input_shape).astype(np.float32)
        processed = model._preprocess_input(numpy_input)

        self.assertIsInstance(processed, torch.Tensor)
        self.assertEqual(processed.dtype, torch.float32)
        self.assertEqual(processed.device.type, "cpu")

    @patch(
        "retail_shelf_monitoring.frameworks.inference_engines."
        "pytorch_tensorrt.torch_tensorrt"
    )
    def test_postprocess_output(self, mock_torch_tensorrt):
        """Test output postprocessing."""

        mock_torch_tensorrt.compile = Mock(return_value=self.test_model)
        mock_torch_tensorrt.Input = Mock()

        model = PyTorchTensorRTModel(
            pytorch_model=self.test_model,
            input_shape=self.input_shape,
            device="cpu",
            optimize_for_inference=False,
        )

        # Test single tensor output
        tensor_output = torch.randn(2, 10)
        processed = model._postprocess_output(tensor_output)

        self.assertIsInstance(processed, np.ndarray)
        self.assertEqual(processed.shape, (2, 10))

        # Test multiple tensor outputs
        tensor_outputs = [torch.randn(2, 10), torch.randn(2, 5)]
        processed_list = model._postprocess_output(tensor_outputs)

        self.assertIsInstance(processed_list, list)
        self.assertEqual(len(processed_list), 2)
        self.assertIsInstance(processed_list[0], np.ndarray)

    def test_precision_conversion(self):
        """Test precision mode conversion."""
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")

        # Create a dummy model instance (without actual initialization)
        model = object.__new__(PyTorchTensorRTModel)
        model.precision = "fp32"

        # Test precision mapping
        model.precision = "fp32"
        self.assertEqual(model._get_precision_mode(), torch.float32)

        model.precision = "fp16"
        self.assertEqual(model._get_precision_mode(), torch.float16)

        model.precision = "int8"
        self.assertEqual(model._get_precision_mode(), torch.int8)

        model.precision = "unknown"
        self.assertEqual(model._get_precision_mode(), torch.float32)


class TestPyTorchTensorRTModelIntegration(unittest.TestCase):
    """Integration tests for PyTorchTensorRTModel."""

    def setUp(self):
        """Set up test fixtures."""
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
    def test_inference_model_interface_compliance(self):
        """Test that PyTorchTensorRTModel properly implements InferenceModel interface."""  # noqa: E501
        # Create a simple model
        simple_model = nn.Linear(10, 5)

        with patch(
            "retail_shelf_monitoring.frameworks.inference_engines."
            "pytorch_tensorrt.torch_tensorrt"
        ) as mock_trt:
            mock_trt.compile = Mock(return_value=simple_model)
            mock_trt.Input = Mock()

            model = PyTorchTensorRTModel(
                pytorch_model=simple_model,
                input_shape=(10,),
                device="cpu",
                optimize_for_inference=False,
            )

            # Test interface compliance
            self.assertIsInstance(model, InferenceModel)
            self.assertTrue(hasattr(model, "input_shape"))
            self.assertTrue(hasattr(model, "infer"))
            self.assertTrue(hasattr(model, "batch_infer"))

            # Test methods are callable
            self.assertTrue(callable(model.infer))
            self.assertTrue(callable(model.batch_infer))


if __name__ == "__main__":
    unittest.main()
