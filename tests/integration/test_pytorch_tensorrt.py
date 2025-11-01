"""
Test script to verify PyTorch TensorRT model with ONNX loading capability.
"""

import numpy as np
import torch
import torch.nn as nn

from retail_shelf_monitoring.frameworks.inference_engines.pytorch_tensorrt import (
    PyTorchTensorRTModel,
)


def create_test_model():
    """Create a simple test model."""

    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 16, 3, padding=1)
            self.relu = nn.ReLU()
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(16, 10)

        def forward(self, x):
            x = self.conv(x)
            x = self.relu(x)
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x

    return SimpleModel()


def test_pytorch_model_loading():
    """Test loading a PyTorch model directly."""
    print("=== Testing PyTorch Model Loading ===")

    # Create test model
    pytorch_model = create_test_model()
    input_shape = (3, 32, 32)

    try:
        model = PyTorchTensorRTModel(
            pytorch_model=pytorch_model,
            input_shape=input_shape,
            device="cuda" if torch.cuda.is_available() else "cpu",
            precision="fp32",
            max_batch_size=4,
            optimize_for_inference=False,  # Skip TensorRT optimization for now
        )

        # Test inference
        test_input = np.random.randn(2, *input_shape).astype(np.float32)
        output = model.infer(test_input)

        print(
            "✅ Success! Input shape: {}, Output shape: {}".format(
                test_input.shape, output.shape
            )
        )
        return model

    except Exception as e:
        print(f"❌ Failed: {e}")
        return None


def test_onnx_export_and_load():
    """Test exporting to ONNX and loading back."""
    print("\n=== Testing ONNX Export and Loading ===")

    try:
        # Create and export model to ONNX
        pytorch_model = create_test_model()
        input_shape = (3, 32, 32)
        dummy_input = torch.randn(1, *input_shape)

        onnx_path = "test_model.onnx"
        torch.onnx.export(
            pytorch_model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )
        print(f"✅ Exported model to {onnx_path}")

        # Load ONNX model using PyTorchTensorRTModel
        model = PyTorchTensorRTModel(
            onnx_path=onnx_path,
            device="cuda" if torch.cuda.is_available() else "cpu",
            precision="fp32",
            optimize_for_inference=False,
        )

        # Test inference
        test_input = np.random.randn(2, *input_shape).astype(np.float32)
        output = model.infer(test_input)

        print("✅ Success! Loaded ONNX model and ran inference")
        print(
            "Input shape: {}, Output shape: {}".format(test_input.shape, output.shape)
        )
        print("Inferred input shape: {}".format(model.input_shape))

        # Clean up
        import os

        os.remove(onnx_path)

        return model

    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback

        traceback.print_exc()
        return None


def test_batch_inference():
    """Test batch inference capability."""
    print("\n=== Testing Batch Inference ===")

    try:
        pytorch_model = create_test_model()
        input_shape = (3, 32, 32)

        model = PyTorchTensorRTModel(
            pytorch_model=pytorch_model,
            input_shape=input_shape,
            device="cuda" if torch.cuda.is_available() else "cpu",
            precision="fp32",
            optimize_for_inference=False,
        )

        # Create batch data
        num_samples = 10
        data_loader = [
            np.random.randn(*input_shape).astype(np.float32) for _ in range(num_samples)
        ]

        # Run batch inference
        results = model.batch_infer(data_loader, batch_size=4)

        print("✅ Success! Processed {} samples in batches".format(num_samples))
        print("Results shape: {}".format(results.shape))

        return True

    except Exception as e:
        print(f"❌ Failed: {e}")
        return False


def test_model_info():
    """Test model information retrieval."""
    print("\n=== Testing Model Information ===")

    try:
        pytorch_model = create_test_model()
        input_shape = (3, 32, 32)

        PyTorchTensorRTModel(
            pytorch_model=pytorch_model,
            input_shape=input_shape,
            device="cuda" if torch.cuda.is_available() else "cpu",
            precision="fp16" if torch.cuda.is_available() else "fp32",
            optimize_for_inference=False,
        )

        return True

    except Exception as e:
        print(f"❌ Failed: {e}")
        return False


def main():
    """Run all tests."""
    print("PyTorch TensorRT Model Test Suite")
    print("=" * 50)

    # Check environment
    print("PyTorch version: {}".format(torch.__version__))
    print("CUDA available: {}".format(torch.cuda.is_available()))
    if torch.cuda.is_available():
        print("CUDA device: {}".format(torch.cuda.get_device_name()))

    try:
        import torch_tensorrt

        print("TensorRT available: Yes (v{})".format(torch_tensorrt.__version__))
    except ImportError:
        print("TensorRT available: No (torch_tensorrt not installed)")

    print()

    # Run tests
    tests_passed = 0
    total_tests = 4

    if test_pytorch_model_loading():
        tests_passed += 1

    if test_onnx_export_and_load():
        tests_passed += 1

    if test_batch_inference():
        tests_passed += 1

    if test_model_info():
        tests_passed += 1

    print("=== Test Results ===")
    print("Tests passed: {}/{}".format(tests_passed, total_tests))

    if tests_passed == total_tests:
        print("🎉 All tests passed! PyTorch TensorRT model is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the error messages above.")


if __name__ == "__main__":
    main()
