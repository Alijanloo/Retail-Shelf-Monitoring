from typing import List, Optional

import numpy as np

try:
    import onnxruntime as ort

    ONNXRUNTIME_AVAILABLE = True
except ImportError:
    ONNXRUNTIME_AVAILABLE = False

try:
    import onnx  # noqa: F401

    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

from retail_shelf_monitoring.frameworks.logging_config import get_logger
from retail_shelf_monitoring.usecases.interfaces.inference_model import InferenceModel


class ONNXRuntimeModel(InferenceModel):
    """
    ONNX Runtime inference model for efficient inference with ONNX models.
    Supports both CPU and GPU execution with automatic provider selection.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        providers: Optional[List[str]] = None,
        session_options: Optional[ort.SessionOptions] = None,
    ):
        """
        Initialize ONNX Runtime model.

        Args:
            model_path: Path to ONNX model file
            device: Device to run inference on ('cuda' or 'cpu')
            providers: List of execution providers. If None, will auto-select
                based on device
            session_options: ONNX Runtime session options for optimization
        """
        self.logger = get_logger(__name__)

        if not ONNXRUNTIME_AVAILABLE:
            raise ImportError(
                "onnxruntime is required for ONNX model inference. "
                "Install with: uv pip install onnxruntime or onnxruntime-gpu"
            )

        self.model_path = model_path
        self.device = device

        # Set up execution providers
        if providers is None:
            if device == "cuda":
                self.providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                self.providers = ["CPUExecutionProvider"]
        else:
            self.providers = providers

        # Create session options with optimizations
        if session_options is None:
            self.session_options = ort.SessionOptions()
            self.session_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )
            # Reduce memory copy warnings
            self.session_options.log_severity_level = 3
        else:
            self.session_options = session_options

        # Initialize ONNX Runtime session
        try:
            self.session = ort.InferenceSession(
                self.model_path,
                providers=self.providers,
                sess_options=self.session_options,
            )
            self.logger.info(
                "ONNX Runtime session created with providers: "
                f"{self.session.get_providers()}"
            )
        except Exception as e:
            self.logger.error(f"Failed to create ONNX Runtime session: {e}")
            raise

        # Get model input/output information
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]

        # Infer input shape
        self._input_shape = self._infer_input_shape()

        self.logger.info("ONNXRuntimeModel initialized successfully")
        self.logger.info(f"Input shape: {self._input_shape}")
        self.logger.info(f"Input name: {self.input_name}")
        self.logger.info(f"Output names: {self.output_names}")

    @property
    def input_shape(self):
        """Return the input shape of the model (excluding batch dimension)."""
        return self._input_shape

    def _infer_input_shape(self) -> tuple:
        """Infer input shape from ONNX Runtime session."""
        try:
            input_info = self.session.get_inputs()[0]
            input_shape = input_info.shape

            # Remove batch dimension and handle dynamic dimensions
            shape = []
            for dim in input_shape[1:]:
                if isinstance(dim, int) and dim > 0:
                    shape.append(dim)
                elif isinstance(dim, str):  # Dynamic dimension
                    shape.append(-1)
                else:
                    shape.append(224)  # Default fallback

            return tuple(shape)

        except Exception as e:
            self.logger.warning(f"Could not infer input shape from ONNX Runtime: {e}")
            return (3, 224, 224)  # Default shape

    def _validate_and_convert_input(self, input_data: np.ndarray) -> np.ndarray:
        """
        Validate and convert input data to the expected format.

        Args:
            input_data: Input data as numpy array

        Returns:
            Converted numpy array ready for inference
        """
        # Convert to numpy if needed
        if not isinstance(input_data, np.ndarray):
            input_data = np.array(input_data)

        # Ensure input is float32 as expected by most ONNX models
        if input_data.dtype != np.float32:
            input_data = input_data.astype(np.float32)

        # Validate input shape (excluding batch dimension)
        expected_shape = self._input_shape
        actual_shape = (
            input_data.shape[1:]
            if len(input_data.shape) > len(expected_shape)
            else input_data.shape
        )

        # Only validate non-dynamic dimensions
        for i, (expected, actual) in enumerate(zip(expected_shape, actual_shape)):
            if expected != -1 and expected != actual:
                self.logger.warning(
                    f"Input shape mismatch at dimension {i}: expected {expected}, got "
                    f"{actual}"
                )

        return input_data

    def infer(self, input_data: np.ndarray) -> np.ndarray:
        """
        Run inference for a single batch of inputs.

        Args:
            input_data: Input data as numpy array with shape (batch_size, ...)

        Returns:
            Output as numpy array
        """
        # Validate and convert input
        processed_input = self._validate_and_convert_input(input_data)

        # Run inference
        try:
            outputs = self.session.run(
                self.output_names, {self.input_name: processed_input}
            )

            # Return single output or list of outputs
            if len(outputs) == 1:
                return outputs[0]
            else:
                return outputs

        except Exception as e:
            self.logger.error(f"Inference failed: {e}")
            raise

    def batch_infer(self, data_loader, batch_size: int = 8) -> np.ndarray:
        """
        Run inference on multiple inputs in batches.

        Args:
            data_loader: List or array of input data
            batch_size: Batch size for processing

        Returns:
            Concatenated outputs as numpy array
        """
        results = []

        for i in range(0, len(data_loader), batch_size):
            batch_data = data_loader[i : i + batch_size]

            if not isinstance(batch_data, np.ndarray):
                batch_data = np.array(batch_data)

            batch_output = self.infer(batch_data)
            results.append(batch_output)

        return np.concatenate(results, axis=0)

    def get_model_info(self) -> dict:
        """
        Get comprehensive information about the ONNX model.

        Returns:
            Dictionary containing model metadata
        """
        info = {
            "model_path": self.model_path,
            "device": self.device,
            "providers": self.session.get_providers(),
            "input_shape": self._input_shape,
            "input_name": self.input_name,
            "output_names": self.output_names,
        }

        # Add input/output details
        inputs_info = []
        for input_info in self.session.get_inputs():
            inputs_info.append(
                {
                    "name": input_info.name,
                    "type": input_info.type,
                    "shape": input_info.shape,
                }
            )

        outputs_info = []
        for output_info in self.session.get_outputs():
            outputs_info.append(
                {
                    "name": output_info.name,
                    "type": output_info.type,
                    "shape": output_info.shape,
                }
            )

        info["inputs_info"] = inputs_info
        info["outputs_info"] = outputs_info

        return info

    def warm_up(self, num_iterations: int = 3):
        """
        Warm up the model with dummy inputs to optimize performance.

        Args:
            num_iterations: Number of warm-up iterations
        """
        self.logger.info(f"Warming up model with {num_iterations} iterations...")

        # Create dummy input based on input shape
        batch_size = 1
        if self._input_shape:
            # Replace dynamic dimensions with reasonable values
            dummy_shape = [batch_size]
            for dim in self._input_shape:
                if dim == -1:
                    dummy_shape.append(224)  # Default size for dynamic dims
                else:
                    dummy_shape.append(dim)
        else:
            dummy_shape = [batch_size, 3, 224, 224]  # Default shape

        dummy_input = np.random.randn(*dummy_shape).astype(np.float32)

        for i in range(num_iterations):
            try:
                _ = self.infer(dummy_input)
            except Exception as e:
                self.logger.warning(f"Warm-up iteration {i+1} failed: {e}")

        self.logger.info("Model warm-up completed")
