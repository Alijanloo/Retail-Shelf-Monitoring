import numpy as np
import torch

try:
    import torch_tensorrt

    TORCH_TENSORRT_AVAILABLE = True
except ImportError:
    TORCH_TENSORRT_AVAILABLE = False

try:
    import onnx
    import onnx2torch

    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

from retail_shelf_monitoring.frameworks.logging_config import get_logger
from retail_shelf_monitoring.usecases.interfaces.inference_model import InferenceModel


class PyTorchTensorRTModel(InferenceModel):
    """
    PyTorch TensorRT inference model that optimizes PyTorch models using TensorRT
    while maintaining PyTorch's dynamic nature and ease of use.
    Supports loading from PyTorch models, TorchScript, and ONNX files.
    """

    def __init__(
        self,
        model_path: str = None,
        pytorch_model: torch.nn.Module = None,
        onnx_path: str = None,
        input_shape: tuple = None,
        device: str = "cuda",
        precision: str = "fp16",
        workspace_size: int = 1 << 30,
        max_batch_size: int = 8,
        optimize_for_inference: bool = True,
    ):
        """
        Initialize PyTorch TensorRT model.

        Args:
            model_path: Path to saved PyTorch model or TensorRT-optimized model
            pytorch_model: PyTorch model instance to optimize
            onnx_path: Path to ONNX model file
            input_shape: Input shape for the model (without batch dimension)
            device: Device to run inference on ('cuda' or 'cpu')
            precision: Precision mode ('fp32', 'fp16', 'int8')
            workspace_size: TensorRT workspace size in bytes
            max_batch_size: Maximum batch size for optimization
            optimize_for_inference: Whether to optimize the model with TensorRT
        """
        self.logger = get_logger(__name__)
        self.device = torch.device(device)
        self.precision = precision
        self.workspace_size = workspace_size
        self.max_batch_size = max_batch_size
        self._input_shape = input_shape

        model_sources = [model_path, pytorch_model, onnx_path]
        provided_sources = [src for src in model_sources if src is not None]

        if len(provided_sources) != 1:
            raise ValueError(
                "You must provide exactly one of: "
                "model_path, pytorch_model, or onnx_path."
            )

        if model_path:
            self.model = self._load_model(model_path)
        elif onnx_path:
            self.model = self._load_onnx_model(onnx_path)
        else:
            self.model = pytorch_model

        if (
            optimize_for_inference
            and self.device.type == "cuda"
            and TORCH_TENSORRT_AVAILABLE
        ):
            self._optimize_model()
        elif optimize_for_inference and not TORCH_TENSORRT_AVAILABLE:
            self.logger.warning(
                "torch_tensorrt not available. " "Install it for TensorRT optimization."
            )
        elif optimize_for_inference and self.device.type == "cpu":
            self.logger.info(
                "TensorRT optimization not available on CPU. "
                "Using standard PyTorch model."
            )

        self.model.to(self.device)
        self.model.eval()

        self.logger.info(
            f"PyTorchTensorRTModel initialized successfully on {self.device}"
        )

    @property
    def input_shape(self):
        """Return the input shape of the model (excluding batch dimension)."""
        return self._input_shape

    def _get_precision_mode(self):
        """Convert precision string to torch_tensorrt dtype."""
        precision_map = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "int8": torch.int8,
        }
        return precision_map.get(self.precision, torch.float32)

    def _load_onnx_model(self, onnx_path: str) -> torch.nn.Module:
        """
        Load ONNX model and convert to PyTorch.

        Args:
            onnx_path: Path to ONNX model file

        Returns:
            PyTorch model converted from ONNX
        """
        self.logger.info(f"Loading ONNX model from {onnx_path}...")

        if not ONNX_AVAILABLE:
            self.logger.error(
                "onnx and onnx2torch are required to load ONNX models. "
                "Install with: uv pip install onnx onnx2torch"
            )
            raise ImportError(
                "onnx and onnx2torch packages are required " "for ONNX model loading"
            )

        try:
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            self.logger.info("ONNX model validation passed")

            pytorch_model = onnx2torch.convert(onnx_model)
            self.logger.info("Successfully converted ONNX to PyTorch model")

            if self._input_shape is None:
                self._infer_input_shape_from_onnx(onnx_model)

            return pytorch_model

        except Exception as e:
            self.logger.error(f"Failed to load ONNX model: {e}")
            raise RuntimeError(f"Failed to load ONNX model from {onnx_path}: {e}")

    def _infer_input_shape_from_onnx(self, onnx_model):
        """Infer input shape from ONNX model if not provided."""
        try:
            input_info = onnx_model.graph.input[0]
            input_dims = input_info.type.tensor_type.shape.dim

            shape = []
            for dim in input_dims[1:]:
                if hasattr(dim, "dim_value") and dim.dim_value > 0:
                    shape.append(dim.dim_value)
                elif hasattr(dim, "dim_param"):
                    shape.append(-1)
                else:
                    shape.append(224)

            self._input_shape = tuple(shape)
            self.logger.info(f"Inferred input shape from ONNX: {self._input_shape}")

        except Exception as e:
            self.logger.warning(f"Could not infer input shape from ONNX model: {e}")
            self._input_shape = (3, 224, 224)

    def _optimize_model(self):
        """Optimize the PyTorch model using TensorRT."""
        if not self._input_shape:
            raise ValueError("Input shape must be provided for model optimization.")

        if not TORCH_TENSORRT_AVAILABLE:
            self.logger.warning(
                "torch_tensorrt not available. Skipping TensorRT optimization."
            )
            return

        try:
            self.logger.info("Optimizing model with TensorRT...")

            compile_spec = {
                "inputs": [
                    torch_tensorrt.Input(
                        min_shape=(1, *self._input_shape),
                        opt_shape=(self.max_batch_size // 2, *self._input_shape),
                        max_shape=(self.max_batch_size, *self._input_shape),
                        dtype=self._get_precision_mode(),
                    )
                ],
                "enabled_precisions": {self._get_precision_mode()},
                "workspace_size": self.workspace_size,
            }

            self.model = torch_tensorrt.compile(self.model, **compile_spec)
            self.logger.info("Model optimization completed successfully")

        except Exception as e:
            self.logger.warning(f"TensorRT optimization failed: {e}")
            self.logger.info("Falling back to standard PyTorch model")

    def _load_model(self, model_path: str):
        """Load a PyTorch model from file."""
        self.logger.info(f"Loading model from {model_path}...")

        try:
            model = torch.jit.load(model_path, map_location=self.device)
            self.logger.info("Loaded TensorRT-optimized model")
        except Exception:
            try:
                model = torch.load(model_path, map_location=self.device)
                self.logger.info("Loaded standard PyTorch model")
            except Exception as e:
                raise RuntimeError(f"Failed to load model from {model_path}: {e}")

        return model

    def save_model(self, path: str):
        """Save the optimized model."""
        torch.jit.save(self.model, path)
        self.logger.info(f"Model saved to {path}")

    def _preprocess_input(self, input_data: np.ndarray) -> torch.Tensor:
        """Convert numpy array to PyTorch tensor and move to device."""
        if isinstance(input_data, np.ndarray):
            tensor = torch.from_numpy(input_data).to(self.device)
        else:
            tensor = input_data.to(self.device)

        if self.precision == "fp16":
            tensor = tensor.half()
        elif self.precision == "fp32":
            tensor = tensor.float()

        return tensor

    def _postprocess_output(self, output: torch.Tensor) -> np.ndarray:
        """Convert PyTorch tensor output to numpy array."""
        if isinstance(output, (list, tuple)):
            return [tensor.detach().cpu().numpy() for tensor in output]
        else:
            return output.detach().cpu().numpy()

    def infer(self, input_data: np.ndarray) -> np.ndarray:
        """
        Run inference for a single batch of inputs.

        Args:
            input_data: Input data as numpy array with shape (batch_size, ...)

        Returns:
            Output as numpy array
        """
        with torch.no_grad():
            input_tensor = self._preprocess_input(input_data)
            output = self.model(input_tensor)
            return self._postprocess_output(output)

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

        with torch.no_grad():
            for i in range(0, len(data_loader), batch_size):
                batch_data = data_loader[i : i + batch_size]

                if not isinstance(batch_data, np.ndarray):
                    batch_data = np.array(batch_data)

                batch_output = self.infer(batch_data)
                results.append(batch_output)

        return np.concatenate(results, axis=0)
