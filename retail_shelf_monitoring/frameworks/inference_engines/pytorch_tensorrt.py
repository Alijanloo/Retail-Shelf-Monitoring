import importlib

import numpy as np
import torch
from tqdm import tqdm

try:
    import torch_tensorrt

    TORCH_TENSORRT_AVAILABLE = True
except ImportError:
    TORCH_TENSORRT_AVAILABLE = False

from retail_shelf_monitoring.frameworks.logging_config import get_logger
from retail_shelf_monitoring.usecases.interfaces.inference_model import InferenceModel


class PyTorchTensorRTModel(InferenceModel):
    """
    PyTorch TensorRT inference model that optimizes PyTorch models using TensorRT
    while maintaining PyTorch's dynamic nature and ease of use.
    Supports loading from PyTorch models, and TorchScript files.
    """

    def __init__(
        self,
        model_path: str = None,
        pytorch_model: str = None,
        device: str = "cuda",
        precision: str = "fp16",
        workspace_size: int = 1 << 30,
        max_batch_size: int = 8,
        optimize_for_inference: bool = True,
    ):
        """
        Initialize PyTorch TensorRT model.

        Args:
            model_path: Path to saved PyTorch model, TensorRT-optimized model, or
                .pth weights file
            pytorch_model: Path to Python file containing PyTorch model definition
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

        if model_path and not pytorch_model:
            self.model = self._load_model(model_path)
        elif model_path and pytorch_model:
            self.model = self._load_weights_into_model(pytorch_model, model_path)
        else:
            raise ValueError("No model source provided")

        self._input_shape = (
            self.model.input_shape
            if hasattr(self.model, "input_shape")
            else (3, 640, 640)
        )
        if (
            optimize_for_inference
            and self.device.type == "cuda"
            and TORCH_TENSORRT_AVAILABLE
        ):
            self._optimize_model()
        elif optimize_for_inference and not TORCH_TENSORRT_AVAILABLE:
            self.logger.warning(
                "torch_tensorrt not available. Install it for TensorRT optimization."
            )
        elif optimize_for_inference and self.device.type == "cpu":
            self.logger.info(
                "TensorRT optimization not available on CPU. "
                "Using standard PyTorch model."
            )

        self.model.to(self.device)
        self.model.eval()

        # Convert model to the specified precision
        if self.precision == "fp16":
            self.model.half()
        elif self.precision == "fp32":
            self.model.float()

        self.logger.info(
            f"PyTorchTensorRTModel initialized successfully on {self.device} with "
            f"precision {self.precision}"
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

    def _load_pytorch_model_from_file(self, pytorch_model_path: str) -> torch.nn.Module:
        """Load PyTorch model from Python file using importlib."""
        self.logger.info(
            f"Loading PyTorch model definition from {pytorch_model_path}..."
        )

        try:
            module_name = pytorch_model_path.split(".")[-1]
            pytorch_model_path = pytorch_model_path.split(".")[:-1]
            pytorch_model_path = ".".join(pytorch_model_path)

            spec = importlib.import_module(pytorch_model_path)
            if spec is None:
                raise ImportError(f"Could not load spec from {pytorch_model_path}")

            model_factory = getattr(spec, module_name)

            if callable(model_factory):
                try:
                    model = model_factory()
                except TypeError:
                    try:
                        model = model_factory()
                    except Exception:
                        raise RuntimeError(
                            f"Failed to instantiate model from {pytorch_model_path}. "
                            f"Model factory '{model_factory.__name__}' requires "
                            "arguments."
                        )
            else:
                model = model_factory

            if not isinstance(model, torch.nn.Module):
                raise TypeError(
                    f"Expected torch.nn.Module, got {type(model)} from "
                    f"{pytorch_model_path}"
                )

            self.logger.info(
                f"Successfully loaded PyTorch model from {pytorch_model_path}"
            )
            return model

        except Exception as e:
            raise RuntimeError(
                f"Failed to load PyTorch model from {pytorch_model_path}: {e}"
            )

    def _load_weights_into_model(self, pytorch_model_path: str, weights_path: str):
        """Load weights from .pth file into a PyTorch model definition."""
        self.logger.info(f"Loading weights from {weights_path} into model...")

        try:
            pytorch_model = self._load_pytorch_model_from_file(pytorch_model_path)

            checkpoint = torch.load(
                weights_path, map_location=self.device, weights_only=False
            )

            if isinstance(checkpoint, dict):
                if "model" in checkpoint:
                    state_dict = checkpoint["model"]
                elif "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint

            pytorch_model.load_state_dict(state_dict, strict=False)
            self.logger.info("Successfully loaded weights into model")

            return pytorch_model

        except Exception as e:
            raise RuntimeError(f"Failed to load weights from {weights_path}: {e}")

    def _load_model(self, model_path: str):
        """Load a PyTorch model from file."""
        self.logger.info(f"Loading model from {model_path}...")

        try:
            model = torch.jit.load(model_path, map_location=self.device)
            self.logger.info("Loaded TensorRT-optimized model")
        except Exception:
            try:
                model = torch.load(
                    model_path, map_location=self.device, weights_only=False
                )["model"]
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
            return output[0].detach().cpu().numpy()
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
            for i in tqdm(range(0, len(data_loader), batch_size)):
                batch_data = data_loader[i : i + batch_size]

                if not isinstance(batch_data, np.ndarray):
                    # Pad data to the maximum length
                    if batch_data:
                        max_height = max(img.shape[0] for img in batch_data)
                        max_width = max(img.shape[1] for img in batch_data)
                        padded_batch = []
                        for img in batch_data:
                            h, w, c = img.shape
                            img = img.transpose(
                                2, 0, 1
                            )  # Convert from HWC to CHW format
                            padded_img = np.zeros(
                                (c, max_height, max_width), dtype=img.dtype
                            )
                            padded_img[:, :h, :w] = img
                            padded_batch.append(padded_img)
                        batch_data = np.array(padded_batch)
                    else:
                        batch_data = np.array(batch_data)

                batch_output = self.infer(batch_data)
                results.append(batch_output)

        return np.concatenate(results, axis=0)
