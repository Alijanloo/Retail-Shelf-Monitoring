from pathlib import Path

import numpy as np
from openvino.runtime import Core

from retail_shelf_monitoring.frameworks.logging_config import get_logger
from retail_shelf_monitoring.usecases.interfaces.inference_model import InferenceModel


class OpenVINOModel(InferenceModel):
    """
    OpenVINO inference model that implements the InferenceModel interface.

    Provides a unified interface for OpenVINO-based models while maintaining
    compatibility with the existing InferenceModel abstraction.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "CPU",
    ):
        """
        Initialize OpenVINO model.

        Args:
            model_path: Path to OpenVINO model (.xml file)
            device: OpenVINO target device (CPU, GPU, etc.)
        """
        self.logger = get_logger(__name__)
        self.model_path = Path(model_path)
        self.device = device

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.logger.info(f"Loading OpenVINO model from {model_path}")
        self.core = Core()
        self.model = self.core.read_model(str(self.model_path))
        self.compiled_model = self.core.compile_model(self.model, self.device)

        self.input_layer = self.compiled_model.input(0)
        self.output_layer = self.compiled_model.output(0)

        self._input_shape = eval(
            self.input_layer.partial_shape[1:].to_string()
        )  # Exclude batch dimension

        self.logger.info(f"OpenVINO model loaded successfully on {device}")
        self.logger.info(f"Input shape: {self.input_layer.partial_shape}")

    @property
    def input_shape(self):
        """Return the input shape of the model (excluding batch dimension)."""
        return self._input_shape

    def infer(self, input_data: np.ndarray) -> np.ndarray:
        """
        Run inference for a single batch of inputs.

        Args:
            input_data: Input data as numpy array with shape (batch_size, ...)

        Returns:
            Output as numpy array
        """
        outputs = self.compiled_model([input_data])
        return outputs[self.output_layer]

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
