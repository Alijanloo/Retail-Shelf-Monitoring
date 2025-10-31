from abc import ABC, abstractmethod

import numpy as np


class InferenceModel(ABC):
    """Interface for TensorRT model inference"""

    @abstractmethod
    def input_shape(self):
        """Return the input shape of the model (excluding batch dimension)."""

    @abstractmethod
    def infer(self, input_data: np.ndarray) -> np.ndarray:
        """Run inference for a single batch of inputs."""
        pass

    @abstractmethod
    def batch_infer(self, data_loader, batch_size: int = 8) -> np.ndarray:
        """Run inference on multiple inputs in batches."""
        pass
