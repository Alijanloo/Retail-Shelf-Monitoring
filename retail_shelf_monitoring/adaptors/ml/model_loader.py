from pathlib import Path

from openvino.runtime import Core

from ...frameworks.logging_config import get_logger

logger = get_logger(__name__)


class ModelLoader:
    def __init__(self, device: str = "CPU"):
        self.device = device
        self.core = Core()
        logger.info(f"Initialized OpenVINO Core with device: {device}")

    def load_model(self, model_path: str):
        model_path_obj = Path(model_path)

        if not model_path_obj.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        logger.info(f"Loading model from {model_path}")
        model = self.core.read_model(str(model_path_obj))
        compiled_model = self.core.compile_model(model, self.device)

        logger.info(f"Successfully loaded and compiled model on {self.device}")

        return compiled_model

    def get_available_devices(self):
        devices = self.core.available_devices
        logger.info(f"Available OpenVINO devices: {devices}")
        return devices
