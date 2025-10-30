from abc import ABC, abstractmethod
from typing import List

from retail_shelf_monitoring.entities.detection import Detection


class Tracker(ABC):
    @abstractmethod
    def update(self, detections: List[Detection]) -> List[Detection]:
        """
        Update tracker with new detections

        Args:
            detections: List of Detection objects

        Returns:
            List of detections with added 'track_id' field
        """
        pass

    @abstractmethod
    def predict(self) -> List[Detection]:
        """
        Predict the current state of tracked objects

        Returns:
            List of predicted Detection objects
        """
        pass

    @abstractmethod
    def reset(self):
        """Reset tracker state"""
        pass
