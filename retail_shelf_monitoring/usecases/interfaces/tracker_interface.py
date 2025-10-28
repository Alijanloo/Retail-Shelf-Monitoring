from abc import ABC, abstractmethod
from typing import Dict, List


class Tracker(ABC):
    @abstractmethod
    def update(self, detections: List[Dict]) -> List[Dict]:
        """
        Update tracker with new detections

        Args:
            detections: List of detection dicts with 'bbox' and 'confidence'

        Returns:
            List of detections with added 'track_id' field
        """
        pass

    @abstractmethod
    def reset(self):
        """Reset tracker state"""
        pass
