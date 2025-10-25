from abc import ABC, abstractmethod
from typing import List, Optional
from datetime import datetime
from ...entities.shelf import Shelf
from ...entities.planogram import Planogram
from ...entities.detection import Detection
from ...entities.alert import Alert


class ShelfRepository(ABC):
    @abstractmethod
    async def create(self, shelf: Shelf) -> Shelf:
        pass

    @abstractmethod
    async def get_by_id(self, shelf_id: str) -> Optional[Shelf]:
        pass

    @abstractmethod
    async def get_all(self, active_only: bool = True) -> List[Shelf]:
        pass

    @abstractmethod
    async def update(self, shelf: Shelf) -> Shelf:
        pass

    @abstractmethod
    async def delete(self, shelf_id: str) -> bool:
        pass


class PlanogramRepository(ABC):
    @abstractmethod
    async def create(self, planogram: Planogram) -> Planogram:
        pass

    @abstractmethod
    async def get_by_shelf_id(self, shelf_id: str) -> Optional[Planogram]:
        pass

    @abstractmethod
    async def get_all(self) -> List[Planogram]:
        pass

    @abstractmethod
    async def update(self, planogram: Planogram) -> Planogram:
        pass

    @abstractmethod
    async def delete(self, shelf_id: str) -> bool:
        pass


class DetectionRepository(ABC):
    @abstractmethod
    async def create(self, detection: Detection) -> Detection:
        pass

    @abstractmethod
    async def create_batch(self, detections: List[Detection]) -> List[Detection]:
        pass

    @abstractmethod
    async def get_by_id(self, detection_id: str) -> Optional[Detection]:
        pass

    @abstractmethod
    async def get_by_shelf(
        self, 
        shelf_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Detection]:
        pass

    @abstractmethod
    async def get_recent_by_cell(
        self,
        shelf_id: str,
        row_idx: int,
        item_idx: int,
        limit: int = 10
    ) -> List[Detection]:
        pass


class AlertRepository(ABC):
    @abstractmethod
    async def create(self, alert: Alert) -> Alert:
        pass

    @abstractmethod
    async def get_by_id(self, alert_id: str) -> Optional[Alert]:
        pass

    @abstractmethod
    async def get_active_alerts(
        self, 
        shelf_id: Optional[str] = None
    ) -> List[Alert]:
        pass

    @abstractmethod
    async def get_by_cell(
        self,
        shelf_id: str,
        row_idx: int,
        item_idx: int
    ) -> Optional[Alert]:
        pass

    @abstractmethod
    async def update(self, alert: Alert) -> Alert:
        pass

    @abstractmethod
    async def confirm_alert(
        self,
        alert_id: str,
        confirmed_by: str
    ) -> Alert:
        pass

    @abstractmethod
    async def dismiss_alert(self, alert_id: str) -> Alert:
        pass
