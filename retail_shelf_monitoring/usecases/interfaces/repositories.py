from abc import ABC, abstractmethod
from typing import List, Optional

from ...entities.alert import Alert
from ...entities.planogram import Planogram


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


class AlertRepository(ABC):
    @abstractmethod
    async def create(self, alert: Alert) -> Alert:
        pass

    @abstractmethod
    async def get_by_id(self, alert_id: str) -> Optional[Alert]:
        pass

    @abstractmethod
    async def get_active_alerts(self, shelf_id: Optional[str] = None) -> List[Alert]:
        pass

    @abstractmethod
    async def get_by_cell(
        self, shelf_id: str, row_idx: int, item_idx: int
    ) -> Optional[Alert]:
        pass

    @abstractmethod
    async def update(self, alert: Alert) -> Alert:
        pass

    @abstractmethod
    async def confirm_alert(self, alert_id: str, confirmed_by: str) -> Alert:
        pass

    @abstractmethod
    async def dismiss_alert(self, alert_id: str) -> Alert:
        pass
