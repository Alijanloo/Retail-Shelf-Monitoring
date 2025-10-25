from enum import Enum
from typing import Tuple
from pydantic import BaseModel, Field


class AlertType(str, Enum):
    OOS = "out_of_stock"
    MISPLACEMENT = "misplacement"
    UNKNOWN = "unknown"


class CellState(str, Enum):
    OK = "ok"
    OOS = "out_of_stock"
    MISPLACED = "misplaced"
    EMPTY = "empty"
    UNKNOWN = "unknown"


class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class BoundingBox(BaseModel):
    x1: float = Field(..., ge=0, description="Top-left X coordinate")
    y1: float = Field(..., ge=0, description="Top-left Y coordinate")
    x2: float = Field(..., gt=0, description="Bottom-right X coordinate")
    y2: float = Field(..., gt=0, description="Bottom-right Y coordinate")

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    class Config:
        frozen = True
