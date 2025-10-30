from enum import Enum
from typing import Tuple

from pydantic import BaseModel, Field, validator


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


class BoundingBox(BaseModel):
    x1: float = Field(..., description="Top-left X coordinate")
    y1: float = Field(..., description="Top-left Y coordinate")
    x2: float = Field(..., description="Bottom-right X coordinate")
    y2: float = Field(..., description="Bottom-right Y coordinate")

    @validator("x1")
    def check_x1(cls, v):
        if v < 0:
            return 0
        return v

    @validator("y1")
    def check_y1(cls, v):
        if v < 0:
            return 0
        return v

    @validator("x2")
    def check_x2(cls, v):
        if v < 0:
            return 0
        return v

    @validator("y2")
    def check_y2(cls, v):
        if v < 0:
            return 0
        return v

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def width(self) -> float:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    class Config:
        frozen = True
