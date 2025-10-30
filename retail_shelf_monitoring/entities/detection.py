from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field, validator

from .common import BoundingBox


class Detection(BaseModel):
    detection_id: Optional[str] = Field(None, description="Unique detection identifier")
    shelf_id: str = Field(..., description="Associated shelf identifier")
    frame_timestamp: datetime = Field(..., description="Timestamp of the frame")
    bbox: BoundingBox = Field(..., description="Bounding box in reference coordinates")
    class_id: int = Field(..., ge=0, description="Detected class ID")
    sku_id: Optional[str] = Field(None, description="SKU identifier if mapped")
    confidence: float = Field(..., ge=0, le=1, description="Detection confidence")
    track_id: Optional[int] = Field(
        None, description="Tracker ID for temporal consistency"
    )
    row_idx: Optional[int] = Field(
        None, ge=0, description="Assigned planogram row index"
    )
    item_idx: Optional[int] = Field(
        None, ge=0, description="Assigned planogram item index"
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @validator("confidence")
    def confidence_reasonable(cls, v):
        if v < 0.1:
            raise ValueError(f"Detection confidence {v} is unreasonably low (< 0.1)")
        return v

    class Config:
        frozen = False
        json_encoders = {datetime: lambda v: v.isoformat()}
