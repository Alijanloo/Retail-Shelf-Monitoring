from datetime import datetime, timezone
from typing import List, Optional

from pydantic import BaseModel, Field, validator

from .common import AlertType


class Alert(BaseModel):
    alert_id: Optional[str] = Field(None, description="Unique alert identifier")
    shelf_id: str = Field(..., description="Associated shelf identifier")
    row_idx: int = Field(..., ge=0, description="Planogram row index")
    item_idx: int = Field(..., ge=0, description="Planogram item index")
    alert_type: AlertType = Field(..., description="Type of alert")
    expected_sku: Optional[str] = Field(
        None, description="Expected SKU at this position"
    )
    detected_sku: Optional[str] = Field(
        None, description="Actually detected SKU (for misplacement)"
    )
    first_seen: datetime = Field(
        ..., description="First timestamp alert condition detected"
    )
    last_seen: datetime = Field(
        ..., description="Last timestamp alert condition detected"
    )
    confirmed: bool = Field(
        default=False, description="Whether alert is confirmed by staff"
    )
    confirmed_by: Optional[str] = Field(None, description="Staff ID who confirmed")
    confirmed_at: Optional[datetime] = Field(None, description="Confirmation timestamp")
    dismissed: bool = Field(default=False, description="Whether alert was dismissed")
    evidence_paths: List[str] = Field(
        default_factory=list, description="Paths to evidence images"
    )
    consecutive_frames: int = Field(
        default=1, ge=1, description="Number of consecutive frames with issue"
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @validator("confirmed_at")
    def confirmed_at_requires_confirmed(cls, v, values):
        if v is not None and not values.get("confirmed", False):
            raise ValueError("confirmed_at can only be set when confirmed is True")
        return v

    @validator("confirmed_by")
    def confirmed_by_requires_confirmed(cls, v, values):
        if v is not None and not values.get("confirmed", False):
            raise ValueError("confirmed_by can only be set when confirmed is True")
        return v

    class Config:
        frozen = False
        json_encoders = {datetime: lambda v: v.isoformat()}
