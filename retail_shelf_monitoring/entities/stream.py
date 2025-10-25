from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class StreamConfig(BaseModel):
    stream_id: str = Field(..., description="Unique stream identifier")
    source_url: str = Field(..., description="RTSP URL or video file path")
    fps: float = Field(default=30.0, gt=0, description="Target frames per second")
    process_every_n_frames: int = Field(
        default=30, ge=1, description="Process every Nth frame"
    )
    max_width: Optional[int] = Field(
        default=1920, gt=0, description="Max frame width (for resizing)"
    )
    max_height: Optional[int] = Field(
        default=1080, gt=0, description="Max frame height (for resizing)"
    )
    enable_stabilization: bool = Field(
        default=False, description="Enable motion stabilization"
    )
    active: bool = Field(default=True, description="Stream is active")
    store_id: Optional[str] = Field(None, description="Associated store ID")
    camera_location: Optional[str] = Field(None, description="Physical camera location")
    created_at: datetime = Field(default_factory=datetime.utcnow)

    @field_validator("source_url")
    @classmethod
    def validate_source_url(cls, v):
        if not v or not v.strip():
            raise ValueError("Source URL cannot be empty")
        return v.strip()

    @property
    def frame_interval(self) -> float:
        return self.process_every_n_frames / self.fps

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }
