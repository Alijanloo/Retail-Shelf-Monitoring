from datetime import datetime
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field


class Frame(BaseModel):
    frame_id: str = Field(..., description="Unique frame identifier")
    frame_img: np.ndarray = Field(..., description="Original frame image")
    timestamp: datetime = Field(..., description="Frame capture timestamp")
    is_keyframe: bool = Field(default=False, description="Whether this is a keyframe")
    shelf_id: Optional[str] = Field(None, description="Detected shelf ID if localized")
    homography_matrix: Optional[list] = Field(
        None, description="3x3 homography matrix (flattened)"
    )
    alignment_confidence: Optional[float] = Field(
        None, ge=0, le=1, description="Confidence of shelf alignment"
    )
    inlier_ratio: Optional[float] = Field(
        None, ge=0, le=1, description="RANSAC inlier ratio"
    )

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            np.ndarray: lambda v: v.tolist(),
        }

    @property
    def width(self) -> float:
        return self.frame.shape[1] if self.frame is not None else 0

    @property
    def height(self) -> float:
        return self.frame.shape[0] if self.frame is not None else 0


class AlignedFrame(BaseModel):
    frame: Frame = Field(..., description="Original frame metadata")
    aligned_image_path: str = Field(..., description="Path to aligned/warped image")
    shelf_id: str = Field(..., description="Matched shelf identifier")
    confidence: float = Field(..., ge=0, le=1, description="Alignment confidence")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }
