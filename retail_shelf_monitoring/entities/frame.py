from datetime import datetime
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field


class Frame(BaseModel):
    frame_id: str = Field(..., description="Unique frame identifier")
    stream_id: str = Field(..., description="Source stream identifier")
    timestamp: datetime = Field(..., description="Frame capture timestamp")
    frame_number: int = Field(..., ge=0, description="Sequential frame number")
    width: int = Field(..., gt=0, description="Frame width in pixels")
    height: int = Field(..., gt=0, description="Frame height in pixels")
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


class AlignedFrame(BaseModel):
    frame: Frame = Field(..., description="Original frame metadata")
    aligned_image_path: str = Field(..., description="Path to aligned/warped image")
    shelf_id: str = Field(..., description="Matched shelf identifier")
    confidence: float = Field(..., ge=0, le=1, description="Alignment confidence")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }
