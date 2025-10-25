from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any
from datetime import datetime
from .common import Priority


class Shelf(BaseModel):
    shelf_id: str = Field(..., min_length=1, description="Unique shelf identifier")
    store_id: str = Field(..., description="Store identifier")
    aisle: Optional[str] = Field(None, description="Aisle location")
    section: Optional[str] = Field(None, description="Section within aisle")
    priority: Priority = Field(default=Priority.MEDIUM, description="Monitoring priority")
    active: bool = Field(default=True, description="Whether shelf is actively monitored")
    meta: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    @validator('shelf_id')
    def shelf_id_valid_format(cls, v):
        if not v or not v.strip():
            raise ValueError("shelf_id cannot be empty")
        return v.strip()

    class Config:
        frozen = False
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
