from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class SKU(BaseModel):
    sku_id: str = Field(..., description="Unique SKU identifier")
    name: str = Field(..., min_length=1, description="Product name")
    category: Optional[str] = Field(None, description="Product category")
    barcode: Optional[str] = Field(None, description="Product barcode")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        frozen = False
        json_encoders = {datetime: lambda v: v.isoformat()}
