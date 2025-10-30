from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator

from .common import BoundingBox


class PlanogramItem(BaseModel):
    item_idx: int = Field(..., ge=0, description="Item index within row")
    bbox: BoundingBox = Field(..., description="Bounding box in reference coords")
    sku_id: str = Field(..., description="Expected SKU at this position")
    confidence: float = Field(
        default=1.0, ge=0, le=1, description="Detection confidence in reference"
    )

    class Config:
        frozen = True


class PlanogramRow(BaseModel):
    row_idx: int = Field(..., ge=0, description="Row index (0 = top)")
    avg_y: float = Field(..., ge=0, description="Average Y coordinate of row")
    items: List[PlanogramItem] = Field(
        ..., min_items=1, description="Items in this row"
    )

    @validator("items")
    def items_sorted_by_x(cls, v):
        sorted_items = sorted(v, key=lambda item: item.bbox.x1)
        for idx, item in enumerate(sorted_items):
            if item.item_idx != idx:
                raise ValueError(
                    f"Item indices must be sequential from 0; found {item.item_idx}"
                    " at position {idx}"
                )
        return sorted_items

    class Config:
        frozen = True


class PlanogramGrid(BaseModel):
    rows: List[PlanogramRow] = Field(
        ..., min_items=1, description="Rows in the planogram"
    )

    @validator("rows")
    def rows_sorted_by_y(cls, v):
        sorted_rows = sorted(v, key=lambda row: row.avg_y)
        for idx, row in enumerate(sorted_rows):
            if row.row_idx != idx:
                raise ValueError(
                    f"Row indices must be sequential from 0; found {row.row_idx}"
                    " at position {idx}"
                )
        return sorted_rows

    @property
    def total_items(self) -> int:
        return sum(len(row.items) for row in self.rows)

    def get_cell(self, row_idx: int, item_idx: int) -> Optional[PlanogramItem]:
        if 0 <= row_idx < len(self.rows):
            row = self.rows[row_idx]
            if 0 <= item_idx < len(row.items):
                return row.items[item_idx]
        return None

    class Config:
        frozen = True


class ClusteringParams(BaseModel):
    row_clustering_method: str = Field(
        default="dbscan", description="Clustering method (dbscan, kmeans)"
    )
    eps: float = Field(default=15.0, gt=0, description="DBSCAN epsilon parameter")
    min_samples: int = Field(
        default=2, ge=1, description="DBSCAN min_samples parameter"
    )

    class Config:
        frozen = True


class Planogram(BaseModel):
    shelf_id: str = Field(..., description="Associated shelf identifier")
    reference_image_path: str = Field(..., description="Path to reference image")
    grid: PlanogramGrid = Field(..., description="Grid structure with expected SKUs")
    clustering_params: ClusteringParams = Field(
        ..., description="Parameters used for grid generation"
    )
    meta: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )
    created_at: datetime = Field(default_factory=datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=datetime.now(timezone.utc))

    class Config:
        frozen = False
        json_encoders = {datetime: lambda v: v.isoformat()}
