from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field


class DatabaseConfig(BaseModel):
    host: str = Field(default="localhost")
    port: int = Field(default=5432)
    database: str = Field(default="retail_shelf_monitoring")
    user: str = Field(default="postgres")
    password: str = Field(default="postgres")

    @property
    def url(self) -> str:
        return (
            f"postgresql://{self.user}:{self.password}@"
            f"{self.host}:{self.port}/{self.database}"
        )


class RedisConfig(BaseModel):
    host: str = Field(default="localhost")
    port: int = Field(default=6379)
    db: int = Field(default=0)
    password: Optional[str] = Field(default=None)

    @property
    def url(self) -> str:
        if self.password:
            return f"redis://:{self.password}@{self.host}:{self.port}/{self.db}"
        return f"redis://{self.host}:{self.port}/{self.db}"


class LoggingConfig(BaseModel):
    level: str = Field(default="INFO")
    format: str = Field(default="json")
    file_path: Optional[str] = Field(default=None)


class MLConfig(BaseModel):
    model_path: str = Field(default="models/yolov11_retail.xml")
    confidence_threshold: float = Field(default=0.35, ge=0, le=1)
    nms_threshold: float = Field(default=0.45, ge=0, le=1)
    device: str = Field(default="CPU")


class GridConfig(BaseModel):
    clustering_method: str = Field(default="dbscan")
    eps: float = Field(default=15.0, gt=0)
    min_samples: int = Field(default=2, ge=1)
    position_tolerance: int = Field(default=1, ge=0)


class TrackingConfig(BaseModel):
    track_thresh: float = Field(default=0.5, ge=0, le=1)
    match_thresh: float = Field(default=0.3, ge=0, le=1)
    max_age: int = Field(default=30, ge=1)
    sku_mapping_file: Optional[str] = Field(default=None)


class StreamingConfig(BaseModel):
    max_queue_size: int = Field(default=100, ge=1)
    frame_buffer_size: int = Field(default=100, ge=1)
    process_every_n_frames: int = Field(default=30, ge=1)
    max_width: int = Field(default=1920, gt=0)
    max_height: int = Field(default=1080, gt=0)
    enable_stabilization: bool = Field(default=False)


class FeatureMatchingConfig(BaseModel):
    feature_type: str = Field(default="orb")
    max_features: int = Field(default=5000, gt=0)
    match_threshold: float = Field(default=0.75, ge=0, le=1)
    min_matches: int = Field(default=10, ge=1)


class HomographyConfig(BaseModel):
    ransac_reproj_threshold: float = Field(default=5.0, gt=0)
    min_inlier_ratio: float = Field(default=0.3, ge=0, le=1)
    min_inliers: int = Field(default=10, ge=1)
    max_iterations: int = Field(default=2000, gt=0)
    min_alignment_confidence: float = Field(default=0.3, ge=0, le=1)


class AlignmentConfig(BaseModel):
    output_dir: str = Field(default="data/aligned_frames")


class AppConfig(BaseModel):
    app_name: str = Field(default="Retail Shelf Monitoring")
    debug: bool = Field(default=False)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    ml: MLConfig = Field(default_factory=MLConfig)
    grid: GridConfig = Field(default_factory=GridConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    streaming: StreamingConfig = Field(default_factory=StreamingConfig)
    feature_matching: FeatureMatchingConfig = Field(
        default_factory=FeatureMatchingConfig
    )
    homography: HomographyConfig = Field(default_factory=HomographyConfig)
    alignment: AlignmentConfig = Field(default_factory=AlignmentConfig)

    @classmethod
    def from_yaml(cls, config_path: str) -> "AppConfig":
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(path, "r") as f:
            config_dict = yaml.safe_load(f) or {}

        return cls(**config_dict)

    @classmethod
    def from_yaml_or_default(cls, config_path: str = "config.yaml") -> "AppConfig":
        try:
            return cls.from_yaml(config_path)
        except FileNotFoundError:
            return cls()
