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


class AppConfig(BaseModel):
    app_name: str = Field(default="Retail Shelf Monitoring")
    debug: bool = Field(default=False)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    ml: MLConfig = Field(default_factory=MLConfig)
    grid: GridConfig = Field(default_factory=GridConfig)

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
