from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime, Text, JSON, 
    ForeignKey, Index, Enum as SQLEnum, create_engine
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.pool import NullPool
from datetime import datetime
import enum

Base = declarative_base()


class PriorityEnum(enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertTypeEnum(enum.Enum):
    OOS = "out_of_stock"
    MISPLACEMENT = "misplacement"
    UNKNOWN = "unknown"


class ShelfModel(Base):
    __tablename__ = "shelves"

    shelf_id = Column(String(255), primary_key=True)
    store_id = Column(String(255), nullable=False, index=True)
    aisle = Column(String(100))
    section = Column(String(100))
    priority = Column(SQLEnum(PriorityEnum), default=PriorityEnum.MEDIUM, nullable=False)
    active = Column(Boolean, default=True, nullable=False)
    meta = Column(JSON, default={})
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    planogram = relationship("PlanogramModel", back_populates="shelf", uselist=False, cascade="all, delete-orphan")
    detections = relationship("DetectionModel", back_populates="shelf", cascade="all, delete-orphan")
    alerts = relationship("AlertModel", back_populates="shelf", cascade="all, delete-orphan")

    __table_args__ = (
        Index('idx_shelf_store_active', 'store_id', 'active'),
    )


class PlanogramModel(Base):
    __tablename__ = "planograms"

    shelf_id = Column(String(255), ForeignKey("shelves.shelf_id", ondelete="CASCADE"), primary_key=True)
    reference_image_path = Column(String(500), nullable=False)
    grid = Column(JSON, nullable=False)
    clustering_params = Column(JSON, nullable=False)
    meta = Column(JSON, default={})
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    shelf = relationship("ShelfModel", back_populates="planogram")


class DetectionModel(Base):
    __tablename__ = "detections"

    detection_id = Column(String(255), primary_key=True)
    shelf_id = Column(String(255), ForeignKey("shelves.shelf_id", ondelete="CASCADE"), nullable=False, index=True)
    frame_timestamp = Column(DateTime, nullable=False, index=True)
    bbox_x1 = Column(Float, nullable=False)
    bbox_y1 = Column(Float, nullable=False)
    bbox_x2 = Column(Float, nullable=False)
    bbox_y2 = Column(Float, nullable=False)
    class_id = Column(Integer, nullable=False)
    sku_id = Column(String(255))
    confidence = Column(Float, nullable=False)
    track_id = Column(Integer)
    row_idx = Column(Integer)
    item_idx = Column(Integer)
    aligned_frame_path = Column(String(500))
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    shelf = relationship("ShelfModel", back_populates="detections")

    __table_args__ = (
        Index('idx_detection_shelf_timestamp', 'shelf_id', 'frame_timestamp'),
        Index('idx_detection_cell', 'shelf_id', 'row_idx', 'item_idx'),
        Index('idx_detection_track', 'track_id'),
    )


class AlertModel(Base):
    __tablename__ = "alerts"

    alert_id = Column(String(255), primary_key=True)
    shelf_id = Column(String(255), ForeignKey("shelves.shelf_id", ondelete="CASCADE"), nullable=False, index=True)
    row_idx = Column(Integer, nullable=False)
    item_idx = Column(Integer, nullable=False)
    alert_type = Column(SQLEnum(AlertTypeEnum), nullable=False)
    expected_sku = Column(String(255))
    detected_sku = Column(String(255))
    priority = Column(SQLEnum(PriorityEnum), default=PriorityEnum.MEDIUM, nullable=False)
    first_seen = Column(DateTime, nullable=False)
    last_seen = Column(DateTime, nullable=False)
    confirmed = Column(Boolean, default=False, nullable=False)
    confirmed_by = Column(String(255))
    confirmed_at = Column(DateTime)
    dismissed = Column(Boolean, default=False, nullable=False)
    evidence_paths = Column(JSON, default=[])
    consecutive_frames = Column(Integer, default=1, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    shelf = relationship("ShelfModel", back_populates="alerts")

    __table_args__ = (
        Index('idx_alert_shelf_active', 'shelf_id', 'confirmed', 'dismissed'),
        Index('idx_alert_cell', 'shelf_id', 'row_idx', 'item_idx'),
        Index('idx_alert_type_priority', 'alert_type', 'priority'),
    )


class SKUModel(Base):
    __tablename__ = "skus"

    sku_id = Column(String(255), primary_key=True)
    name = Column(String(500), nullable=False)
    category = Column(String(255))
    barcode = Column(String(255))
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    __table_args__ = (
        Index('idx_sku_category', 'category'),
    )


class DatabaseManager:
    def __init__(self, database_url: str):
        self.engine = create_engine(
            database_url,
            pool_pre_ping=True,
            pool_size=10,
            max_overflow=20,
            echo=False,
            poolclass=NullPool if "sqlite" in database_url else None
        )
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )

    def create_tables(self):
        Base.metadata.create_all(bind=self.engine)

    def drop_tables(self):
        Base.metadata.drop_all(bind=self.engine)

    def get_session(self):
        return self.SessionLocal()
