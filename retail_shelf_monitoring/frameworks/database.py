import enum
from datetime import datetime

from sqlalchemy import JSON, Boolean, Column, DateTime
from sqlalchemy import Enum as SQLEnum
from sqlalchemy import ForeignKey, Index, Integer, String, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.pool import NullPool

Base = declarative_base()


class AlertTypeEnum(enum.Enum):
    OOS = "out_of_stock"
    MISPLACEMENT = "misplacement"
    UNKNOWN = "unknown"


class PlanogramModel(Base):
    __tablename__ = "planograms"

    shelf_id = Column(String(255), primary_key=True)
    reference_image_path = Column(String(500), nullable=False)
    grid = Column(JSON, nullable=False)
    clustering_params = Column(JSON, nullable=False)
    meta = Column(JSON, default={})
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )


class AlertModel(Base):
    __tablename__ = "alerts"

    alert_id = Column(String(255), primary_key=True)
    shelf_id = Column(
        String(255),
        ForeignKey("planograms.shelf_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    row_idx = Column(Integer, nullable=False)
    item_idx = Column(Integer, nullable=False)
    alert_type = Column(SQLEnum(AlertTypeEnum), nullable=False)
    expected_sku = Column(String(255))
    detected_sku = Column(String(255))
    first_seen = Column(DateTime, nullable=False)
    last_seen = Column(DateTime, nullable=False)
    confirmed = Column(Boolean, default=False, nullable=False)
    confirmed_by = Column(String(255))
    confirmed_at = Column(DateTime)
    dismissed = Column(Boolean, default=False, nullable=False)
    evidence_paths = Column(JSON, default=[])
    consecutive_frames = Column(Integer, default=1, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    shelf = relationship("PlanogramModel", back_populates="alerts")

    __table_args__ = (
        Index("idx_alert_shelf_active", "shelf_id", "confirmed", "dismissed"),
        Index("idx_alert_cell", "shelf_id", "row_idx", "item_idx"),
    )


class SKUModel(Base):
    __tablename__ = "skus"

    sku_id = Column(String(255), primary_key=True)
    name = Column(String(500), nullable=False)
    category = Column(String(255))
    barcode = Column(String(255))
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    __table_args__ = (Index("idx_sku_category", "category"),)


class DatabaseManager:
    def __init__(self, database_url: str):
        self.engine = create_engine(
            database_url,
            pool_pre_ping=True,
            pool_size=10,
            max_overflow=20,
            echo=False,
            poolclass=NullPool if "sqlite" in database_url else None,
        )
        self.SessionLocal = sessionmaker(
            autocommit=False, autoflush=False, bind=self.engine
        )

    def create_tables(self):
        Base.metadata.create_all(bind=self.engine)

    def drop_tables(self):
        Base.metadata.drop_all(bind=self.engine)

    def get_session(self):
        return self.SessionLocal()
