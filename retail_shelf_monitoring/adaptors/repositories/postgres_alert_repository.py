from datetime import datetime
from typing import List, Optional

from sqlalchemy import and_
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ...entities.alert import Alert
from ...entities.common import AlertType
from ...frameworks.database import AlertModel, AlertTypeEnum
from ...frameworks.exceptions import DatabaseError, EntityNotFoundError
from ...frameworks.logging_config import get_logger
from ...usecases.interfaces.repositories import AlertRepository

logger = get_logger(__name__)


class PostgresAlertRepository(AlertRepository):
    def __init__(self, session_factory):
        self.session_factory = session_factory

    async def create(self, alert: Alert) -> Alert:
        session: Session = self.session_factory()
        try:
            db_alert = AlertModel(
                alert_id=alert.alert_id,
                shelf_id=alert.shelf_id,
                row_idx=alert.row_idx,
                item_idx=alert.item_idx,
                alert_type=AlertTypeEnum[alert.alert_type.name],
                expected_sku=alert.expected_sku,
                detected_sku=alert.detected_sku,
                first_seen=alert.first_seen,
                last_seen=alert.last_seen,
                confirmed=alert.confirmed,
                confirmed_by=alert.confirmed_by,
                confirmed_at=alert.confirmed_at,
                dismissed=alert.dismissed,
                evidence_paths=alert.evidence_paths,
                consecutive_frames=alert.consecutive_frames,
                created_at=alert.created_at,
                updated_at=alert.updated_at,
            )

            session.add(db_alert)
            session.commit()
            session.refresh(db_alert)

            return self._to_entity(db_alert)

        except IntegrityError as e:
            session.rollback()
            raise DatabaseError(f"Failed to create alert: {str(e)}")
        finally:
            session.close()

    async def get_by_id(self, alert_id: str) -> Optional[Alert]:
        session: Session = self.session_factory()
        try:
            db_alert = session.query(AlertModel).filter_by(alert_id=alert_id).first()
            return self._to_entity(db_alert) if db_alert else None
        finally:
            session.close()

    async def get_active_alerts(self, shelf_id: Optional[str] = None) -> List[Alert]:
        session: Session = self.session_factory()
        try:
            query = session.query(AlertModel).filter(
                and_(~AlertModel.confirmed, ~AlertModel.dismissed)
            )

            if shelf_id:
                query = query.filter_by(shelf_id=shelf_id)

            db_alerts = query.all()
            return [self._to_entity(db_alert) for db_alert in db_alerts]
        finally:
            session.close()

    async def get_by_cell(
        self, shelf_id: str, row_idx: int, item_idx: int
    ) -> Optional[Alert]:
        session: Session = self.session_factory()
        try:
            db_alert = (
                session.query(AlertModel)
                .filter(
                    and_(
                        AlertModel.shelf_id == shelf_id,
                        AlertModel.row_idx == row_idx,
                        AlertModel.item_idx == item_idx,
                        ~AlertModel.dismissed,
                    )
                )
                .first()
            )

            return self._to_entity(db_alert) if db_alert else None
        finally:
            session.close()

    async def update(self, alert: Alert) -> Alert:
        session: Session = self.session_factory()
        try:
            db_alert = (
                session.query(AlertModel).filter_by(alert_id=alert.alert_id).first()
            )
            if not db_alert:
                raise EntityNotFoundError("Alert", alert.alert_id)

            db_alert.last_seen = alert.last_seen
            db_alert.consecutive_frames = alert.consecutive_frames
            db_alert.evidence_paths = alert.evidence_paths
            db_alert.updated_at = datetime.utcnow()

            session.commit()
            session.refresh(db_alert)

            return self._to_entity(db_alert)

        except IntegrityError as e:
            session.rollback()
            raise DatabaseError(f"Failed to update alert: {str(e)}")
        finally:
            session.close()

    async def confirm_alert(self, alert_id: str, confirmed_by: str) -> Alert:
        session: Session = self.session_factory()
        try:
            db_alert = session.query(AlertModel).filter_by(alert_id=alert_id).first()
            if not db_alert:
                raise EntityNotFoundError("Alert", alert_id)

            db_alert.confirmed = True
            db_alert.confirmed_by = confirmed_by
            db_alert.confirmed_at = datetime.utcnow()
            db_alert.updated_at = datetime.utcnow()

            session.commit()
            session.refresh(db_alert)

            logger.info(f"Alert {alert_id} confirmed by {confirmed_by}")

            return self._to_entity(db_alert)

        except IntegrityError as e:
            session.rollback()
            raise DatabaseError(f"Failed to confirm alert: {str(e)}")
        finally:
            session.close()

    async def dismiss_alert(self, alert_id: str) -> Alert:
        session: Session = self.session_factory()
        try:
            db_alert = session.query(AlertModel).filter_by(alert_id=alert_id).first()
            if not db_alert:
                raise EntityNotFoundError("Alert", alert_id)

            db_alert.dismissed = True
            db_alert.updated_at = datetime.utcnow()

            session.commit()
            session.refresh(db_alert)

            logger.info(f"Alert {alert_id} dismissed")

            return self._to_entity(db_alert)

        except IntegrityError as e:
            session.rollback()
            raise DatabaseError(f"Failed to dismiss alert: {str(e)}")
        finally:
            session.close()

    def _to_entity(self, db_alert: AlertModel) -> Alert:
        return Alert(
            alert_id=db_alert.alert_id,
            shelf_id=db_alert.shelf_id,
            row_idx=db_alert.row_idx,
            item_idx=db_alert.item_idx,
            alert_type=AlertType[db_alert.alert_type.name],
            expected_sku=db_alert.expected_sku,
            detected_sku=db_alert.detected_sku,
            first_seen=db_alert.first_seen,
            last_seen=db_alert.last_seen,
            confirmed=db_alert.confirmed,
            confirmed_by=db_alert.confirmed_by,
            confirmed_at=db_alert.confirmed_at,
            dismissed=db_alert.dismissed,
            evidence_paths=db_alert.evidence_paths,
            consecutive_frames=db_alert.consecutive_frames,
            created_at=db_alert.created_at,
            updated_at=db_alert.updated_at,
        )
