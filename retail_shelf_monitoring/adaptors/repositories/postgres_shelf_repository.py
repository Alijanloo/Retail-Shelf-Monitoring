from typing import List, Optional

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ...entities.common import Priority
from ...entities.shelf import Shelf
from ...frameworks.database import PriorityEnum, ShelfModel
from ...frameworks.exceptions import (
    DatabaseError,
    EntityAlreadyExistsError,
    EntityNotFoundError,
)
from ...frameworks.logging_config import get_logger
from ...usecases.interfaces.repositories import ShelfRepository

logger = get_logger(__name__)


class PostgresShelfRepository(ShelfRepository):
    def __init__(self, session_factory):
        self.session_factory = session_factory

    async def create(self, shelf: Shelf) -> Shelf:
        session: Session = self.session_factory()
        try:
            existing = (
                session.query(ShelfModel).filter_by(shelf_id=shelf.shelf_id).first()
            )
            if existing:
                raise EntityAlreadyExistsError("Shelf", shelf.shelf_id)

            db_shelf = ShelfModel(
                shelf_id=shelf.shelf_id,
                store_id=shelf.store_id,
                aisle=shelf.aisle,
                section=shelf.section,
                priority=PriorityEnum(shelf.priority.value),
                active=shelf.active,
                meta=shelf.meta,
                created_at=shelf.created_at,
                updated_at=shelf.updated_at,
            )

            session.add(db_shelf)
            session.commit()
            session.refresh(db_shelf)

            logger.info(f"Created shelf: {shelf.shelf_id}")
            return self._to_entity(db_shelf)

        except IntegrityError as e:
            session.rollback()
            raise DatabaseError(f"Failed to create shelf: {str(e)}")
        finally:
            session.close()

    async def get_by_id(self, shelf_id: str) -> Optional[Shelf]:
        session: Session = self.session_factory()
        try:
            db_shelf = session.query(ShelfModel).filter_by(shelf_id=shelf_id).first()
            return self._to_entity(db_shelf) if db_shelf else None
        finally:
            session.close()

    async def get_all(self, active_only: bool = True) -> List[Shelf]:
        session: Session = self.session_factory()
        try:
            query = session.query(ShelfModel)
            if active_only:
                query = query.filter_by(active=True)

            db_shelves = query.all()
            return [self._to_entity(db_shelf) for db_shelf in db_shelves]
        finally:
            session.close()

    async def update(self, shelf: Shelf) -> Shelf:
        session: Session = self.session_factory()
        try:
            db_shelf = (
                session.query(ShelfModel).filter_by(shelf_id=shelf.shelf_id).first()
            )
            if not db_shelf:
                raise EntityNotFoundError("Shelf", shelf.shelf_id)

            db_shelf.store_id = shelf.store_id
            db_shelf.aisle = shelf.aisle
            db_shelf.section = shelf.section
            db_shelf.priority = PriorityEnum(shelf.priority.value)
            db_shelf.active = shelf.active
            db_shelf.meta = shelf.meta
            db_shelf.updated_at = shelf.updated_at

            session.commit()
            session.refresh(db_shelf)

            logger.info(f"Updated shelf: {shelf.shelf_id}")
            return self._to_entity(db_shelf)

        except IntegrityError as e:
            session.rollback()
            raise DatabaseError(f"Failed to update shelf: {str(e)}")
        finally:
            session.close()

    async def delete(self, shelf_id: str) -> bool:
        session: Session = self.session_factory()
        try:
            db_shelf = session.query(ShelfModel).filter_by(shelf_id=shelf_id).first()
            if not db_shelf:
                return False

            session.delete(db_shelf)
            session.commit()

            logger.info(f"Deleted shelf: {shelf_id}")
            return True

        except IntegrityError as e:
            session.rollback()
            raise DatabaseError(f"Failed to delete shelf: {str(e)}")
        finally:
            session.close()

    def _to_entity(self, db_shelf: ShelfModel) -> Shelf:
        return Shelf(
            shelf_id=db_shelf.shelf_id,
            store_id=db_shelf.store_id,
            aisle=db_shelf.aisle,
            section=db_shelf.section,
            priority=Priority(db_shelf.priority.value),
            active=db_shelf.active,
            meta=db_shelf.meta,
            created_at=db_shelf.created_at,
            updated_at=db_shelf.updated_at,
        )
