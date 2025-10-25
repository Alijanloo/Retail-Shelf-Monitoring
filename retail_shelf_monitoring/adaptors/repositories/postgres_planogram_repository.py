from typing import List, Optional

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ...entities.planogram import ClusteringParams, Planogram, PlanogramGrid
from ...frameworks.database import PlanogramModel
from ...frameworks.exceptions import DatabaseError, EntityNotFoundError
from ...frameworks.logging_config import get_logger
from ...usecases.interfaces.repositories import PlanogramRepository

logger = get_logger(__name__)


class PostgresPlanogramRepository(PlanogramRepository):
    def __init__(self, session_factory):
        self.session_factory = session_factory

    async def create(self, planogram: Planogram) -> Planogram:
        session: Session = self.session_factory()
        try:
            existing = (
                session.query(PlanogramModel)
                .filter_by(shelf_id=planogram.shelf_id)
                .first()
            )
            if existing:
                return await self.update(planogram)

            grid_json = planogram.grid.dict()
            clustering_params_json = planogram.clustering_params.dict()

            db_planogram = PlanogramModel(
                shelf_id=planogram.shelf_id,
                reference_image_path=planogram.reference_image_path,
                grid=grid_json,
                clustering_params=clustering_params_json,
                meta=planogram.meta,
                created_at=planogram.created_at,
                updated_at=planogram.updated_at,
            )

            session.add(db_planogram)
            session.commit()
            session.refresh(db_planogram)

            logger.info(f"Created planogram for shelf: {planogram.shelf_id}")
            return self._to_entity(db_planogram)

        except IntegrityError as e:
            session.rollback()
            raise DatabaseError(f"Failed to create planogram: {str(e)}")
        finally:
            session.close()

    async def get_by_shelf_id(self, shelf_id: str) -> Optional[Planogram]:
        session: Session = self.session_factory()
        try:
            db_planogram = (
                session.query(PlanogramModel).filter_by(shelf_id=shelf_id).first()
            )
            return self._to_entity(db_planogram) if db_planogram else None
        finally:
            session.close()

    async def get_all(self) -> List[Planogram]:
        session: Session = self.session_factory()
        try:
            db_planograms = session.query(PlanogramModel).all()
            return [self._to_entity(db_planogram) for db_planogram in db_planograms]
        finally:
            session.close()

    async def update(self, planogram: Planogram) -> Planogram:
        session: Session = self.session_factory()
        try:
            db_planogram = (
                session.query(PlanogramModel)
                .filter_by(shelf_id=planogram.shelf_id)
                .first()
            )
            if not db_planogram:
                raise EntityNotFoundError("Planogram", planogram.shelf_id)

            db_planogram.reference_image_path = planogram.reference_image_path
            db_planogram.grid = planogram.grid.dict()
            db_planogram.clustering_params = planogram.clustering_params.dict()
            db_planogram.meta = planogram.meta
            db_planogram.updated_at = planogram.updated_at

            session.commit()
            session.refresh(db_planogram)

            logger.info(f"Updated planogram for shelf: {planogram.shelf_id}")
            return self._to_entity(db_planogram)

        except IntegrityError as e:
            session.rollback()
            raise DatabaseError(f"Failed to update planogram: {str(e)}")
        finally:
            session.close()

    async def delete(self, shelf_id: str) -> bool:
        session: Session = self.session_factory()
        try:
            db_planogram = (
                session.query(PlanogramModel).filter_by(shelf_id=shelf_id).first()
            )
            if not db_planogram:
                return False

            session.delete(db_planogram)
            session.commit()

            logger.info(f"Deleted planogram for shelf: {shelf_id}")
            return True

        except IntegrityError as e:
            session.rollback()
            raise DatabaseError(f"Failed to delete planogram: {str(e)}")
        finally:
            session.close()

    def _to_entity(self, db_planogram: PlanogramModel) -> Planogram:
        grid = PlanogramGrid(**db_planogram.grid)
        clustering_params = ClusteringParams(**db_planogram.clustering_params)

        return Planogram(
            shelf_id=db_planogram.shelf_id,
            reference_image_path=db_planogram.reference_image_path,
            grid=grid,
            clustering_params=clustering_params,
            meta=db_planogram.meta,
            created_at=db_planogram.created_at,
            updated_at=db_planogram.updated_at,
        )
