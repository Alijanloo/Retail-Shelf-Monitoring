from typing import List

from ..entities.shelf import Shelf
from ..frameworks.exceptions import EntityNotFoundError
from ..frameworks.logging_config import get_logger
from .interfaces.repositories import ShelfRepository

logger = get_logger(__name__)


class ShelfManagementUseCase:
    def __init__(
        self,
        shelf_repository: ShelfRepository,
    ):
        self.shelf_repository = shelf_repository

    async def create_shelf(self, shelf: Shelf) -> Shelf:
        logger.info(f"Creating shelf: {shelf.shelf_id}")
        created_shelf = await self.shelf_repository.create(shelf)
        logger.info(f"Successfully created shelf: {shelf.shelf_id}")
        return created_shelf

    async def get_shelf(self, shelf_id: str) -> Shelf:
        logger.info(f"Retrieving shelf: {shelf_id}")
        shelf = await self.shelf_repository.get_by_id(shelf_id)
        if not shelf:
            raise EntityNotFoundError("Shelf", shelf_id)
        return shelf

    async def get_all_shelves(self, active_only: bool = True) -> List[Shelf]:
        logger.info(f"Retrieving all shelves (active_only={active_only})")
        shelves = await self.shelf_repository.get_all(active_only=active_only)
        logger.info(f"Retrieved {len(shelves)} shelves")
        return shelves

    async def update_shelf(self, shelf: Shelf) -> Shelf:
        logger.info(f"Updating shelf: {shelf.shelf_id}")
        updated_shelf = await self.shelf_repository.update(shelf)
        logger.info(f"Successfully updated shelf: {shelf.shelf_id}")
        return updated_shelf

    async def delete_shelf(self, shelf_id: str) -> bool:
        logger.info(f"Deleting shelf: {shelf_id}")
        deleted = await self.shelf_repository.delete(shelf_id)
        if deleted:
            logger.info(f"Successfully deleted shelf: {shelf_id}")
        else:
            logger.warning(f"Shelf not found for deletion: {shelf_id}")
        return deleted

    async def activate_shelf(self, shelf_id: str) -> Shelf:
        logger.info(f"Activating shelf: {shelf_id}")
        shelf = await self.get_shelf(shelf_id)
        shelf.active = True
        return await self.shelf_repository.update(shelf)

    async def deactivate_shelf(self, shelf_id: str) -> Shelf:
        logger.info(f"Deactivating shelf: {shelf_id}")
        shelf = await self.get_shelf(shelf_id)
        shelf.active = False
        return await self.shelf_repository.update(shelf)
