import asyncio
import sys

from template_project.frameworks.logging_config import get_logger
from template_project.container import ApplicationContainer

logger = get_logger(__name__)


async def main():
    try:
        ApplicationContainer()
    except Exception as e:
        logger.error(f"Critical error in main: {e}")
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Application terminated by user")
    except Exception as e:
        logger.error(f"Application failed: {e}")
        sys.exit(1)
