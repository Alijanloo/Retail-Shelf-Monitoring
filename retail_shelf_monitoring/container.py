from dependency_injector import containers, providers
from .frameworks.config import AppConfig
from .frameworks.database import DatabaseManager
from .frameworks.logging_config import get_logger


class ApplicationContainer(containers.DeclarativeContainer):
    wiring_config = containers.WiringConfiguration(
        modules=[
            "retail_shelf_monitoring.__main__",
        ]
    )

    config = providers.Singleton(AppConfig.from_yaml_or_default)

    logger = providers.Singleton(
        get_logger,
        name="retail_shelf_monitoring"
    )

    database_manager = providers.Singleton(
        DatabaseManager,
        database_url=config.provided.database.url
    )


class TestContainer(containers.DeclarativeContainer):
    """Test-specific dependency injection container with mocked dependencies."""

    # Configuration with test defaults
    config = providers.Configuration()

    # Mock dependencies for testing
    # Add your mock providers here
    # Example:
    # mock_database_client = providers.Factory(
    #     lambda: None,  # Mock database client for testing
    # )

    # mock_user_repository = providers.Factory(
    #     lambda: None,  # Mock repository for testing
    # )

    # mock_user_service = providers.Factory(
    #     lambda: None,  # Mock service for testing
    # )
