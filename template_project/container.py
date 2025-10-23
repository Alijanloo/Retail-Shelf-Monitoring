from dependency_injector import containers, providers


class ApplicationContainer(containers.DeclarativeContainer):
    """Main dependency injection container for the application."""

    # Configuration
    config = providers.Configuration(yaml_files=["config.yaml"])

    # Core Services
    # Add your service providers here as needed
    # Example:
    # database_client = providers.Resource(
    #     database_client_resource,
    #     connection_string=config.database.connection_string,
    # )

    # Repository Layer
    # Add your repository providers here
    # Example:
    # user_repository = providers.Singleton(
    #     UserRepository,
    #     database_client=database_client,
    # )

    # Service Layer  
    # Add your service providers here
    # Example:
    # user_service = providers.Factory(
    #     UserService,
    #     user_repository=user_repository,
    # )


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
