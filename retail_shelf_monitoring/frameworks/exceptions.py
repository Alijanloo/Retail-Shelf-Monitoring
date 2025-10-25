class RetailShelfMonitoringError(Exception):
    pass


class EntityNotFoundError(RetailShelfMonitoringError):
    def __init__(self, entity_type: str, entity_id: str):
        self.entity_type = entity_type
        self.entity_id = entity_id
        super().__init__(f"{entity_type} with ID '{entity_id}' not found")


class EntityAlreadyExistsError(RetailShelfMonitoringError):
    def __init__(self, entity_type: str, entity_id: str):
        self.entity_type = entity_type
        self.entity_id = entity_id
        super().__init__(f"{entity_type} with ID '{entity_id}' already exists")


class ValidationError(RetailShelfMonitoringError):
    pass


class DatabaseError(RetailShelfMonitoringError):
    pass


class CacheError(RetailShelfMonitoringError):
    pass


class ConfigurationError(RetailShelfMonitoringError):
    pass
