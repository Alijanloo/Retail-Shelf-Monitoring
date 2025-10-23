# Dependency Injection System

This document describes the dependency injection (DI) system implemented in the MultiModal RAG project using the `dependency-injector` library.

## Overview

The dependency injection container provides centralized configuration and dependency management for the entire application, making it easier to:

- Configure services through environment variables or configuration files
- Mock dependencies for testing
- Manage singleton instances of expensive resources
- Maintain clean separation of concerns


## Usage Examples

### TestContainer

A separate test container provides mocked dependencies for testing:

```python
from template_project.containers import TestContainer

test_container = TestContainer()
# All dependencies are mocked for isolated testing
```

### Basic Usage with Injection

```python
from dependency_injector.wiring import Provide, inject
from template_project.containers import ApplicationContainer
from template_project.usecases.document_indexing import DocumentIndexingUseCase

@inject
async def index_document(
    indexing_use_case: DocumentIndexingUseCase = Provide[ApplicationContainer.document_indexing_use_case],
    document: DoclingDocument,
    document_id: str
) -> None:
    result = await indexing_use_case.index_document(document, document_id)
    print(f"Indexed: {result.success}")

# Setup container and wire
container = ApplicationContainer()
container.wire(modules=[__name__])

# Function automatically gets dependencies injected
await index_document(my_document, "doc_123")
```

### Testing with Mocks

```python
import unittest.mock as mock
from template_project.containers import ApplicationContainer

def test_document_indexing():
    container = ApplicationContainer()
    
    # Mock the repository
    mock_repository = mock.AsyncMock()
    mock_repository.index_document.return_value = IndexDocumentResponse(
        document_id="test", success=True, message="Success"
    )
    
    # Override dependency
    with container.document_repository.override(mock_repository):
        container.wire(modules=[__name__])
        
        # Test your code - mock will be injected automatically
        indexing_use_case = container.document_indexing_use_case()
        # ... test logic
```

## Common Provider Types

1. **Factory**: Creates new instance each time
2. **Singleton**: Creates one instance, reuses it
3. **Resource**: Manages lifecycle of expensive objects
4. **Configuration**: Provides configuration values
5. **Callable**: Wraps functions/callables

The key insight is that `Resource` is specifically designed for objects that need both creation AND cleanup, making it perfect for database connections, file handles, network clients, etc.
