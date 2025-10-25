# 🧠 Copilot Instructions: Clean Architecture

## Overview

This project implements a Clean Architecture approach, structuring the application into distinct layers to promote separation of concerns, scalability, and maintainability. The application is designed to be framework-agnostic and adaptable to various technologies and business domains.

## Architectural Layers

### 1. **Entities (Enterprise Business Rules)**

* **Purpose**: Define the core business objects and rules that are independent of external frameworks and technologies.
* **Examples**:

  * Data models representing domain concepts using Pydantic (e.g., `User`, `Product`, `Order`).
  * Validation logic and business rules with Pydantic validators.
  * Type-safe entities with automatic serialization/deserialization.

Consider an `Employee` entity in a payroll system:
```python
from pydantic import BaseModel, Field, validator

class Employee(BaseModel):
    name: str
    salary: float = Field(..., gt=0)
    tax_rate: float = Field(..., ge=0, le=1)

    @validator('name')
    def name_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError('Name cannot be empty')
        return v

    def calculate_net_salary(self) -> float:
        return self.salary * (1 - self.tax_rate)
```
In this example, the `Employee` entity uses Pydantic for data validation and encapsulates the core business logic for calculating an employee's net salary, independent of how the data is stored or presented.

### 2. **Use Cases (Application Business Rules)**

* **Purpose**: Encapsulate the application's specific business logic, orchestrating the flow between entities and interface adapters.
* **Examples**:

  * Handling user requests and orchestrating business operations.
  * Managing complex workflows and business processes.
  * Implementing domain-specific algorithms and computations.
  * Coordinating data processing and transformation tasks.


### 3. **Interface Adapters**

* **Purpose**: Translate data between the use cases and the external world, adapting interfaces for communication.
* **Examples**:

  * Controllers handling HTTP requests and responses.
  * Presenters formatting data for the UI or API consumers.
  * Gateways interfacing with external services like databases, vector stores, or third-party APIs.

### 4. **Frameworks and Drivers**

* **Purpose**: Contain the implementation details of external frameworks and tools. This layer is the most volatile and should be kept separate from the core business logic.
* **Examples**:

  * Web frameworks (e.g., FastAPI, Flask).
  * External service providers (e.g., payment processors, notification services).
  * Databases and storage systems (e.g., PostgreSQL, Redis, MongoDB).

## Placement of Interfaces

Interfaces play a crucial role in maintaining the independence of the core layers from external implementations. Here's how to manage them:

* **Define Interfaces in the Core Layers**: Interfaces that represent contracts for external dependencies (e.g., repositories, services) should be defined in the Use Cases layer. This allows the core application logic to remain agnostic of external implementations.
* **Implement Interfaces in the Outer Layers**: Concrete implementations of these interfaces should reside in the Interface Adapters or Frameworks and Drivers layers, depending on their nature. For instance, a repository interface defined in the Use Cases layer would have its implementation in the Interface Adapters layer, interacting with a specific database.

## Guidelines for Copilot

* **Maintain Layer Boundaries**: Ensure that code within a layer does not directly depend on layers outside its immediate neighbor. For instance, use cases should not depend on frameworks or drivers.
* **Dependency Rule**: Dependencies should always point inward, from outer layers to inner layers.
* **Use Dependency Inversion**: Rely on abstractions (e.g., interfaces or protocols) to invert dependencies, allowing inner layers to remain unaffected by changes in outer layers.
* **Isolate Frameworks**: Encapsulate framework-specific code within the Frameworks and Drivers layer to prevent it from leaking into core business logic.
* **Design for Testability**: Structure code to facilitate unit testing, particularly for use cases and entities, by avoiding tight coupling with external systems.


## Notes

* Use interfaces or abstract classes to allow the core application to remain agnostic of specific external providers or frameworks.
* Keep the core business logic free from external dependencies to enhance portability and maintainability.
* Use Pydantic models for all entities to ensure type safety, automatic validation, and consistent serialization/deserialization.
* Leverage Pydantic validators for business rule enforcement at the entity level.
* The `pytest` is the preferred package for testing.
