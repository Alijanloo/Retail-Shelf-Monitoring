# Retail Shelf Monitoring System

**AI-Powered SKU Detection & Planogram Compliance for Real-Time Inventory Management**

An intelligent computer vision system that monitors retail shelves using CCTV cameras to detect out-of-stock (OOS) situations and product misplacements in real-time. The system automatically generates planograms from reference images and provides temporal consensus algorithms to minimize false alerts.

## 🏗️ Project Architecture

### Clean Architecture Implementation
This project follows **Clean Architecture** principles with strict dependency inversion:

```
┌─────────────────────────────────────────────────────────────┐
│                     FRAMEWORKS                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                   ADAPTORS                          │    │
│  │  ┌─────────────────────────────────────────────┐    │    │
│  │  │               USE CASES                     │    │    │
│  │  │  ┌─────────────────────────────────────┐    │    │    │
│  │  │  │            ENTITIES                 │    │    │    │
│  │  │  │  • SKU, Shelf, Planogram           │    │    │    │
│  │  │  │  • Detection, Alert                │    │    │    │
│  │  │  │  • Domain Business Rules           │    │    │    │
│  │  │  └─────────────────────────────────────┘    │    │    │
│  │  │  • Shelf monitoring workflows              │    │    │
│  │  │  • Detection & tracking logic              │    │    │
│  │  │  • Alert generation & validation           │    │    │
│  │  └─────────────────────────────────────────────┘    │    │
│  │  • Database repositories                            │    │
│  │  • Camera & stream adaptors                         │    │
│  │  • ML model interfaces                              │    │
│  │  • External service adaptors                        │    │
│  └─────────────────────────────────────────────────────┘    │
│  • Logging, configuration, database ORM                     │
│  • Dependency injection container                           │
│  • Web frameworks, async I/O                                │
└─────────────────────────────────────────────────────────────┘
```

**Key Principles:**
- **Entities**: Pure business logic with no external dependencies
- **Use Cases**: Application-specific business rules and workflows
- **Adaptors**: Interface implementations for external systems
- **Frameworks**: Tools, databases, web servers, logging, etc.

---

## 🔧 Core Technologies

### Backend Stack
- **Python 3.11**: Modern async/await, type hints, Pydantic 2.0+
- **dependency-injector**: Clean dependency injection pattern
- **SQLAlchemy 2.0+**: Database ORM with async support
- **PostgreSQL 15+**: Primary database for persistence
- **Redis 7+**: Caching, message queues, and real-time alerts
- **FastAPI**: High-performance async web framework
- **Pydantic**: Data validation and serialization

### Computer Vision & ML
- **YOLOv11**: State-of-the-art object detection for SKU identification
- **OpenCV**: Image processing, alignment, and feature matching
- **OpenVINO/ONNX**: Model optimization for edge deployment
- **scikit-learn**: Clustering algorithms for planogram generation
- **ByteTrack/SORT**: Multi-object tracking for temporal consistency

### Development & Testing
- **pytest**: Test framework with async support
- **pre-commit**: Code quality and formatting
- **Docker Compose**: Local development environment
- **GitHub Actions**: CI/CD pipeline

---

## 🧠 System Intelligence

### Automated Planogram Generation
The system automatically creates planograms from reference shelf images:

1. **SKU Detection**: Run YOLOv11 on reference image
2. **Row Clustering**: Use DBSCAN to group products by Y-coordinate (rows)
3. **Item Ordering**: Sort products left-to-right within each row
4. **Grid Structure**: Generate `(row_idx, item_idx)` cell positions
5. **Persistence**: Store grid with expected SKU per cell

### Real-Time Monitoring Pipeline
```
CCTV Stream → Frame Sampling → Shelf Detection → 
Image Alignment → SKU Detection → Grid Mapping → 
Temporal Consensus → Alert Generation → Dashboard
```

### Temporal Consensus Algorithm
- **Multi-frame validation**: Require N consecutive frames before alerting
- **Track-based filtering**: Use object tracking to reduce false positives
- **State machine**: Per-cell state tracking (OK, OOS, MISPLACED)
- **Confidence thresholds**: Configurable detection confidence levels

---

### Documentation
- **[Project Description](docs/project_description.md)**: Complete system specification
- **[Implementation Tree](docs/project_tree.md)**: Current development status


# Setup `pre-commit`

1. Install `pre-commit` (e.g. `pip install pre-commit`)
2. Run `pre-commit install` to install Git hooks
3. Optionally run `pre-commit run --all-files` to check/format the entire repo
