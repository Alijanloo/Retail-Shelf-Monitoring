# Retail Shelf Monitoring - Project Structure

## Overview
This document outlines the current implementation status and structure of the Retail Shelf Monitoring system following Clean Architecture principles.

### Directory Structure

```
retail_shelf_monitoring/
├── entities/
│   ├── __init__.py
│   ├── common.py          ✅ Enums, BoundingBox
│   ├── sku.py             ✅ SKU entity
│   ├── shelf.py           ✅ Shelf entity
│   ├── planogram.py       ✅ Planogram with grid
│   ├── detection.py       ✅ Detection entity
│   ├── alert.py           ✅ Alert entity
│   ├── frame.py           ✅ Frame & AlignedFrame entities (Phase 3)
│   └── stream.py          ✅ StreamConfig entity (Phase 3)
├── usecases/
│   ├── __init__.py
│   ├── interfaces/
│   │   ├── __init__.py
│   │   ├── repositories.py ✅ Repository ABCs
│   │   └── services.py     ✅ Service ABCs
│   ├── planogram_generation.py ✅ Planogram generation use case
│   ├── shelf_management.py     ✅ Shelf management use case
│   ├── stream_processing.py    ✅ Stream processing orchestration (Phase 3)
│   ├── shelf_localization.py   ✅ Shelf localization use case (Phase 3)
│   ├── detection_processing.py ✅ Detection processing orchestration (Phase 4)
│   └── cell_state_computation.py ✅ Cell state computation (Phase 4)
├── adaptors/
│   ├── __init__.py
│   ├── repositories/
│   │   ├── __init__.py
│   │   ├── postgres_shelf_repository.py      ✅ Shelf repository impl
│   │   ├── postgres_planogram_repository.py  ✅ Planogram repository impl
│   │   └── postgres_detection_repository.py  ✅ Detection repository impl (Phase 4)
│   ├── ml/
│   │   ├── __init__.py
│   │   ├── model_loader.py    ✅ OpenVINO model loader
│   │   ├── yolo_detector.py   ✅ YOLOv11 detector
│   │   └── sku_mapper.py      ✅ Class ID to SKU mapping (Phase 4)
│   ├── tracking/              ✅ Phase 4: Object tracking
│   │   ├── __init__.py
│   │   ├── tracker_interface.py ✅ Tracker ABC
│   │   └── bytetrack.py         ✅ SimpleTracker implementation
│   ├── grid/
│   │   ├── __init__.py
│   │   ├── clustering.py      ✅ DBSCAN/K-means clustering
│   │   └── grid_detector.py   ✅ Grid detection logic
│   ├── video/                 ✅ Phase 3: Video stream processing
│   │   ├── __init__.py
│   │   ├── stream_reader.py   ✅ RTSP/video stream reader
│   │   ├── frame_extractor.py ✅ Frame extraction logic
│   │   └── keyframe_selector.py ✅ Keyframe selection
│   ├── vision/                ✅ Phase 3: Computer vision
│   │   ├── __init__.py
│   │   ├── feature_matcher.py ✅ ORB/SIFT feature matching
│   │   ├── homography.py      ✅ Homography computation
│   │   └── image_aligner.py   ✅ Image alignment/warping
│   └── preprocessing/         ✅ Phase 3: Image preprocessing
│       ├── __init__.py
│       ├── stabilization.py   ✅ Motion stabilization
│       └── image_processing.py ✅ General preprocessing
├── frameworks/
│   ├── __init__.py
│   ├── logging_config.py   ✅ Logging framework
│   ├── database.py         ✅ SQLAlchemy models
│   ├── config.py           ✅ Config management (updated for Phase 3)
│   ├── exceptions.py       ✅ Exception hierarchy
│   └── streaming/          ✅ Phase 3: Streaming infrastructure
│       ├── __init__.py
│       ├── frame_buffer.py    ✅ Thread-safe frame buffer
│       └── stream_manager.py  ✅ Stream lifecycle management
├── container.py            ✅ DI container (updated for Phase 3)
└── __main__.py             ✅ Entry point

tests/
├── unit/
│   ├── entities/
│   │   ├── test_common.py      ✅ Common types tests
│   │   ├── test_shelf.py       ✅ Shelf tests
│   │   ├── test_sku.py         ✅ SKU tests
│   │   ├── test_planogram.py   ✅ Planogram tests
│   │   ├── test_detection.py   ✅ Detection tests
│   │   └── test_alert.py       ✅ Alert tests
│   └── adaptors/
│       ├── __init__.py
│       ├── test_clustering.py      ✅ Clustering algorithm tests
│       ├── test_grid_detector.py   ✅ Grid detector tests
│       └── phase3/                 ⏳ TODO: Phase 3 tests
│           ├── test_feature_matcher.py
│           ├── test_homography.py
│           └── test_stream_processing.py
└── integration/            ⏳ TODO: Phase 2-4 integration tests

## Phase 4 Completion Status

### ✅ Completed Components

1. **Tracking Infrastructure**
   - `adaptors/tracking/tracker_interface.py` - Abstract tracker interface
   - `adaptors/tracking/bytetrack.py` - SimpleTracker with IoU-based matching
   - Track management with configurable thresholds

2. **SKU Mapping**
   - `adaptors/ml/sku_mapper.py` - Class ID to SKU identifier mapping
   - JSON configuration support
   - Bidirectional mapping (class↔SKU)

3. **Detection Processing**
   - `usecases/detection_processing.py` - Detection orchestration use case
   - Integration of detector + tracker + SKU mapper
   - Batch persistence to database

4. **Cell State Computation**
   - `usecases/cell_state_computation.py` - Grid matching logic
   - Cell state determination (OK/EMPTY/MISPLACED)
   - Summary statistics generation

5. **Detection Repository**
   - `adaptors/repositories/postgres_detection_repository.py` - PostgreSQL persistence
   - Batch operations for performance
   - Cell-based and temporal queries

6. **Configuration & DI**
   - Updated `config.yaml` with tracking parameters
   - Updated `container.py` with Phase 4 providers
   - Example SKU mapping file

7. **Demo & Documentation**
   - `examples/phase4_demo.py` - Complete Phase 4 demonstration
   - `data/sku_mapping.json` - Example SKU mapping
   - Updated project tree documentation

### 🎯 Key Features

- **Object Tracking**: Maintain detection consistency across frames
- **Grid Matching**: Match current detections against reference planogram
- **Cell States**: Determine product availability and placement status
- **Batch Operations**: Optimized database persistence
- **SKU Mapping**: Flexible class-to-product mapping
- **Temporal Queries**: Retrieve detection history by cell

### 📊 Architecture

```
Aligned Frame → YOLO Detection → Tracking → SKU Mapping →
Detection Entities → Grid Matching → Cell States →
Database Persistence
```
