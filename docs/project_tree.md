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
│   └── shelf_localization.py   ✅ Shelf localization use case (Phase 3)
├── adaptors/
│   ├── __init__.py
│   ├── repositories/
│   │   ├── __init__.py
│   │   ├── postgres_shelf_repository.py      ✅ Shelf repository impl
│   │   └── postgres_planogram_repository.py  ✅ Planogram repository impl
│   ├── ml/
│   │   ├── __init__.py
│   │   ├── model_loader.py    ✅ OpenVINO model loader
│   │   └── yolo_detector.py   ✅ YOLOv11 detector
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
└── integration/            ⏳ TODO: Phase 2 & 3 integration tests
```
