# Retail Shelf Monitoring - Project Structure

## Overview
This document outlines the current implementation status and structure of the Retail Shelf Monitoring system following Clean Architecture principles.

### Directory Structure

```
retail_shelf_monitoring/
├── entities/
│   ├── __init__.py
│   ├── common.py
│   ├── sku.py
│   ├── planogram.py
│   ├── detection.py
│   ├── alert.py
│   ├── frame.py
├── usecases/
│   ├── __init__.py
│   ├── interfaces/
│   │   ├── __init__.py
│   │   ├── repositories.py
│   │   └── tracker_interface.py
│   ├── planogram_generation.py
│   ├── stream_processing.py
│   ├── detection_processing.py
│   ├── cell_state_computation.py
│   ├── temporal_consensus.py
│   └── alert_generation.py
├── adaptors/
│   ├── __init__.py
│   ├── repositories/
│   │   ├── __init__.py
│   │   ├── postgres_planogram_repository.py
│   │   └── postgres_alert_repository.py
│   ├── ml/
│   │   ├── __init__.py
│   │   ├── yolo_detector.py
│   │   └── sku_detector.py    MobileNet + FAISS SKU identification
│   ├── tracking/
│   │   ├── __init__.py
│   │   └── sort.py
│   ├── grid/
│   │   ├── __init__.py
│   │   ├── clustering.py
│   │   └── grid_detector.py
│   ├── vision/
│   │   ├── __init__.py
│   │   ├── feature_matcher.py
│   │   ├── homography.py
│   │   └── shelf_aligner.py
│   └── messaging/
│       ├── __init__.py
│       ├── redis_stream.py      Redis Stream client
│       └── alert_publisher.py   Alert publishing
├── frameworks/
│   ├── __init__.py
│   ├── logging_config.py
│   ├── database.py           SQLAlchemy models
│   ├── config.py
│   ├── exceptions.py
│   └── ui/
│       ├── __init__.py
│       ├── main_window.py
│       ├── threads/
│       │   ├── __init__.py
│       │   ├── capture_thread.py
│       │   ├── inference_thread.
│       │   └── alert_thread.py
│       ├── widgets/
│       │   ├── __init__.py
│       │   ├── video_widget.py
│       │   └── alert_panel.py
│       └── resources/
│           └── styles.qss
├── container.py
└── __main__.py

tests/
└── unit/
    ├── entities/
    │   ├── test_common.py
    │   ├── test_sku.py
    │   ├── test_planogram.py
    │   ├── test_detection.py
    │   └── test_alert.py
    ├── usecases/
    │   └── test_temporal_consensus.py
    └── adaptors/
        ├── __init__.py
        ├── test_clustering.py
        ├── test_grid_detector.py
        ├── test_feature_matcher.py
        └── test_homography.py
