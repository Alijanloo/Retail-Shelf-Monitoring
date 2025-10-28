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
│   ├── frame.py           ✅ Frame & AlignedFrame entities
│   └── stream.py          ✅ StreamConfig entity
├── usecases/
│   ├── __init__.py
│   ├── interfaces/
│   │   ├── __init__.py
│   │   ├── repositories.py ✅ Repository ABCs
│   │   ├── tracker_interface.py ✅ Tracker ABC
│   │   └── services.py     ✅ Service ABCs
│   ├── planogram_generation.py ✅ Planogram generation use case
│   ├── shelf_management.py     ✅ Shelf management use case
│   ├── stream_processing.py    ✅ Stream processing orchestration
│   ├── shelf_localization.py   ✅ Shelf localization use case
│   ├── detection_processing.py ✅ Detection processing orchestration
│   ├── cell_state_computation.py ✅ Cell state computation
│   ├── temporal_consensus.py   ✅ Temporal consensus manager
│   └── alert_generation.py     ✅ Alert generation & management
├── adaptors/
│   ├── __init__.py
│   ├── repositories/
│   │   ├── __init__.py
│   │   ├── postgres_shelf_repository.py      ✅ Shelf repository impl
│   │   ├── postgres_planogram_repository.py  ✅ Planogram repository impl
│   │   ├── postgres_detection_repository.py  ✅ Detection repository impl
│   │   └── postgres_alert_repository.py      ✅ Alert repository impl
│   ├── ml/
│   │   ├── __init__.py
│   │   ├── yolo_detector.py   ✅ YOLOv11 detector
│   │   └── sku_mapper.py      ✅ Class ID to SKU mapping
│   ├── tracking/              ✅ Object tracking
│   │   ├── __init__.py
│   │   └── bytetrack.py         ✅ SimpleTracker implementation
│   ├── grid/
│   │   ├── __init__.py
│   │   ├── clustering.py      ✅ DBSCAN/K-means clustering
│   │   └── grid_detector.py   ✅ Grid detection logic
│   ├── video/                 ✅ Video stream processing
│   │   ├── __init__.py
│   │   ├── stream_reader.py   ✅ RTSP/video stream reader
│   │   ├── frame_extractor.py ✅ Frame extraction logic
│   │   └── keyframe_selector.py ✅ Keyframe selection
│   ├── vision/                ✅ Computer vision
│   │   ├── __init__.py
│   │   ├── feature_matcher.py ✅ ORB/SIFT feature matching
│   │   ├── homography.py      ✅ Homography computation
│   │   └── image_aligner.py   ✅ Image alignment/warping
│   ├── preprocessing/         ✅ Image preprocessing
│       ├── __init__.py
│       ├── stabilization.py   ✅ Motion stabilization
│       └── image_processing.py ✅ General preprocessing
│   └── messaging/             ✅ Redis Streams messaging
│       ├── __init__.py
│       ├── redis_stream.py    ✅ Redis Stream client
│       └── alert_publisher.py ✅ Alert publishing
├── frameworks/
│   ├── __init__.py
│   ├── logging_config.py   ✅ Logging framework
│   ├── database.py         ✅ SQLAlchemy models
│   ├── config.py           ✅ Config management
│   ├── exceptions.py       ✅ Exception hierarchy
│   ├── streaming/          ✅ Streaming infrastructure
│   │   ├── __init__.py
│   │   ├── frame_buffer.py    ✅ Thread-safe frame buffer
│   │   └── stream_manager.py  ✅ Stream lifecycle management
│   └── ui/                 ✅ Desktop UI (Phase 6)
│       ├── __init__.py
│       ├── main_window.py      ✅ Main application window
│       ├── threads/
│       │   ├── __init__.py
│       │   ├── capture_thread.py   ✅ Video capture worker
│       │   ├── inference_thread.py ✅ ML inference worker
│       │   └── alert_thread.py     ✅ Alert processing worker
│       ├── widgets/
│       │   ├── __init__.py
│       │   ├── video_widget.py     ✅ Live video display
│       │   └── alert_panel.py      ✅ Alert list and actions
│       └── resources/
│           └── styles.qss          ✅ Qt stylesheet
├── container.py            ✅ DI container
└── __main__.py             ✅ Entry point

tests/
└── unit/
│   ├── entities/
│   │   ├── test_common.py      ✅ Common types tests
│   │   ├── test_shelf.py       ✅ Shelf tests
│   │   ├── test_sku.py         ✅ SKU tests
│   │   ├── test_planogram.py   ✅ Planogram tests
│   │   ├── test_detection.py   ✅ Detection tests
│   │   └── test_alert.py       ✅ Alert tests
│   ├── usecases/
│   │   └── test_temporal_consensus.py ✅ Temporal consensus tests
│   ├── adaptors/
│   │   ├── __init__.py
│   │   ├── test_clustering.py      ✅ Clustering algorithm tests
│   │   ├── test_grid_detector.py   ✅ Grid detector tests
│   │   ├── test_feature_matcher.py
│   │   └── test_homography.py
│   └── frameworks/
│       └── test_streaming.py
