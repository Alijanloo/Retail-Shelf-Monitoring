# Retail Shelf Monitoring System - Technical Report

## Executive Summary

This technical report presents the comprehensive implementation of a real-time retail shelf monitoring system utilizing computer vision and machine learning technologies. The system leverages existing CCTV infrastructure to automatically detect out-of-stock (OOS) situations and product misplacements through advanced video analysis techniques.

## 1. System Overview

### 1.1 Project Objective
The system aims to monitor retail shelves using existing CCTV cameras, providing real-time analysis and alerts for:
- Out-of-stock (OOS) detection
- Product misplacement identification

### 1.2 Core Architecture
The system follows Clean Architecture principles with clear separation of concerns:
- **Entities**: Core business objects (SKU, Planogram, Detection, Alert)
- **Use Cases**: Business logic implementation
- **Adaptors**: External service interfaces (ML models, databases, tracking)
- **Frameworks**: Infrastructure and UI components

## 2. Technical Implementation

### 2.1 Machine Learning Pipeline

#### 2.1.1 Object Detection
- **Model**: YOLOv12-Medium fine-tuned on SKU-110k dataset
- **Performance**: mAP50-95 of 0.56
- **Optimization**: ONNX export for OpenVINO (CPU) and TensorRT (GPU) inference
- **Speed Improvement**: Up to 3x faster inference through hardware acceleration

#### 2.1.2 SKU Identification
- **Feature Extraction**: MobileNetV3-based embedding model
- **Similarity Search**: FAISS vector database for nearest neighbor retrieval
- **Accuracy**: 93% SKU identification accuracy Top-1
- **Reference Dataset**: Based on [shelf_management](https://github.com/rokopi-byte/shelf_management) repository

### 2.2 Video Processing Pipeline

#### 2.2.1 Frame Processing Strategy
```
Input Video → Keyframe Detection → Object Detection → SKU Identification → Grid Analysis → Alert Generation
```

**Keyframe Selection**:
- Intensity-based change detection to identify significant frames
- Reduces computational load by processing only meaningful frames
- Non-keyframes utilize tracking algorithms (SORT) for continuity

#### 2.2.2 Shelf Alignment and Localization
- **Feature Matching**: ORB/SIFT descriptors for robust feature extraction
- **Geometric Estimation**: RANSAC-based homography estimation
- **Reference Matching**: Template matching against stored shelf references
- **Coordinate Transformation**: Warping detected frames to reference coordinate system

### 2.3 Grid Detection and Planogram Generation

#### 2.3.1 Automated Grid Creation
**Indexing Phase** (One-time setup per shelf):
1. Capture reference shelf images with correct product placements
2. Run YOLOv12 detection on reference images
3. Apply clustering algorithm (DBSCAN) for row identification:
   - Cluster bounding boxes by Y-coordinate
   - Sort rows top-to-bottom by average Y-coordinate
   - Order items within rows left-to-right by X-coordinate
4. Generate grid matrix structure: `{shelf_id, rows: [{row_idx, items: [{item_idx, bbox, sku_id}]}]}`
5. Store planogram in PostgreSQL database

#### 2.3.2 Real-time Grid Matching
**Runtime Processing**:
1. Apply same clustering algorithm to current frame detections
2. Match current grid structure to reference planogram
3. Compare detected SKUs with expected SKUs per cell position
4. Generate cell states: `OK`, `OOS`, `MISPLACED`, `UNKNOWN`

### 2.4 Performance Optimizations

#### 2.4.1 Inference Acceleration
- **ONNX Export**: Model serialization for cross-platform compatibility
- **OpenVINO**: Intel CPU optimization
- **TensorRT**: NVIDIA GPU acceleration for real-time processing
- **PyTorch-TensorRT**: Hybrid optimization framework

#### 2.4.2 Multi-threading Architecture
```
Main Thread
├── Capture Thread (Video Input)
├── Inference Thread (Shelf Aligner, SKU Detection/Recognition, Tracker)
├── Alert Analysis Thread (Generate Grid, Analyse compliance, Generate Alerts into Redis)
└── Alert Thread (Reads Alerts from Redis and updated Notification System)
```

**Threading Benefits**:
- Decoupled detection from grid analysis and alert generation
- Faster video rendering with real-time overlay updates
- Improved system responsiveness and throughput

#### 2.4.3 Tracking Integration
- **SORT Algorithm**: Simple Online and Realtime Tracking
- **Kalman Filter**: State prediction for object continuity
- **IoU Matching**: Hungarian algorithm for detection-track association
- **Temporal Consistency**: Reduces false positives through frame-to-frame tracking

### 2.5 Temporal Consensus and Alert Logic

#### 2.5.1 State Management
- **Cell State Tracking**: Maintain status per planogram cell position
- **Temporal Filtering**: Require 3+ consecutive detections before alert generation
- **False Positive Reduction**: Prevent alerts from temporary occlusions or motion blur

## 3. System Components

### 3.1 Desktop Application (PySide6)
- **Real-time Video Display**: Live stream with detection overlays
- **Interactive Alert Panel**: Staff confirmation/dismissal interface
- **Planogram Visualization**: Grid overlay with cell status indicators


## 4. Technical Challenges and Solutions

### 4.1 Performance Optimization
**Challenge**: Real-time processing requirements with limited computational resources

**Solutions**:
- Hardware-accelerated inference (OpenVINO, TensorRT)
- Keyframe-based processing to reduce computational load
- Multi-threaded architecture for parallel processing
- Efficient tracking algorithms (SORT) for non-detection frames

## 5. Conclusion

The Retail Shelf Monitoring System successfully demonstrates the integration of modern computer vision and machine learning technologies for practical retail automation. The system achieves real-time performance through strategic optimization techniques while maintaining high accuracy through robust algorithmic approaches.
