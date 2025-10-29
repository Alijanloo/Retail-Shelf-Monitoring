# UI and Use Case Integration Architecture

## Overview

The Retail Shelf Monitoring system follows Clean Architecture principles with clear separation between the UI layer (Frameworks & Drivers) and the Use Case layer (Application Business Rules). This document describes how the UI properly integrates with use cases.

## Architecture Layers

### 1. Use Cases (Application Business Rules)

Use cases orchestrate the business logic and coordinate between entities and adapters. They are independent of UI frameworks.

#### Stream Processing Use Cases

**`StreamProcessingUseCase`**
- **Purpose**: Full pipeline orchestration for frame processing with alignment and alerting
- **Dependencies**:
  - ShelfAligner
  - DetectionProcessingUseCase
  - PlanogramRepository
  - CellStateComputation
  - TemporalConsensusManager
  - AlertGenerationUseCase
- **Methods**:
  - `async process_frame(frame, frame_id, timestamp, expected_shelf_id)` → dict
    - Aligns frame to reference shelf
    - Runs detection pipeline
    - Computes cell states against planogram
    - Updates temporal consensus
    - Generates/clears alerts
    - Returns detections, alerts, and cell states

**`DetectionProcessingUseCase`**
- **Purpose**: Detection and tracking coordination
- **Dependencies**:
  - YOLOv11Detector
  - Tracker
  - SKUDetector
- **Methods**:
  - `async process_aligned_frame(aligned_image, frame_metadata, shelf_id)` → List[Detection]

**`AlertManagementUseCase`**
- **Purpose**: Alert query and management operations
- **Dependencies**:
  - AlertRepository
  - AlertPublisher
- **Methods**:
  - `async get_active_alerts(shelf_id=None)` → List[Alert]
  - `async get_alert_by_id(alert_id)` → Optional[Alert]
  - `async confirm_alert(alert_id, confirmed_by)` → Alert
  - `async dismiss_alert(alert_id)` → Alert

**`AlertGenerationUseCase`**
- **Purpose**: Alert creation and lifecycle management
- **Dependencies**:
  - AlertRepository
  - PlanogramRepository
  - AlertPublisher
- **Methods**:
  - `async generate_alert(alert_data, evidence_paths)` → Alert
  - `async clear_cell_alerts(shelf_id, row_idx, item_idx)`

### 2. UI Layer (Frameworks & Drivers)

The UI layer uses Qt/PySide6 and integrates with use cases through dependency injection.

#### Thread Architecture

**`CaptureThread`** (QThread)
- **Purpose**: Frame capture from video source
- **Dependencies**: None (pure OpenCV)
- **Signals**:
  - `frame_signal(np.ndarray, str)` - emits captured frame and timestamp
  - `fps_signal(float)` - emits current FPS
  - `error_signal(str)` - emits error messages

**`InferenceThread`** (QThread)
- **Purpose**: Run detection inference using use case
- **Dependencies**: `StreamProcessingUseCase` (injected)
- **Key Change**: Now delegates to use case instead of directly calling detector
- **Signals**:
  - `result_signal(List[Detection], str)` - emits detections and shelf_id
  - `latency_signal(float)` - emits processing latency
  - `error_signal(str)` - emits error messages

**Before (Direct Access - WRONG)**:
```python
class InferenceThread(QThread):
    def __init__(self, detection_use_case):
        self.detection_use_case = detection_use_case

    def run(self):
        # WRONG: Directly accessing internal components
        raw_detections = self.detection_use_case.detector.detect(frame)
        tracked = self.detection_use_case.tracker.update(raw_detections)
        # ... manual SKU detection
```

**After (Use Case Delegation - CORRECT)**:
```python
class InferenceThread(QThread):
    def __init__(self, stream_processing_use_case):
        self.stream_processing_use_case = stream_processing_use_case

    def run(self):
        # CORRECT: Delegate to use case
        detections = asyncio.run(
            self.stream_processing_use_case.process_frame_simple(
                frame=frame,
                frame_id=frame_id,
                shelf_id=self.shelf_id,
                timestamp=datetime.utcnow(),
            )
        )
```

**`AlertThread`** (QThread)
- **Purpose**: Poll for new alerts
- **Dependencies**: `AlertManagementUseCase` (injected)
- **Key Change**: Uses dedicated event loop for async operations
- **Signals**:
  - `new_alert_signal(Alert)` - emits new alerts
  - `alert_update_signal(Alert)` - emits alert updates
  - `error_signal(str)` - emits error messages

**Before (Async in wrong context - PROBLEMATIC)**:
```python
def run(self):
    while not self._stop_event.is_set():
        # PROBLEMATIC: Creates new event loop each iteration
        alerts = asyncio.run(self.alert_use_case.get_active_alerts())
```

**After (Proper async handling - CORRECT)**:
```python
def run(self):
    self._loop = asyncio.new_event_loop()
    asyncio.set_event_loop(self._loop)

    while not self._stop_event.is_set():
        # CORRECT: Reuse event loop
        alerts = self._loop.run_until_complete(
            self.alert_use_case.get_active_alerts()
        )

    self._loop.close()
```

#### Main Window

**`MainWindow`**
- **Dependencies**: `ApplicationContainer` (for dependency injection)
- **Key Responsibilities**:
  - Initialize UI components
  - Create and wire threads
  - Delegate alert operations to use cases
  - Display results

**Before (Direct use case instantiation - WRONG)**:
```python
def _on_start(self):
    detection_use_case = self.container.detection_processing_usecase()
    self.inference_thread = InferenceThread(detection_use_case, conf_threshold=0.35)
```

**After (Proper use case injection - CORRECT)**:
```python
def _on_start(self):
    stream_processing_use_case = self.container.simple_stream_processing_usecase()
    self.inference_thread = InferenceThread(
        stream_processing_use_case=stream_processing_use_case,
        shelf_id=self.current_shelf_id,
    )
```

**Alert Handling (CORRECT)**:
```python
@Slot(str, str)
def _on_alert_confirmed(self, alert_id, staff_id):
    async def confirm():
        alert_use_case = self.container.alert_management_usecase()
        await alert_use_case.confirm_alert(alert_id, staff_id)

    asyncio.run(confirm())
    self.alert_panel.remove_alert(alert_id)
```

## Dependency Injection Container

The `ApplicationContainer` wires all dependencies:

```python
class ApplicationContainer(containers.DeclarativeContainer):
    # ... adapters and repositories ...

    detection_processing_usecase = providers.Factory(
        DetectionProcessingUseCase,
        detector=yolo_detector,
        tracker=tracker,
        sku_detector=sku_detector,
    )

    stream_processing_usecase = providers.Factory(
        StreamProcessingUseCase,
        shelf_aligner=shelf_aligner,
        detection_processing=detection_processing_usecase,
        planogram_repository=planogram_repository,
        cell_state_computation=cell_state_computation,
        temporal_consensus=temporal_consensus_manager,
        alert_generation=alert_generation_usecase,
    )

    alert_management_usecase = providers.Factory(
        AlertManagementUseCase,
        alert_repository=alert_repository,
        alert_publisher=alert_publisher,
    )
```

## Data Flow

### Simple Detection Flow (Current UI Implementation)

```
Frame Capture → Frame Queue → Inference Thread → Use Case → Detections → UI Display
                                    ↓
                            StreamProcessingUseCase
                                    ↓
                          DetectionProcessingUseCase
                                    ↓
                    (Detector → Tracker → SKU Detector)
```

### Full Pipeline Flow (For Future Extension)

```
Frame → StreamProcessingUseCase
           ↓
        Shelf Aligner (align to reference)
           ↓
        Detection Processing (detect + track + SKU)
           ↓
        Cell State Computation (compare to planogram)
           ↓
        Temporal Consensus (filter false positives)
           ↓
        Alert Generation (create/clear alerts)
           ↓
        Return {detections, alerts, cell_states}
```

### Alert Flow

```
Alert Thread (poll) → AlertManagementUseCase.get_active_alerts()
                              ↓
                       AlertRepository
                              ↓
                       New Alerts → UI Panel
                              ↓
User Action (confirm/dismiss) → MainWindow slot
                              ↓
                    AlertManagementUseCase.confirm_alert()
                              ↓
                       AlertRepository + AlertPublisher
```

## Key Principles Applied

### 1. Dependency Rule
- Dependencies point inward: UI → Use Cases → Entities
- Use cases don't know about Qt, threads, or UI specifics
- UI gets use cases through dependency injection

### 2. Single Responsibility
- **Threads**: Handle Qt event loop and signal emission
- **Use Cases**: Orchestrate business logic
- **Adapters**: Interface with external systems (ML models, DB, etc.)

### 3. Interface Segregation
- UI only depends on use case interfaces
- Use cases depend on repository/detector interfaces (not implementations)
- Easy to swap implementations without changing UI

### 4. Inversion of Control
- Container manages all dependencies
- UI receives fully-configured use cases
- No manual instantiation of dependencies

## Benefits of This Architecture

1. **Testability**: Use cases can be tested without UI
2. **Flexibility**: Can swap UI framework without changing use cases
3. **Maintainability**: Clear boundaries and responsibilities
4. **Reusability**: Use cases can be used by different UIs (desktop, web, CLI)
5. **Async Handling**: Proper event loop management in threads

## Migration Checklist

When adding new UI features:

- [ ] Create/identify the appropriate use case
- [ ] Add use case to dependency injection container
- [ ] Inject use case into UI component (thread/window)
- [ ] UI calls use case methods (don't access internal components)
- [ ] Handle async properly (dedicated event loop in threads)
- [ ] Emit signals with domain entities (not raw data)
- [ ] Keep UI code minimal (delegate to use cases)

## Anti-Patterns to Avoid

❌ **Direct adapter access from UI**:
```python
detector = use_case.detector  # WRONG
detector.detect(frame)
```

❌ **Business logic in UI**:
```python
# WRONG: SKU matching logic in thread
for det in detections:
    cropped = frame[y1:y2, x1:x2]
    sku_id = sku_detector.get_sku_id(cropped)
```

❌ **Creating new event loop per call**:
```python
# WRONG: Creates overhead
asyncio.run(use_case.method())  # in a loop
```

✅ **Correct pattern**:
```python
# Inject use case
self.use_case = use_case

# Call use case method
result = await self.use_case.process(data)

# Or in thread with dedicated loop
result = self._loop.run_until_complete(self.use_case.process(data))
```
