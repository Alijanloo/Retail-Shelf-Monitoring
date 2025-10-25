The pipeline focuses on reliability for a moving camera: detection → alignment to a reference shelf image (planogram) → compare product placements → decide OOS / misplacement → alert & persist.

# 1 — High-level summary

1. **Create canonical shelf references (planograms)**: for each shelf location, capture reference image(s) showing *correct* placements, run SKU detection, and automatically generate planogram grid structure by clustering detected SKU coordinates (row clustering + item ordering within rows).
2. **Ingest CCTV stream**: extract frames, detect candidate shelf regions, and align each candidate frame to the matching reference shelf using feature matching + homography or embedding similarity.
3. **Run SKU detection**: run your pretrained YOLOv11 on the aligned shelf crop(s) (or on full frames then crop) to get detected SKU boxes + confidences.
4. **Map detections → planogram regions**: use the same clustering/grid-detection pipeline to dynamically extract grid structure from current detections, then match against the reference grid to assign each detected bounding box to an expected planogram cell.
5. **Decision logic**:

   * If an expected cell has no detection (above confidence) for N consecutive frames → **Out-of-Stock (OOS)**.
   * If a detected SKU appears in a different planogram cell than expected (SKU id mismatch) → **Misplacement**.
6. **Temporal filtering & alerts**: use tracking and temporal consensus to avoid false positives. Enqueue alerts to Redis and persist events to PostgreSQL.
7. **Dashboard & APIs**: view results, confirm/cancel alerts (feedback loop).

---

# 2 — Assumptions & definitions

* **Input**: single CCTV moving camera video stream (RTSP/MP4).
* **Model**: YOLOv11 (trained on SKU dataset) that outputs `class_id, confidence, bbox [x1,y1,x2,y2]`.
* **Reference planogram**: for each shelf `S`, a reference image `ref_S.jpg` and automatically generated planogram structure created by:
  1. Running YOLOv11 detection on the reference image
  2. Clustering detected SKU bboxes by Y-coordinate to identify rows
  3. Ordering SKUs within each row by X-coordinate to determine item indices
  4. Storing the resulting grid structure with expected SKU per cell (rows may have different numbers of items)
* **Real-time constraints**: lightweight → prioritize `yolov11-nano/small` variants; use ONNX/TensorRT or PyTorch mobile for edge.
* **Libraries**: OpenCV, PyTorch (or ONNX runtime), scikit-learn (for clustering), FAISS (optional), Redis, PostgreSQL, FastAPI/uvicorn, ByteTrack or SORT for tracking.

---

# 3 — Components (services)

1. **Camera Ingest Service** (Python, OpenCV)

   * Get frames, perform stabilization (optional), keyframe sampling (e.g., 1fps or configurable).
   * Output: frames to local process or message queue.
   * Refer to `docs/stream_example.py`

2. **Shelf Localization & Alignment Service**

   * Detect shelf regions in frame (template matching / object detector or feature-based ROI extraction).
   * Match crop to the correct reference shelf using feature matching (ORB/SIFT + FLANN + RANSAC) → homography to warp image to reference coordinates;
   * Output: aligned shelf crop, matched reference id, homography matrix.

3. **Detection & Tracking Service**

   * Load YOLOv11 model (OpenVino). Run inference on the aligned crop.
   * Run tracker (ByteTrack/SORT) to persist identities across frames for temporal filtering.
   * Output: list of detections per shelf crop with bbox in reference coordinates, tracker IDs.

4. **Grid Detection & Planogram Matcher**

   * **Grid Detection Pipeline** (shared between indexing and inference):
     * Cluster detected SKU bboxes by Y-coordinate (using DBSCAN or K-means) to identify rows
     * Sort rows by average Y-coordinate (top to bottom)
     * Within each row, sort SKUs by X-coordinate (left to right) to assign item indices
     * Generate cell structure: `{row_idx, item_idx, bbox, sku_id}`

   * **During Indexing** (reference creation):
     * Run grid detection on reference image detections
     * Store grid structure with expected SKU per cell position
     * Persist to database as the planogram for shelf `S`

   * **During Inference** (real-time analysis):
     * Run same grid detection on current frame detections
     * Match current grid cells to reference grid cells by position (row_idx, item_idx)
     * Decide per-cell state: `OK`, `OOS`, `MISPLACED`, `UNKNOWN`
     * Use temporal rules (see below) to confirm events.

5. **Alerting & Persistence**

   * Push alert messages to Redis Stream (priority field).
   * Persist raw detection frames, alert events and confirmations into PostgreSQL.

6. **API & Dashboard**

   * FastAPI for queries and push notifications (WebSocket for dashboard).
   * Dashboard UI (Streamlit / React) displays live shelf images, detections, and alert history.

7. **Admin / Feedback UI**

   * Staff confirm/cancel alerts; these labels persist to PostgreSQL for future retraining or improvement.

---

# 4 — Data formats & planogram spec

**Planogram Structure** (generated automatically, stored in DB):

```json
{
  "shelf_id": "S_aisle5_row2",
  "reference_image": "ref_S.jpg",
  "grid": {
    "rows": [
      {
        "row_idx": 0,
        "avg_y": 45,
        "items": [
          {"item_idx": 0, "bbox": [10, 40, 50, 80], "sku_id": "sku_123"},
          {"item_idx": 1, "bbox": [55, 42, 95, 82], "sku_id": "sku_456"},
          ...
        ]
      },
      {
        "row_idx": 1,
        "avg_y": 120,
        "items": [
          {"item_idx": 0, "bbox": [12, 115, 52, 155], "sku_id": "sku_789"},
          ...
        ]
      }
    ]
  },
  "clustering_params": {
    "row_clustering_method": "dbscan",
    "eps": 15,
    "min_samples": 2
  },
  "meta": {"store": "store_01", "aisle": 5, "priority": "high"}
}
```

**Grid Detection Algorithm** (used for both indexing and inference):

1. **Row Clustering**: Group bboxes by Y-coordinate using DBSCAN (eps parameter controls vertical tolerance)
2. **Row Sorting**: Sort clusters by average Y-coordinate (top to bottom)
3. **Item Ordering**: Within each row, sort items by X-coordinate (left to right) to assign item_idx
4. **Cell Assignment**: Each cell identified by `(row_idx, item_idx)`

**Detection record**:

```
{ shelf_id, frame_ts, bbox_ref_coords, class_id, conf, track_id, aligned_frame_image_path }
```

**Alert record (DB)**:

```
{ alert_id, shelf_id, row_idx, item_idx, alert_type (OOS/MISPLACEMENT), detected_sku, expected_sku, ts_first_seen, ts_confirmed, confirmed_by }
```

---

# 5 — Detailed step-by-step pipeline (runtime)

### Step 0 — Initialization & Planogram Creation

**A. Planogram Indexing Phase** (one-time per shelf):
* Capture reference image `ref_S.jpg` for each shelf
* Run YOLOv11 detection on reference image
* Apply grid detection algorithm:
  1. Cluster bboxes by Y-coordinate (DBSCAN/K-means) → identify rows
  2. Sort rows top-to-bottom by avg Y
  3. Within each row, sort items left-to-right by X → assign item_idx
  4. Store grid structure: `{shelf_id, rows: [{row_idx, items: [{item_idx, bbox, sku_id}]}]}`
* Persist planogram to database with clustering parameters

**B. Runtime Initialization**:
* Load all planogram structures into memory (or Redis cache)
* Load YOLOv11 model (optionally ONNX or TensorRT optimized)
* Start Camera Ingest service and Shelf Localization

### Step 1 — Frame acquisition & keyframe selection

* Read frames at `raw_fps`. Option A: process every Nth frame (configurable) or process on scene-change/keyframe detection.
* Optionally run small motion-stabilization or cropping if camera jiggles.

### Step 2 — Shelf localization in frame

* Two methods (choose one or combine):

  1. **Feature-based**: run ORB (fast) on frame and on each reference; compute matches → RANSAC to find homography; pick reference with best inlier ratio. If homography found → warp frame to reference coordinates (aligned crop).
  2. **Embedding-based**: crop possible shelf ROIs (sliding windows or heuristics) and compute embeddings; search against reference embeddings via FAISS; pick best match.
* Output: `shelf_id`, `aligned_image`, `H` (homography).

**Implementation notes**

* For performance, pre-compute features or embeddings for references. Limit feature matching to top-K likely references if many.

### Step 3 — Align & crop

* Apply homography `H` to warp the detected shelf ROI to the **reference coordinate frame** (so planogram coordinates align).
* Resize to model input (maintain aspect ratio; pad if needed).

### Step 4 — SKU detection & tracking

* Run YOLOv11 on the aligned crop.

  * Model outputs: `[class_id, conf, x1,y1,x2,y2]` in reference coords.
  * Recommended thresholds: `conf_thresh = 0.35`, `nms_iou = 0.45` (tweak per dataset).
* Run tracker (ByteTrack/SORT) over time to assign `track_id` and smooth transient mis-detections.
* Optionally compute embeddings for each crop for fallback matching and to identify unknown SKUs.

### Step 5 — Map detections to planogram cells (dynamic grid matching)

* Apply the **same grid detection algorithm** to current frame detections:
  1. Cluster current bboxes by Y-coordinate → identify current rows
  2. Sort rows top-to-bottom
  3. Within each row, sort items left-to-right → assign current item_idx
* Match current grid `(row_idx, item_idx)` to reference grid `(row_idx, item_idx)`
* For each matched cell position, compare:
  * `expected_sku` (from reference planogram at this row_idx, item_idx)
  * `detected_sku` (from current frame at same position)
* Compute cell states:
  * `present = detection exists at (row_idx, item_idx) with conf > threshold and sku_id matches expected`
  * `misplaced = detection exists but sku_id ≠ expected_sku`
  * `empty = no detection at expected (row_idx, item_idx)`

**Handling variable row lengths:**
* If reference row has N items but current row has M items (N ≠ M):
  * Match items by position up to min(N, M)
  * Flag extra items (M > N) as potential misplacements
  * Flag missing items (M < N) as potential OOS

### Step 6 — Temporal consensus & rules

* Use short time-window state machine for each cell position `(shelf_id, row_idx, item_idx)`:

  * Maintain per-cell counters: `consecutive_empty_frames`, `consecutive_misplaced_frames`.
  * Require `N_confirm` frames before raising an alert (e.g., `N_confirm = 3` frames or 2 seconds). This compensates for occlusion or motion blur.
  * If a cell oscillates between present/missing rapidly, raise `uncertain` flag and delay alert.
* Use tracker information: if an item moved across cells (same `track_id` appears in another cell) → label as **movement** (not necessarily misplacement until it remains).

### Step 7 — Produce & enqueue alerts

* When `consecutive_empty_frames >= N_confirm` and item expected in that cell → create `OOS` alert.
* When `consecutive_misplaced_frames >= N_confirm` → create `MISPLACEMENT` alert.
* Enqueue alert to Redis stream with priority metadata and include cropped evidence images and FPS timestamps.
* Persist alert metadata (unconfirmed by default) to PostgreSQL.

### Step 8 — Delivery & feedback

* Alert consumer service dequeues Redis stream messages, sends push to dashboard/mobile.
* Provide UI to **confirm** or **dismiss**. Staff confirmations are written back to DB and trigger analytics/feedback.

---

# 7 — Practical thresholds & tuning advice

* `conf_thresh` (detection): start 0.35 → increase to reduce false positives.
* `nms_iou`: 0.45.
* `N_confirm` (temporal): 2–5 frames (or 1–3 seconds) depending on camera FPS.
* `eps` (DBSCAN row clustering): 10-20 pixels depending on image resolution and shelf spacing. Start with 15.
* `min_samples` (DBSCAN): 1-2 for minimum cluster size. Use 1 for shelves with sparse items.
* `position_tolerance`: when matching current grid to reference, allow ±1 position tolerance for slight alignment variations.
* `movement_threshold`: if same track_id moves across >1 cell within T seconds, treat as probable repositioning rather than missing.

---

# 8 — Robustness & edge cases

* **Moving camera**: rely on robust ORB + RANSAC homography and require a minimal inlier ratio (e.g., > 30). If homography fails, fallback to embedding matching or skip frame.
* **Occlusion / hands in frame**: require `N_confirm` frames to reduce false positives.
* **Partial crops & low resolution**: use tiling — split aligned image into tiles to run detector at higher resolution if necessary.
* **New SKUs**: use image-embedding retrieval (FAISS) to match new visuals against a maintained gallery — flag as `unknown` if low similarity.
* **Lighting changes**: normalize and apply CLAHE if needed before detection.
* **Variable row lengths**: the grid detection algorithm naturally handles rows with different numbers of items by clustering and ordering independently per row.
* **Grid alignment drift**: if current grid structure significantly differs from reference (e.g., major reorganization), flag for manual review and potential planogram re-indexing.
* **Noise in clustering**: outlier detections may form singleton clusters; filter out rows with < min_items threshold or merge nearby clusters.

---

# 9 — Deployment & optimization

* **Model optimization**: export YOLOv11 to ONNX → convert to TensorRT / OpenVINO for lower-latency on edge devices. Quantize to INT8 if accuracy drop acceptable.
* **Batching**: process each shelf crop as a single inference; if many shelves exist, use batch inference across crops for throughput.
* **Edge inference**: Jetson (TensorRT), Intel NCS2 (OpenVINO), or CPU with ONNX runtime with OpenMP.
* **Containerization**: Docker-compose or Kubernetes: services: `camera-ingest`, `shelf-localizer`, `detector`, `alert-worker`, `api`, `postgres`, `redis`.

Example Docker Compose services:

* `camera` (ingest)
* `detector` (YOLOv11 + tracker)
* `analyzer` (planogram + decision logic)
* `alerts` (Redis consumer)
* `api` (dashboard)

---

# 10 — DB schema / minimal tables

* `shelves(shelf_id, ref_image_path, planogram_json, clustering_params, meta)`
  * `planogram_json`: stores the dynamically generated grid structure
  * `clustering_params`: stores DBSCAN/clustering parameters used (eps, min_samples)
* `detections(id, shelf_id, frame_ts, bbox, class_id, conf, track_id, row_idx, item_idx, evidence_path)`
* `alerts(alert_id, shelf_id, row_idx, item_idx, alert_type, first_seen, last_seen, confirmed, confirmed_by)`
* `gallery(sku_id, example_images[], embeddings[])` (for retrieval)

---

# 11 — Metrics & testing

* **Metrics to monitor**:

  * Precision / Recall for OOS and misplacement events (use hand-labelled test sequences).
  * False Positive Rate (important for staff fatigue).
  * Time-to-alert (latency from frame to alert).
  * Inference FPS and queue length.
* **Offline testing**:

  * Create labeled video clips with ground-truth OOS/misplacement events.
  * Run pipeline offline and compute per-event precision/recall and per-frame detection metrics.
* **A/B tests**:

  * Try `N_confirm=3` vs `N_confirm=5` and measure operator-confirmed false positives.

---

# 12 — Logging & observability

* Collect per-frame logs: inference_time, homography_inliers, num_detections, queue_lag.
* Instrument metrics to Prometheus and visualize in Grafana: alerts/sec, TPS, latency, GPU utilization.

---

# 13 — Minimal MVP checklist (deliverables)

1. Tooling & infra: Redis + PostgreSQL + Docker Compose skeleton.
2. **Grid detection module**: implement DBSCAN-based clustering algorithm for row detection and item ordering (shared across indexing and inference).
3. **Reference ingestion pipeline**:
   - UI or script to upload reference shelf image
   - Auto-detect SKUs and generate planogram grid structure
   - Store in database with clustering parameters
4. Camera ingest & shelf-localizer working on sample video.
5. YOLOv11 inference service that returns detections in reference coords.
6. **Dynamic planogram matcher**: applies grid detection to current frame and matches against reference grid by position.
7. Decision logic with `N_confirm` temporal rule and alert enqueue.
8. Simple dashboard showing live shelf crop, detections, grid overlay, and alert history with confirm/dismiss actions.

---

# 14 — Next improvements (after MVP)

* Add segmentation (SAM) or monocular depth for improved OOS detection.
* Use active learning: when staff confirms/dismisses, create a small retraining pipeline to expand your SKU set.
* Multi-camera fusion across time for occlusion resolution.
