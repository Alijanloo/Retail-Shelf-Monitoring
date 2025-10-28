# SKU Detection System

## Overview

The SKU detection system uses a two-stage approach:

1. **YOLO Detector**: Detects general object classes (e.g., "bottle", "can", "box") in shelf images
2. **SKU Detector**: Identifies specific SKU products from cropped detections using MobileNet embeddings and FAISS similarity search

## Architecture

### Stage 1: Object Detection (YOLO)
- Model: YOLOv11 (OpenVINO format)
- Purpose: Detect and localize products on shelves
- Output: Bounding boxes with general class labels

### Stage 2: SKU Identification
- Model: MobileNet (OpenVINO format)
- Technology: FAISS vector similarity search
- Purpose: Identify specific SKU from cropped product images
- Output: Precise SKU ID based on visual embeddings

## Setup

### 1. Prepare SKU Reference Dataset

Organize your SKU reference images in the following structure:

```
data/sku_references/
    0/
        image1.jpg
        image2.jpg
        image3.jpg
    1/
        image1.jpg
        image2.jpg
    2/
        image1.jpg
        ...
```

- Each subdirectory represents a unique SKU class (0, 1, 2, ...)
- The directory name MUST be an integer
- Multiple reference images per SKU improve accuracy
- Images should be clear, well-lit product photos

### 2. Export MobileNet to OpenVINO

If you have a PyTorch or TensorFlow MobileNet model, export it to OpenVINO format:

```bash
# Using OpenVINO Model Optimizer
mo --input_model mobilenet_v2.onnx \
   --output_dir data/ \
   --model_name mobilenet_sku
```

This creates `mobilenet_sku.xml` and `mobilenet_sku.bin`.

### 3. Build FAISS Index

```bash
uv run python data/build_sku_index.py \
    --data-dir data/sku_references \
    --model data/mobilenet_sku.xml \
    --output data/sku_index.faiss \
    --device CPU
```

This will:
- Process all images in `data/sku_references/`
- Extract embeddings using MobileNet
- Build FAISS index for fast similarity search
- Save index to `data/sku_index.faiss`
- Save metadata to `data/sku_index.labels.npy` and `data/sku_index.paths.npy`

### 4. Configure Application

Update `config.yaml`:

```yaml
sku_detection:
  model_path: "data/mobilenet_sku.xml"  # Path to MobileNet model
  index_path: "data/sku_index.faiss"    # Path to FAISS index
  device: "CPU"                          # OpenVINO device
  top_k: 1                               # Number of similar SKUs to retrieve
```

## How It Works

### Indexing Phase (One-time)

1. Load MobileNet model via OpenVINO
2. For each reference image:
   - Preprocess image (resize, normalize)
   - Extract embedding vector
   - Store embedding with SKU label
3. Build FAISS index from all embeddings
4. Save index to disk

### Inference Phase (Runtime)

1. YOLO detects objects and returns bounding boxes
2. For each detection:
   - Crop image using bounding box
   - Resize to 224×224 pixels
   - Convert BGR to RGB
   - Normalize using ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
   - Extract embedding using MobileNet via OpenVINO
   - L2 normalize the embedding vector
   - Search FAISS index for most similar embeddings
   - Return SKU ID of closest match
3. Assign SKU ID to detection

## API Usage

### Building Index Programmatically

```python
from retail_shelf_monitoring.adaptors.ml.sku_detector import SKUDetector

detector = SKUDetector(
    model_path="data/mobilenet_sku.xml",
    index_path=None,  # Will be saved later
    device="CPU"
)

# Build index from reference images
num_images, num_classes = detector.build_index(
    data_dir="data/sku_references",
    index_output_path="data/sku_index.faiss"
)

print(f"Indexed {num_images} images across {num_classes} SKU classes")
```

### Identifying SKUs at Runtime

```python
import cv2
from retail_shelf_monitoring.adaptors.ml.sku_detector import SKUDetector

# Load pre-built index
detector = SKUDetector(
    model_path="data/mobilenet_sku.xml",
    index_path="data/sku_index.faiss",
    device="CPU"
)

# Identify SKU from cropped image
cropped_image = cv2.imread("cropped_product.jpg")
sku_id = detector.get_sku_id(cropped_image)
print(f"Detected SKU: {sku_id}")  # e.g., "sku_42"

# Get detailed results
sku_label, confidence, top_k_results = detector.identify_sku(cropped_image)
print(f"SKU: {sku_label}, Confidence: {confidence:.3f}")
print(f"Top-K results: {top_k_results}")
```

## Configuration Options

### `sku_detection.model_path`
Path to MobileNet OpenVINO model (.xml file)

### `sku_detection.index_path`
Path to FAISS index file (created by build script)

### `sku_detection.device`
OpenVINO target device: CPU, GPU, MYRIAD, etc.

### `sku_detection.top_k`
Number of top similar SKUs to retrieve (default: 1)
- `top_k=1`: Return only best match
- `top_k=5`: Return 5 most similar SKUs (useful for debugging)

## Troubleshooting

### No images found
Ensure directory structure follows `data_dir/class_id/*.jpg` pattern

### Model loading fails
Verify OpenVINO model path and check model format (.xml + .bin files)

### Poor accuracy
- Add more reference images per SKU
- Ensure reference images are high quality
- Check lighting consistency between reference and runtime images
- Consider increasing `top_k` to examine alternative matches

### FAISS index file not found
Run `build_sku_index.py` script before starting the application

## Performance

- **Indexing**: ~10-50ms per image (depends on model size)
- **Runtime**: ~5-15ms per detection (embedding + search)
- **Memory**: Proportional to number of reference images (typically <100MB)

## Dependencies

- `faiss-cpu>=1.7.4` - Vector similarity search
- `openvino>=2023.0.0` - Model inference
- `opencv-python>=4.8.0` - Image processing
- `numpy>=1.24.0` - Numerical operations
