# Retail Shelf Monitoring System

A real-time retail shelf monitoring solution powered by computer vision and machine learning. The system leverages existing CCTV infrastructure to automatically detect **out-of-stock (OOS)** situations and **product misplacements**, enabling data-driven shelf management and faster restocking decisions.

---

## 🚀 Key Features

* **Real-time Shelf Analysis** using YOLOv12-M for product detection
* **SKU Recognition** via MobileNetV3 embeddings and FAISS similarity search
* **Planogram Compliance** with automated grid generation and SKU matching
* **Tracking with SORT** for temporal consistency across frames
* **Multi-threaded Architecture** for high throughput and low-latency inference
* **Cross-platform Optimization** with ONNX, OpenVINO, and TensorRT
* **Interactive Desktop App** (PySide6) for visualization and alert management

---

## ⚙️ Processing Pipeline

```
CCTV Stream → Keyframe Selection → Shelf Detection →
Image Alignment → YOLOv12 Detection → SKU Recognition →
Grid Mapping → Temporal Consensus → Alert Generation →
Desktop UI Display → Staff Confirmation
```

**Keyframe-based processing** minimizes redundant computation, while **tracking (SORT + Kalman Filter)** ensures temporal stability across frames.

---

## 🧩 Core Components

* **Detection Model:** YOLOv12-M fine-tuned on SKU-110k (mAP50–95 = 0.56)
* **SKU Recognition:** 93% Top-1 accuracy on custom embeddings
* **Grid Generation:** Automated DBSCAN-based clustering of shelf items
* **Alert Logic:** Temporal consensus filtering to reduce false positives
* **App Interface:** Live feed overlay, planogram visualization, alert confirmation

---

## 🧰 Technologies

* **Python**, **PyTorch**, **OpenVINO**, **TensorRT**, **OnnxRuntime**, **Pytorch-TensorRT**, **FAISS**, **PySide6**, **PostgreSQL**, **Redis**
* **Tracking:** SORT + Hungarian Algorithm
* **Feature Matching:** ORB/SIFT with RANSAC-based homography

---

### Documentation
- **[Technical Report](docs/technical_report.md)**: Comprehensive system implementation details
- **[Project Tree](docs/project_tree.md)**: Current development status and architecture overview

---

## 🚀 Quick Start

### Installation
```bash
# Clone repository
git clone https://github.com/Alijanloo/Retail-Shelf-Monitoring.git
cd Retail-Shelf-Monitoring

uv pip install -e .

# Configure settings
cp config.example.yaml config.yaml
# Edit config.yaml with your preference

# Start PostgreSQL and Redis services
docker-compose up -d

# Launch desktop application
uv run python launch_ui.py
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
