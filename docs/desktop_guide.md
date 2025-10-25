# Desktop UI - User Guide

## Overview
The Retail Shelf Monitoring desktop application is built with PySide6 (Qt6) and provides a native, responsive interface for real-time shelf monitoring.

## Features

### Main Window Components

#### 1. Video Display Panel (Left)
- **Live Video Feed**: Real-time display from selected camera/video source
- **Detection Overlays**: Bounding boxes with SKU labels and confidence scores
- **Planogram Grid**: Optional overlay showing expected product positions
- **Auto-scaling**: Maintains aspect ratio while filling available space

#### 2. Control Panel (Bottom Left)
- **Source Selector**: Choose between cameras or test videos
- **Confidence Threshold**: Adjust detection sensitivity (10-95%)
- **FPS Limit**: Control frame processing rate (1-60 fps)
- **Display Options**: Toggle detection boxes and grid overlay
- **Start/Stop Controls**: Begin or pause monitoring

#### 3. Alert Panel (Right)
- **Active Alerts List**: Real-time alert feed with priority indicators
- **Filter Controls**: Filter by alert type and priority
- **Action Buttons**:
  - ✓ Confirm: Mark alert as verified
  - ✗ Dismiss: Close false positive
  - 📋 Details: View full alert information
- **Statistics**: Total alerts and breakdown by priority

#### 4. Status Bar (Bottom)
- **System Status**: Current operation state
- **FPS Meter**: Live frame capture rate
- **Latency Meter**: ML inference processing time

## Architecture

### Multi-Threading Design

The application uses Qt's threading model for responsiveness:

```
┌─────────────────────────────────────────────────┐
│              Main UI Thread                     │
│  ┌─────────────┐  ┌─────────────┐              │
│  │ Video Widget│  │ Alert Panel │              │
│  └─────────────┘  └─────────────┘              │
└────────┬──────────────────┬────────────────────┘
         │                  │
    ┌────▼──────┐     ┌─────▼──────┐
    │  Signals  │     │  Signals   │
    └────┬──────┘     └─────┬──────┘
         │                  │
┌────────▼──────────────────▼────────────────────┐
│           Worker Threads                       │
│  ┌──────────────┐  ┌──────────────┐           │
│  │CaptureThread │  │InferenceThread│           │
│  │              │  │              │           │
│  │ • Read frames│  │ • ML detect  │           │
│  │ • FPS limit  │  │ • Track objs │           │
│  │ • Emit video │  │ • Emit dets  │           │
│  └──────────────┘  └──────────────┘           │
│  ┌──────────────┐                              │
│  │ AlertThread  │                              │
│  │              │                              │
│  │ • Poll alerts│                              │
│  │ • Emit new   │                              │
│  └──────────────┘                              │
└────────────────────────────────────────────────┘
```

### Thread Communication

All inter-thread communication uses Qt signals/slots for thread safety:
- **frame_signal**: Sends captured frames from Capture → Video Widget
- **result_signal**: Sends detections from Inference → Video Widget
- **new_alert_signal**: Sends alerts from Alert → Alert Panel
- **error_signal**: Reports errors to main UI thread

## Usage Guide

### Starting Monitoring

1. **Select Source**: Choose camera or video file
2. **Configure Settings**: Set confidence threshold and FPS limit
3. **Enable Overlays**: Check "Show Detections" and/or "Show Grid"
4. **Click Start**: Begin monitoring

### Managing Alerts

When alerts appear in the right panel:

1. **Review Alert**: Click to select and view details
2. **Confirm Alert**: If valid, click ✓ Confirm
3. **Dismiss Alert**: If false positive, click ✗ Dismiss
4. **Filter Alerts**: Use dropdowns to focus on specific types/priorities

### Adjusting Settings

While monitoring is active, you can:
- Change confidence threshold (updates immediately)
- Toggle detection overlay visibility
- Toggle grid overlay visibility

### Stopping Monitoring

Click the ■ Stop button to:
- Stop video capture
- Stop ML inference
- Clear the video display
- Retain active alerts

## Keyboard Shortcuts

- **Ctrl+Q**: Quit application
- **Space**: Start/Stop monitoring (when focused on control panel)

## Performance Optimization

### Recommended Settings

**For Live Camera Monitoring**:
- FPS Limit: 15-30 fps
- Confidence: 35-45%
- Hardware: GPU recommended for inference

**For Video File Analysis**:
- FPS Limit: 5-10 fps (slower for accuracy)
- Confidence: 40-50%
- Hardware: CPU acceptable

### Frame Queue Management

The application uses a bounded queue (maxsize=4) between capture and inference:
- **Queue Full**: Drops oldest frames (prioritizes low latency)
- **Queue Empty**: Inference thread waits for new frames

### Memory Usage

Typical memory footprint:
- Base application: ~200-300 MB
- Per active camera: +50-100 MB
- ML model (loaded): +200-400 MB

## Troubleshooting

### Video Not Displaying

**Symptoms**: Black screen or "No Video Feed"
**Solutions**:
- Check camera index (try 0, 1, 2 for different cameras)
- Verify video file path exists
- Check camera permissions
- Review terminal logs for OpenCV errors

### Slow Inference

**Symptoms**: High latency (>500ms), low FPS
**Solutions**:
- Lower FPS limit
- Increase confidence threshold
- Use GPU/OpenVINO acceleration
- Reduce video resolution

### Alerts Not Appearing

**Symptoms**: Expected alerts don't show in panel
**Solutions**:
- Verify alert use case is configured
- Check temporal consensus settings
- Review database connection
- Check filter settings (not filtering out alerts)

### Application Freezing

**Symptoms**: UI becomes unresponsive
**Solutions**:
- Check for blocking operations in main thread
- Verify worker threads are running
- Review error logs
- Restart application

## Configuration Files

### Config.yaml

Key settings for UI:
```yaml
streaming:
  frame_buffer_size: 4  # Queue size
  max_width: 1920
  max_height: 1080

ml:
  confidence_threshold: 0.35
  device: "GPU"  # or "CPU"

alerting:
  poll_interval: 5.0  # Alert check frequency
```

## Development Notes

### Adding New Widgets

1. Create widget class in `ui/widgets/`
2. Inherit from appropriate Qt widget class
3. Implement signals for thread communication
4. Add to main_window layout

### Extending Thread Functionality

1. Add new signals to thread class
2. Connect signals in main_window `_init_threads()`
3. Implement slots in target widget/window
4. Ensure thread-safe data passing

### Custom Styling

Edit `ui/resources/styles.qss` to customize appearance:
- Use Qt stylesheet syntax (CSS-like)
- Reload with `app.setStyleSheet(open('styles.qss').read())`
- Preview changes without restart

## Future Enhancements

Planned features:
- Multi-camera grid view
- Historical alert playback
- Export alert reports (PDF/CSV)
- Settings dialog for advanced configuration
- Planogram editor interface
- Statistics dashboard
