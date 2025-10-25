import time
from pathlib import Path
from queue import Full, Queue
from threading import Thread
from typing import Optional

import cv2

from ...entities.stream import StreamConfig
from ...frameworks.exceptions import ValidationError
from ...frameworks.logging_config import get_logger

logger = get_logger(__name__)


class StreamReader:
    def __init__(
        self, stream_config: StreamConfig, frame_queue: Queue, max_queue_size: int = 100
    ):
        self.config = stream_config
        self.frame_queue = frame_queue
        self.max_queue_size = max_queue_size

        self.capture: Optional[cv2.VideoCapture] = None
        self.is_running = False
        self.thread: Optional[Thread] = None
        self.frame_count = 0
        self.dropped_frames = 0

    def start(self):
        if self.is_running:
            logger.warning(f"Stream {self.config.stream_id} already running")
            return

        self._open_stream()

        self.is_running = True
        self.thread = Thread(target=self._read_loop, daemon=True)
        self.thread.start()

        logger.info(f"Started stream reader for {self.config.stream_id}")

    def stop(self):
        if not self.is_running:
            return

        self.is_running = False

        if self.thread:
            self.thread.join(timeout=5.0)

        if self.capture:
            self.capture.release()

        logger.info(
            f"Stopped stream {self.config.stream_id}. "
            f"Frames: {self.frame_count}, Dropped: {self.dropped_frames}"
        )

    def _open_stream(self):
        source = self.config.source_url

        if source.startswith("rtsp://") or source.startswith("http://"):
            self.capture = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            path = Path(source)
            if not path.exists():
                raise FileNotFoundError(f"Video file not found: {source}")
            self.capture = cv2.VideoCapture(str(path))

        if not self.capture.isOpened():
            raise ValidationError(f"Failed to open stream: {source}")

        actual_fps = self.capture.get(cv2.CAP_PROP_FPS)
        width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        logger.info(
            f"Opened stream {self.config.stream_id}: "
            f"{width}x{height} @ {actual_fps} FPS"
        )

    def _read_loop(self):
        frame_interval = 1.0 / self.config.fps
        last_frame_time = 0

        while self.is_running:
            current_time = time.time()

            if current_time - last_frame_time < frame_interval:
                time.sleep(0.001)
                continue

            ret, frame = self.capture.read()

            if not ret:
                logger.warning(
                    f"Failed to read frame from stream {self.config.stream_id}"
                )

                if not self.config.source_url.startswith("rtsp"):
                    self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    self._reconnect()
                    continue

            self.frame_count += 1

            if self.frame_count % self.config.process_every_n_frames != 0:
                continue

            if self.config.max_width or self.config.max_height:
                frame = self._resize_frame(frame)

            try:
                self.frame_queue.put(
                    {
                        "frame": frame,
                        "stream_id": self.config.stream_id,
                        "frame_number": self.frame_count,
                        "timestamp": current_time,
                    },
                    block=False,
                )
                last_frame_time = current_time
            except Full:
                self.dropped_frames += 1
                if self.dropped_frames % 100 == 0:
                    logger.warning(
                        f"Stream {self.config.stream_id}: "
                        f"Dropped {self.dropped_frames} frames"
                    )

    def _resize_frame(self, frame):
        height, width = frame.shape[:2]

        scale = 1.0
        if self.config.max_width and width > self.config.max_width:
            scale = min(scale, self.config.max_width / width)
        if self.config.max_height and height > self.config.max_height:
            scale = min(scale, self.config.max_height / height)

        if scale < 1.0:
            new_width = int(width * scale)
            new_height = int(height * scale)
            frame = cv2.resize(
                frame, (new_width, new_height), interpolation=cv2.INTER_AREA
            )

        return frame

    def _reconnect(self):
        logger.info(f"Attempting to reconnect stream {self.config.stream_id}")

        if self.capture:
            self.capture.release()

        time.sleep(2.0)

        try:
            self._open_stream()
            logger.info(f"Successfully reconnected stream {self.config.stream_id}")
        except Exception as e:
            logger.error(f"Failed to reconnect stream {self.config.stream_id}: {e}")
