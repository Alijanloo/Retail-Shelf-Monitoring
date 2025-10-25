import uuid
from datetime import datetime
from pathlib import Path
from queue import Queue
from typing import Dict, Optional

import cv2

from ..adaptors.video.stream_reader import StreamReader
from ..adaptors.vision.image_aligner import ShelfAligner
from ..entities.frame import Frame
from ..entities.stream import StreamConfig
from ..frameworks.logging_config import get_logger

logger = get_logger(__name__)


class StreamProcessingUseCase:
    def __init__(
        self, shelf_aligner: ShelfAligner, output_dir: str = "data/aligned_frames"
    ):
        self.shelf_aligner = shelf_aligner
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.stream_readers: Dict[str, StreamReader] = {}
        self.frame_queue = Queue(maxsize=100)

    def start_stream(self, stream_config: StreamConfig):
        if stream_config.stream_id in self.stream_readers:
            logger.warning(f"Stream {stream_config.stream_id} already started")
            return

        reader = StreamReader(stream_config=stream_config, frame_queue=self.frame_queue)

        reader.start()
        self.stream_readers[stream_config.stream_id] = reader

        logger.info(f"Started stream: {stream_config.stream_id}")

    def stop_stream(self, stream_id: str):
        if stream_id not in self.stream_readers:
            logger.warning(f"Stream {stream_id} not found")
            return

        reader = self.stream_readers.pop(stream_id)
        reader.stop()

        logger.info(f"Stopped stream: {stream_id}")

    def stop_all_streams(self):
        for stream_id in list(self.stream_readers.keys()):
            self.stop_stream(stream_id)

    def process_frame_queue(self) -> Optional[dict]:
        if self.frame_queue.empty():
            return None

        frame_data = self.frame_queue.get()

        frame_metadata = Frame(
            frame_id=str(uuid.uuid4()),
            stream_id=frame_data["stream_id"],
            timestamp=datetime.fromtimestamp(frame_data["timestamp"]),
            frame_number=frame_data["frame_number"],
            width=frame_data["frame"].shape[1],
            height=frame_data["frame"].shape[0],
            is_keyframe=True,
        )

        alignment_result = self.shelf_aligner.align_to_best_reference(
            frame=frame_data["frame"], frame_metadata=frame_metadata
        )

        if alignment_result is None:
            logger.debug(
                f"Frame {frame_metadata.frame_id} could not be aligned to any shelf"
            )
            return None

        shelf_id, updated_metadata, aligned_image = alignment_result

        aligned_path = self.output_dir / f"{shelf_id}_{frame_metadata.frame_id}.jpg"
        cv2.imwrite(str(aligned_path), aligned_image)

        logger.info(
            f"Processed frame {frame_metadata.frame_id}: aligned to shelf {shelf_id}"
        )

        return {
            "frame_metadata": updated_metadata,
            "aligned_image_path": str(aligned_path),
            "aligned_image": aligned_image,
            "shelf_id": shelf_id,
        }
