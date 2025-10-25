from typing import Dict, List

import numpy as np

from ...frameworks.logging_config import get_logger
from .tracker_interface import Tracker

logger = get_logger(__name__)


class SimpleTracker(Tracker):
    """
    Simplified tracking based on IoU matching and Kalman filtering
    Note: For production, integrate actual ByteTrack from external library
    """

    def __init__(
        self,
        track_thresh: float = 0.5,
        match_thresh: float = 0.3,
        max_age: int = 30,
    ):
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.max_age = max_age

        self.tracks = {}
        self.next_track_id = 1
        self.frame_count = 0

    def update(self, detections: List[Dict]) -> List[Dict]:
        self.frame_count += 1

        high_conf_dets = [d for d in detections if d["confidence"] >= self.track_thresh]

        (
            matched_tracks,
            unmatched_dets,
            unmatched_tracks,
        ) = self._match_detections_to_tracks(high_conf_dets)

        for track_id, detection in matched_tracks:
            self.tracks[track_id]["bbox"] = detection["bbox"]
            self.tracks[track_id]["age"] = 0
            self.tracks[track_id]["confidence"] = detection["confidence"]
            detection["track_id"] = track_id

        for detection in unmatched_dets:
            track_id = self.next_track_id
            self.next_track_id += 1

            self.tracks[track_id] = {
                "bbox": detection["bbox"],
                "age": 0,
                "confidence": detection["confidence"],
            }
            detection["track_id"] = track_id

        for track_id in unmatched_tracks:
            self.tracks[track_id]["age"] += 1

        dead_tracks = [
            tid for tid, track in self.tracks.items() if track["age"] > self.max_age
        ]
        for tid in dead_tracks:
            del self.tracks[tid]

        logger.debug(
            f"Tracking update: {len(matched_tracks)} matched, "
            f"{len(unmatched_dets)} new, {len(dead_tracks)} removed, "
            f"{len(self.tracks)} active tracks"
        )

        return detections

    def _match_detections_to_tracks(self, detections):
        if not self.tracks or not detections:
            return [], detections, list(self.tracks.keys())

        track_ids = list(self.tracks.keys())
        track_bboxes = [self.tracks[tid]["bbox"] for tid in track_ids]
        det_bboxes = [d["bbox"] for d in detections]

        iou_matrix = self._compute_iou_matrix(det_bboxes, track_bboxes)

        matched_tracks = []
        unmatched_dets = []
        matched_det_indices = set()
        matched_track_indices = set()

        matches = []
        for i in range(len(detections)):
            for j in range(len(track_ids)):
                if iou_matrix[i, j] >= self.match_thresh:
                    matches.append((i, j, iou_matrix[i, j]))

        matches.sort(key=lambda x: x[2], reverse=True)

        for det_idx, track_idx, iou in matches:
            if det_idx in matched_det_indices or track_idx in matched_track_indices:
                continue

            matched_tracks.append((track_ids[track_idx], detections[det_idx]))
            matched_det_indices.add(det_idx)
            matched_track_indices.add(track_idx)

        for i, det in enumerate(detections):
            if i not in matched_det_indices:
                unmatched_dets.append(det)

        unmatched_track_ids = [
            track_ids[j]
            for j in range(len(track_ids))
            if j not in matched_track_indices
        ]

        return matched_tracks, unmatched_dets, unmatched_track_ids

    def _compute_iou_matrix(self, bboxes1, bboxes2):
        """Compute IoU between two sets of bboxes"""
        n1 = len(bboxes1)
        n2 = len(bboxes2)
        iou_matrix = np.zeros((n1, n2))

        for i, bbox1 in enumerate(bboxes1):
            for j, bbox2 in enumerate(bboxes2):
                iou_matrix[i, j] = self._compute_iou(bbox1, bbox2)

        return iou_matrix

    def _compute_iou(self, bbox1, bbox2):
        """Compute IoU between two bboxes [x1, y1, x2, y2]"""
        x1_inter = max(bbox1[0], bbox2[0])
        y1_inter = max(bbox1[1], bbox2[1])
        x2_inter = min(bbox1[2], bbox2[2])
        y2_inter = min(bbox1[3], bbox2[3])

        if x2_inter <= x1_inter or y2_inter <= y1_inter:
            return 0.0

        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)

        bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

        union_area = bbox1_area + bbox2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    def reset(self):
        self.tracks = {}
        self.frame_count = 0
        logger.info("Tracker reset")
