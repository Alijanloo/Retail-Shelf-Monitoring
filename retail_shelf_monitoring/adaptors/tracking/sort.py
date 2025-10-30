from typing import List

import numpy as np
from scipy.optimize import linear_sum_assignment

from retail_shelf_monitoring.entities.common import BoundingBox
from retail_shelf_monitoring.entities.detection import Detection
from retail_shelf_monitoring.frameworks.logging_config import get_logger
from retail_shelf_monitoring.usecases.interfaces.tracker_interface import Tracker

logger = get_logger(__name__)


def iou_bbox(bbox1, bbox2):
    """IoU for [x1,y1,x2,y2]"""
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    a1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    a2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def convert_bbox_to_z(bbox):
    """
    from [x1,y1,x2,y2] to measurement [cx, cy, s, r]
    s = scale = area, r = aspect ratio = w/h
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    cx = bbox[0] + w / 2.0
    cy = bbox[1] + h / 2.0
    s = w * h
    r = w / float(h + 1e-6)
    return np.array([cx, cy, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    from state vector x to [x1,y1,x2,y2]
    state x: [cx, cy, s, r, vx, vy, vs]'
    """
    cx, cy, s, r = x[0], x[1], x[2], x[3]
    w = np.sqrt(max(s * r, 0))
    h = s / (w + 1e-6)
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    if score is None:
        return np.array([x1, y1, x2, y2]).reshape((1, 4))
    else:
        return np.array([x1, y1, x2, y2, score]).reshape((1, 5))


class KalmanBoxTracker:
    """
    Single target Kalman filter based tracker as used in SORT.
    State vector: 7-d [cx, cy, s, r, vx, vy, vs]
    Measurement: 4-d [cx, cy, s, r]
    """

    count = 0

    def __init__(self, detection: Detection):
        # store the full detection object
        self.detection = detection
        # extract bbox for kalman filter initialization
        bbox = [
            detection.bbox.x1,
            detection.bbox.y1,
            detection.bbox.x2,
            detection.bbox.y2,
        ]

        # initialize state
        self._x = np.zeros((7, 1))
        z = convert_bbox_to_z(bbox)
        self._x[0:4] = z
        # state covariance
        self._P = np.eye(7) * 10.0
        # motion matrix (F)
        self._F = np.eye(7)
        self._F[
            0, 4
        ] = 0  # no direct coupling cx<-vx in simple variant; keep simple linear
        # measurement matrix
        self._H = np.zeros((4, 7))
        self._H[0, 0] = 1
        self._H[1, 1] = 1
        self._H[2, 2] = 1
        self._H[3, 3] = 1
        # process and measurement noise
        self._R = np.eye(4) * 1.0
        self._Q = np.eye(7) * 0.01
        # bookkeeping
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count + 1
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 1
        self.hit_streak = 1
        self.age = 0

    def predict(self):
        # Simple constant velocity model for cx,cy,s
        # (we use vx,vy,vs in last three dims)
        F = np.eye(7)
        F[0, 4] = 1.0  # cx += vx
        F[1, 5] = 1.0  # cy += vy
        F[2, 6] = 1.0  # s  += vs
        # update state
        self._x = F @ self._x
        # propagate covariance
        self._P = F @ self._P @ F.T + self._Q
        self.age += 1
        self.time_since_update += 1
        # return predicted bbox
        return convert_x_to_bbox(self._x[:4].flatten())

    def update(self, detection: Detection):
        """Update with measurement detection"""
        # extract bbox from detection
        bbox = [
            detection.bbox.x1,
            detection.bbox.y1,
            detection.bbox.x2,
            detection.bbox.y2,
        ]
        z = convert_bbox_to_z(bbox)

        # update stored detection with new information
        self.detection = detection
        # ensure track_id is preserved/set
        self.detection.track_id = self.id

        # Kalman gain
        S = self._H @ self._P @ self._H.T + self._R
        K = self._P @ self._H.T @ np.linalg.inv(S)
        y = z - (self._H @ self._x)
        self._x = self._x + K @ y
        identity_matrix = np.eye(self._P.shape[0])
        self._P = (identity_matrix - K @ self._H) @ self._P
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1

    def get_state(self):
        return convert_x_to_bbox(self._x.flatten())

    def get_detection(self) -> Detection:
        """Get current detection with updated bbox from Kalman state"""
        bbox_coords = convert_x_to_bbox(self._x.flatten()).flatten()

        updated_detection = self.detection.model_copy(deep=True)
        updated_detection.bbox = BoundingBox(
            x1=float(bbox_coords[0]),
            y1=float(bbox_coords[1]),
            x2=float(bbox_coords[2]),
            y2=float(bbox_coords[3]),
        )
        updated_detection.track_id = self.id

        return updated_detection


class SortTracker(Tracker):
    """
    SORT tracker (lightweight). Use update(detections) each frame.
    Parameters:
      max_age - frames to keep alive without updates
      min_hits - frames before track is considered confirmed
      iou_threshold - matching IoU threshold
    """

    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers: List[KalmanBoxTracker] = []
        self.frame_count = 0

    def update(self, detections: List[Detection]) -> List[Detection]:
        """
        detections: list of Detection objects
        returns detections with "track_id" added for matched/new tracks
        """
        self.frame_count += 1
        # === predict existing trackers ===
        for trk in self.trackers:
            trk.predict()

        # === prepare detections array ===
        dets = (
            np.array([[d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2] for d in detections])
            if detections
            else np.empty((0, 4))
        )

        # === if no trackers, create trackers for all detections ===
        if len(self.trackers) == 0:
            for det in detections:
                trk = KalmanBoxTracker(det)
                self.trackers.append(trk)
                det.track_id = trk.id
            return detections

        # === compute IoU cost matrix between detections and trackers' last state ===
        trks_bboxes = np.array(
            [trk.get_state().flatten() for trk in self.trackers]
        ).reshape(-1, 4)
        if dets.shape[0] > 0 and trks_bboxes.shape[0] > 0:
            iou_matrix = np.zeros(
                (dets.shape[0], trks_bboxes.shape[0]), dtype=np.float32
            )
            for d, det in enumerate(dets):
                for t, tb in enumerate(trks_bboxes):
                    iou_matrix[d, t] = iou_bbox(det, tb)
            # Hungarian assignment on cost = 1 - iou
            cost = 1.0 - iou_matrix
            det_indices, trk_indices = linear_sum_assignment(cost)
            matches, unmatched_dets_idx, unmatched_trks_idx = [], [], []
            matched_det_idx_set, matched_trk_idx_set = set(), set()
            for d, t in zip(det_indices, trk_indices):
                if iou_matrix[d, t] < self.iou_threshold:
                    # treat as unmatched if IoU too low
                    unmatched_dets_idx.append(d)
                    unmatched_trks_idx.append(t)
                else:
                    matches.append((d, t))
                    matched_det_idx_set.add(d)
                    matched_trk_idx_set.add(t)
            for d in range(dets.shape[0]):
                if d not in matched_det_idx_set:
                    unmatched_dets_idx.append(d)
            for t in range(trks_bboxes.shape[0]):
                if t not in matched_trk_idx_set:
                    unmatched_trks_idx.append(t)
        else:
            matches = []
            unmatched_dets_idx = list(range(dets.shape[0]))
            unmatched_trks_idx = list(range(len(self.trackers)))

        # === update matched trackers with assigned detections ===
        for d, t in matches:
            self.trackers[t].update(detections[d])
            # attach track id to detection
            detections[d].track_id = self.trackers[t].id

        # === create new trackers for unmatched detections ===
        for idx in unmatched_dets_idx:
            trk = KalmanBoxTracker(detections[idx])
            self.trackers.append(trk)
            detections[idx].track_id = trk.id

        # === handle unmatched trackers: increase time_since_update and remove stale ===
        for idx in unmatched_trks_idx:
            trk = self.trackers[idx]
            trk.time_since_update += 1
        # remove dead trackers
        new_trackers = []
        for trk in self.trackers:
            if trk.time_since_update > self.max_age:
                # drop
                continue
            new_trackers.append(trk)
        self.trackers = new_trackers

        # optionally: return only detections whose track is confirmed
        # (hit_streak >= min_hits) but for compatibility we'll return
        # all detections annotated with assigned track_id
        logger.debug(
            f"SORT update: frame={self.frame_count}, trackers={len(self.trackers)}"
        )
        return detections

    def predict(self) -> List[Detection]:
        detections = []
        for trk in self.trackers:
            trk.predict()  # updates internal state
            # get detection with predicted bbox
            predicted_detection = trk.get_detection()
            detections.append(predicted_detection)
        return detections

    def reset(self):
        self.trackers = []
        self.frame_count = 0
        logger.info("SORT tracker reset")
