from unittest.mock import MagicMock

from retail_shelf_monitoring.adaptors.tracking.sort import SortTracker

# from retail_shelf_monitoring.entities.MagicMock import MagicMock


class TestSortTracker:
    def test_tracker_creates_new_tracks(self):
        tracker = SortTracker()

        detections = [
            MagicMock(
                **{
                    "bbox": [100, 100, 200, 200],
                    "confidence": 0.8,
                    "class_id": 0,
                }
            ),
            MagicMock(
                **{
                    "bbox": [300, 300, 400, 400],
                    "confidence": 0.9,
                    "class_id": 1,
                }
            ),
        ]

        tracked = tracker.update(detections)

        assert len(tracked) == 2
        assert tracked[0].track_id == 1
        assert tracked[1].track_id == 2
        assert len(tracker.trackers) == 2

    def test_tracker_maintains_tracks_across_frames(self):
        tracker = SortTracker(match_thresh=0.3)

        # Frame 1
        detections_1 = [
            MagicMock(
                **{"bbox": [100, 100, 200, 200], "confidence": 0.8, "class_id": 0}
            )
        ]
        tracked_1 = tracker.update(detections_1)
        track_id_1 = tracked_1[0].track_id

        # Frame 2 - same object slightly moved
        detections_2 = [
            MagicMock(
                **{"bbox": [105, 105, 205, 205], "confidence": 0.8, "class_id": 0}
            )
        ]
        tracked_2 = tracker.update(detections_2)
        track_id_2 = tracked_2[0].track_id

        # Should maintain the same track ID
        assert track_id_1 == track_id_2
        assert len(tracker.trackers) == 1

    def test_tracker_filters_low_confidence_detections(self):
        tracker = SortTracker(iou_threshold=1)

        detections = [
            MagicMock(
                **{"bbox": [100, 100, 200, 200], "confidence": 0.3, "class_id": 0}
            ),
            MagicMock(
                **{"bbox": [300, 300, 400, 400], "confidence": 0.8, "class_id": 1}
            ),
        ]

        tracked = tracker.update(detections)

        # Only high confidence MagicMock should get a track
        assert len([d for d in tracked if d.track_id is not None]) == 1

    def test_tracker_removes_old_tracks(self):
        tracker = SortTracker(max_age=2)

        # Frame 1: Create a track
        detections_1 = [
            MagicMock(
                **{"bbox": [100, 100, 200, 200], "confidence": 0.8, "class_id": 0}
            )
        ]
        tracker.update(detections_1)
        assert len(tracker.trackers) == 1

        # Frame 2-4: No detections (track ages)
        for _ in range(3):
            tracker.update([])

        # Track should be removed after max_age
        assert len(tracker.trackers) == 0

    def test_tracker_reset(self):
        tracker = SortTracker()

        detections = [
            MagicMock(
                **{"bbox": [100, 100, 200, 200], "confidence": 0.8, "class_id": 0}
            )
        ]
        tracker.update(detections)

        assert len(tracker.trackers) > 0

        tracker.reset()

        assert len(tracker.trackers) == 0
        assert tracker.frame_count == 0

    def detections(self):
        tracker = SortTracker()

        tracked = tracker.update([])

        assert len(tracked) == 0
