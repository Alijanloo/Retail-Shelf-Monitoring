from retail_shelf_monitoring.adaptors.tracking.bytetrack import SimpleTracker


class TestSimpleTracker:
    def test_tracker_initialization(self):
        tracker = SimpleTracker(track_thresh=0.5, match_thresh=0.3, max_age=30)

        assert tracker.track_thresh == 0.5
        assert tracker.match_thresh == 0.3
        assert tracker.max_age == 30
        assert tracker.next_track_id == 1
        assert len(tracker.tracks) == 0

    def test_tracker_creates_new_tracks(self):
        tracker = SimpleTracker()

        detections = [
            {
                "bbox": [100, 100, 200, 200],
                "confidence": 0.8,
                "class_id": 0,
            },
            {
                "bbox": [300, 300, 400, 400],
                "confidence": 0.9,
                "class_id": 1,
            },
        ]

        tracked = tracker.update(detections)

        assert len(tracked) == 2
        assert tracked[0]["track_id"] == 1
        assert tracked[1]["track_id"] == 2
        assert len(tracker.tracks) == 2

    def test_tracker_maintains_tracks_across_frames(self):
        tracker = SimpleTracker(match_thresh=0.3)

        # Frame 1
        detections_1 = [
            {"bbox": [100, 100, 200, 200], "confidence": 0.8, "class_id": 0}
        ]
        tracked_1 = tracker.update(detections_1)
        track_id_1 = tracked_1[0]["track_id"]

        # Frame 2 - same object slightly moved
        detections_2 = [
            {"bbox": [105, 105, 205, 205], "confidence": 0.8, "class_id": 0}
        ]
        tracked_2 = tracker.update(detections_2)
        track_id_2 = tracked_2[0]["track_id"]

        # Should maintain the same track ID
        assert track_id_1 == track_id_2
        assert len(tracker.tracks) == 1

    def test_tracker_filters_low_confidence_detections(self):
        tracker = SimpleTracker(track_thresh=0.5)

        detections = [
            {"bbox": [100, 100, 200, 200], "confidence": 0.3, "class_id": 0},
            {"bbox": [300, 300, 400, 400], "confidence": 0.8, "class_id": 1},
        ]

        tracked = tracker.update(detections)

        # Only high confidence detection should get a track
        assert len([d for d in tracked if "track_id" in d]) == 1

    def test_tracker_removes_old_tracks(self):
        tracker = SimpleTracker(max_age=2)

        # Frame 1: Create a track
        detections_1 = [
            {"bbox": [100, 100, 200, 200], "confidence": 0.8, "class_id": 0}
        ]
        tracker.update(detections_1)
        assert len(tracker.tracks) == 1

        # Frame 2-4: No detections (track ages)
        for _ in range(3):
            tracker.update([])

        # Track should be removed after max_age
        assert len(tracker.tracks) == 0

    def test_tracker_reset(self):
        tracker = SimpleTracker()

        detections = [{"bbox": [100, 100, 200, 200], "confidence": 0.8, "class_id": 0}]
        tracker.update(detections)

        assert len(tracker.tracks) > 0
        assert tracker.next_track_id > 1

        tracker.reset()

        assert len(tracker.tracks) == 0
        assert tracker.frame_count == 0

    def test_compute_iou(self):
        tracker = SimpleTracker()

        bbox1 = [0, 0, 100, 100]
        bbox2 = [50, 50, 150, 150]

        iou = tracker._compute_iou(bbox1, bbox2)

        # 50x50 overlap area / (10000 + 10000 - 2500) total area
        expected_iou = 2500 / 17500
        assert abs(iou - expected_iou) < 0.01

    def test_compute_iou_no_overlap(self):
        tracker = SimpleTracker()

        bbox1 = [0, 0, 100, 100]
        bbox2 = [200, 200, 300, 300]

        iou = tracker._compute_iou(bbox1, bbox2)

        assert iou == 0.0

    def test_tracker_handles_empty_detections(self):
        tracker = SimpleTracker()

        tracked = tracker.update([])

        assert len(tracked) == 0
        assert len(tracker.tracks) == 0
