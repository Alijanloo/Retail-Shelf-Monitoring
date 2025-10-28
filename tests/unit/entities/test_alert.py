from datetime import datetime

import pytest

from retail_shelf_monitoring.entities.alert import Alert
from retail_shelf_monitoring.entities.common import AlertType


class TestAlert:
    def test_alert_creation(self):
        now = datetime.utcnow()
        alert = Alert(
            shelf_id="SHELF-001",
            row_idx=0,
            item_idx=2,
            alert_type=AlertType.OOS,
            expected_sku="SKU-123",
            first_seen=now,
            last_seen=now,
        )
        assert alert.shelf_id == "SHELF-001"
        assert alert.row_idx == 0
        assert alert.item_idx == 2
        assert alert.alert_type == AlertType.OOS
        assert alert.expected_sku == "SKU-123"
        assert alert.confirmed is False
        assert alert.dismissed is False

    def test_alert_misplacement_type(self):
        now = datetime.utcnow()
        alert = Alert(
            shelf_id="SHELF-001",
            row_idx=1,
            item_idx=3,
            alert_type=AlertType.MISPLACEMENT,
            expected_sku="SKU-123",
            detected_sku="SKU-456",
            first_seen=now,
            last_seen=now,
        )
        assert alert.alert_type == AlertType.MISPLACEMENT
        assert alert.detected_sku == "SKU-456"

    def test_alert_confirmation_validation(self):
        now = datetime.utcnow()

        with pytest.raises(
            ValueError, match="confirmed_at can only be set when confirmed is True"
        ):
            Alert(
                shelf_id="SHELF-001",
                row_idx=0,
                item_idx=0,
                alert_type=AlertType.OOS,
                first_seen=now,
                last_seen=now,
                confirmed=False,
                confirmed_at=now,
            )

    def test_alert_confirmed_by_validation(self):
        now = datetime.utcnow()

        with pytest.raises(
            ValueError, match="confirmed_by can only be set when confirmed is True"
        ):
            Alert(
                shelf_id="SHELF-001",
                row_idx=0,
                item_idx=0,
                alert_type=AlertType.OOS,
                first_seen=now,
                last_seen=now,
                confirmed=False,
                confirmed_by="STAFF-001",
            )

    def test_alert_confirmed(self):
        now = datetime.utcnow()
        alert = Alert(
            shelf_id="SHELF-001",
            row_idx=0,
            item_idx=0,
            alert_type=AlertType.OOS,
            first_seen=now,
            last_seen=now,
            confirmed=True,
            confirmed_by="STAFF-001",
            confirmed_at=now,
        )
        assert alert.confirmed is True
        assert alert.confirmed_by == "STAFF-001"
        assert alert.confirmed_at == now

    def test_alert_evidence_paths(self):
        now = datetime.utcnow()
        evidence = ["/path/to/frame1.jpg", "/path/to/frame2.jpg"]
        alert = Alert(
            shelf_id="SHELF-001",
            row_idx=0,
            item_idx=0,
            alert_type=AlertType.OOS,
            first_seen=now,
            last_seen=now,
            evidence_paths=evidence,
        )
        assert alert.evidence_paths == evidence

    def test_alert_consecutive_frames(self):
        now = datetime.utcnow()
        alert = Alert(
            shelf_id="SHELF-001",
            row_idx=0,
            item_idx=0,
            alert_type=AlertType.OOS,
            first_seen=now,
            last_seen=now,
            consecutive_frames=5,
        )
        assert alert.consecutive_frames == 5
