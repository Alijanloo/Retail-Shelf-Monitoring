from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from ..entities.common import AlertType, CellState
from ..frameworks.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class CellTemporalState:
    shelf_id: str
    row_idx: int
    item_idx: int
    expected_sku: str

    current_state: CellState = CellState.UNKNOWN
    consecutive_empty_frames: int = 0
    consecutive_misplaced_frames: int = 0
    consecutive_ok_frames: int = 0

    last_detected_sku: Optional[str] = None
    last_update_time: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    state_history: List[CellState] = field(default_factory=list)
    max_history_length: int = 10

    def update(self, new_state: CellState, detected_sku: Optional[str] = None):
        self.last_update_time = datetime.now(timezone.utc)
        self.last_detected_sku = detected_sku

        self.state_history.append(new_state)
        if len(self.state_history) > self.max_history_length:
            self.state_history.pop(0)

        if new_state == CellState.EMPTY:
            self.consecutive_empty_frames += 1
            self.consecutive_misplaced_frames = 0
            self.consecutive_ok_frames = 0
        elif new_state == CellState.MISPLACED:
            self.consecutive_empty_frames = 0
            self.consecutive_misplaced_frames += 1
            self.consecutive_ok_frames = 0
        elif new_state == CellState.OK:
            self.consecutive_empty_frames = 0
            self.consecutive_misplaced_frames = 0
            self.consecutive_ok_frames += 1
        else:
            pass

        self.current_state = new_state

    def is_oscillating(self, window: int = 5) -> bool:
        if len(self.state_history) < window:
            return False

        recent_states = self.state_history[-window:]
        unique_states = set(recent_states)

        return len(unique_states) > 2

    def should_trigger_oos_alert(self, n_confirm: int) -> bool:
        return self.consecutive_empty_frames >= n_confirm and not self.is_oscillating()

    def should_trigger_misplacement_alert(self, n_confirm: int) -> bool:
        return (
            self.consecutive_misplaced_frames >= n_confirm and not self.is_oscillating()
        )

    def should_clear_alert(self, n_clear: int) -> bool:
        return self.consecutive_ok_frames >= n_clear


class TemporalConsensusManager:
    def __init__(
        self,
        n_confirm: int = 3,
        n_clear: int = 2,
        state_timeout: timedelta = timedelta(minutes=5),
    ):
        self.n_confirm = n_confirm
        self.n_clear = n_clear
        self.state_timeout = state_timeout

        self.cell_states: Dict[str, Dict[tuple, CellTemporalState]] = defaultdict(dict)

    def update_cell_states(self, shelf_id: str, cell_state_updates: List[Dict]) -> Dict:
        new_oos_alerts = []
        new_misplacement_alerts = []
        cleared_alerts = []

        for update in cell_state_updates:
            row_idx = update["row_idx"]
            item_idx = update["item_idx"]
            state = update["state"]
            expected_sku = update["expected_sku"]
            detected_sku = update.get("detected_sku")

            cell_key = (row_idx, item_idx)

            if cell_key not in self.cell_states[shelf_id]:
                self.cell_states[shelf_id][cell_key] = CellTemporalState(
                    shelf_id=shelf_id,
                    row_idx=row_idx,
                    item_idx=item_idx,
                    expected_sku=expected_sku,
                )

            cell_state = self.cell_states[shelf_id][cell_key]
            previous_state = cell_state.current_state

            cell_state.update(state, detected_sku)

            if cell_state.should_trigger_oos_alert(self.n_confirm):
                if (
                    previous_state != CellState.EMPTY
                    or cell_state.consecutive_empty_frames == self.n_confirm
                ):
                    new_oos_alerts.append(
                        {
                            "shelf_id": shelf_id,
                            "row_idx": row_idx,
                            "item_idx": item_idx,
                            "expected_sku": expected_sku,
                            "detected_sku": None,
                            "consecutive_frames": cell_state.consecutive_empty_frames,
                            "alert_type": AlertType.OOS,
                        }
                    )

            if cell_state.should_trigger_misplacement_alert(self.n_confirm):
                if (
                    previous_state != CellState.MISPLACED
                    or cell_state.consecutive_misplaced_frames == self.n_confirm
                ):
                    new_misplacement_alerts.append(
                        {
                            "shelf_id": shelf_id,
                            "row_idx": row_idx,
                            "item_idx": item_idx,
                            "expected_sku": expected_sku,
                            "detected_sku": detected_sku,
                            "consecutive_frames": (
                                cell_state.consecutive_misplaced_frames
                            ),
                            "alert_type": AlertType.MISPLACEMENT,
                        }
                    )

            if cell_state.should_clear_alert(self.n_clear):
                history_len = len(cell_state.state_history)
                window_start = -(self.n_confirm + self.n_clear)
                window_end = -self.n_clear

                had_oos_alert = (
                    history_len >= self.n_confirm + self.n_clear
                    and cell_state.state_history[window_start:window_end].count(
                        CellState.EMPTY
                    )
                    >= self.n_confirm
                )
                had_misplacement_alert = (
                    history_len >= self.n_confirm + self.n_clear
                    and cell_state.state_history[window_start:window_end].count(
                        CellState.MISPLACED
                    )
                    >= self.n_confirm
                )

                if had_oos_alert or had_misplacement_alert:
                    cleared_alerts.append(
                        {
                            "shelf_id": shelf_id,
                            "row_idx": row_idx,
                            "item_idx": item_idx,
                        }
                    )

        self._cleanup_stale_states()

        logger.info(
            f"Temporal consensus update: {len(new_oos_alerts)} new OOS, "
            f"{len(new_misplacement_alerts)} new misplacements, "
            f"{len(cleared_alerts)} cleared"
        )

        return {
            "new_alerts": new_oos_alerts + new_misplacement_alerts,
            "cleared_alerts": cleared_alerts,
        }

    def _cleanup_stale_states(self):
        current_time = datetime.now(timezone.utc)

        for shelf_id in list(self.cell_states.keys()):
            for cell_key in list(self.cell_states[shelf_id].keys()):
                cell_state = self.cell_states[shelf_id][cell_key]

                if current_time - cell_state.last_update_time > self.state_timeout:
                    del self.cell_states[shelf_id][cell_key]

    def get_cell_state(
        self, shelf_id: str, row_idx: int, item_idx: int
    ) -> Optional[CellTemporalState]:
        cell_key = (row_idx, item_idx)
        return self.cell_states.get(shelf_id, {}).get(cell_key)
