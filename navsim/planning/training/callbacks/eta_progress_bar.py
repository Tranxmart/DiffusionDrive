from typing import Any, List, Optional
import time

import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar


class EtaProgressBar(TQDMProgressBar):
    """TQDM progress bar that shows `Epoch {cur}/{total}` and an overall ETA.

    Extends the default Lightning progress bar to replace the bare
    `Epoch {n}` description with `Epoch {cur}/{max_epochs}` plus a
    coarse whole-training ETA estimated from the average per-epoch
    training time (validation time is not included in the estimate).
    """

    def __init__(self, refresh_rate: int = 1, process_position: int = 0) -> None:
        super().__init__(refresh_rate=refresh_rate, process_position=process_position)
        self._epoch_start_time: Optional[float] = None
        self._epoch_durations: List[float] = []

    @staticmethod
    def _format_seconds(seconds: float) -> str:
        seconds = int(round(seconds))
        hours, rem = divmod(seconds, 3600)
        minutes, secs = divmod(rem, 60)
        if hours:
            return f"{hours}h{minutes:02d}m"
        return f"{minutes}m{secs:02d}s"

    def _describe(self, trainer: pl.Trainer) -> str:
        current = trainer.current_epoch + 1
        total = trainer.max_epochs
        desc = f"Epoch {current}/{total}"
        if self._epoch_durations:
            avg = sum(self._epoch_durations) / len(self._epoch_durations)
            remaining_epochs = max(0, total - current)
            desc += f" | ETA {self._format_seconds(avg * remaining_epochs)}"
        return desc

    def on_train_epoch_start(self, trainer: pl.Trainer, *_args: Any) -> None:
        self._epoch_start_time = time.time()
        super().on_train_epoch_start(trainer, *_args)
        self.train_progress_bar.set_description(self._describe(trainer))

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, *args: Any) -> None:
        super().on_train_epoch_end(trainer, pl_module, *args)
        if self._epoch_start_time is not None:
            self._epoch_durations.append(time.time() - self._epoch_start_time)
            self._epoch_start_time = None