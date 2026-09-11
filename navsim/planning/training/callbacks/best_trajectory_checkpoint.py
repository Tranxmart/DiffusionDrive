from pathlib import Path
from typing import Any

import pytorch_lightning as pl


class BestTrajectoryCheckpoint(pl.callbacks.ModelCheckpoint):
    """Saves the epoch with the LOWEST ``val/trajectory_loss_epoch``.

    Writes a single file named ``best_epoch_{epoch_id}.ckpt`` in the same
    checkpoint directory. When a new best epoch is found, the previous best
    file is removed, so at most one ``best_epoch_*.ckpt`` exists at a time.
    This complements KeepRecentCheckpoints (recent-N + pinned milestones):
    recent/milestone policies protect against late degradation, while this
    one captures the single best-trajectory epoch for evaluation.
    """

    # match the metric logged by AgentLightningModule (on_epoch=True)
    METRIC = "val/trajectory_loss_epoch"

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("monitor", self.METRIC)
        kwargs.setdefault("mode", "min")
        kwargs.setdefault("save_top_k", 1)
        kwargs.setdefault("every_n_epochs", 1)
        kwargs.setdefault("filename", "best_epoch_{epoch}")
        kwargs.setdefault("auto_insert_metric_name", False)
        super().__init__(**kwargs)

    def _remove_checkpoint(self, trainer: "pl.Trainer", filepath: str) -> None:
        # Replace the default (silent) removal with a log line; exactly the
        # previous best file is passed here by ModelCheckpoint bookkeeping.
        try:
            rank_zero = trainer.is_global_zero
        except Exception:
            rank_zero = True
        if rank_zero:
            from pytorch_lightning.utilities import rank_zero_info

            rank_zero_info(f"New best trajectory epoch: removing previous {Path(filepath).name}")
        super()._remove_checkpoint(trainer, filepath)
