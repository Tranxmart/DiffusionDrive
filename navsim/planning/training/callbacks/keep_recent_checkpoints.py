from pathlib import Path
from typing import Any

import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info, rank_zero_warn


class KeepRecentCheckpoints(pl.callbacks.ModelCheckpoint):
    """ModelCheckpoint that keeps the most RECENT N epoch checkpoints.

    Unlike the default monitor-based ranking (which keeps the BEST epochs and
    can therefore hold on to early checkpoints when the monitored metric
    degrades), this callback saves every epoch and deletes the oldest files
    once more than `keep_recent` epoch checkpoints exist in the directory.

    Additionally, checkpoints at epochs divisible by `every_epoch_keep` are
    pinned and never deleted, so milestone epochs (e.g. 0, 10, 20, ...) are
    always available for comparison/evaluation. `last.ckpt` is never deleted.
    """

    def __init__(
        self,
        keep_recent: int = 10,
        every_epoch_keep: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(save_top_k=-1, **kwargs)
        self.keep_recent = keep_recent
        self.every_epoch_keep = every_epoch_keep

    @staticmethod
    def _epoch_of(p: Path) -> int:
        stem = p.name.replace("epoch=epoch=", "epoch=").replace("-step=step=", "-step=")
        for part in stem.split("-"):
            if part.startswith("epoch="):
                try:
                    return int(part.split("=", 1)[1])
                except ValueError:
                    break
        return -1

    def _is_pinned(self, p: Path) -> bool:
        # last.ckpt is always pinned.
        if p.name in ("last.ckpt", "last.ckpt.tmp"):
            return True
        # The best-trajectory-epoch checkpoint is owned by
        # BestTrajectoryCheckpoint and must never be pruned here.
        if p.name.startswith("best_epoch"):
            return True
        if self.every_epoch_keep and self.every_epoch_keep > 0:
            epoch = self._epoch_of(p)
            if epoch >= 0 and epoch % self.every_epoch_keep == 0:
                return True
        return False

    def _remove_oldest(self, trainer: "pl.Trainer") -> None:
        # Only rank 0 prunes; other ranks would race on file deletion.
        if trainer.is_global_zero is False:
            return
        # Called after each save; prune oldest non-pinned epoch checkpoints.
        try:
            dirpath = Path(self.dirpath)
            ckpts = [p for p in dirpath.glob("*.ckpt") if not self._is_pinned(p)]
            if len(ckpts) <= self.keep_recent:
                return
            # Oldest first: by epoch extracted from filename, fall back to mtime.
            sortable = [(self._epoch_of(p), p.stat().st_mtime, p) for p in ckpts]
            sortable.sort(key=lambda t: (t[0], t[1]))
            for _, _, p in sortable[: len(sortable) - self.keep_recent]:
                rank_zero_info(f"Removing old checkpoint: {p.name}")
                p.unlink(missing_ok=True)
        except Exception as exc:  # pragma: no cover - never kill training for cleanup
            rank_zero_warn(f"Checkpoint cleanup failed: {exc}")

    def _save_checkpoint(self, trainer: "pl.Trainer", filepath: str) -> None:
        super()._save_checkpoint(trainer, filepath)
        self._remove_oldest(trainer)