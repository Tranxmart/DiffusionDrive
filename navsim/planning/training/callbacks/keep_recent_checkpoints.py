from pathlib import Path
from typing import Any

import pytorch_lightning as pl


class KeepRecentCheckpoints(pl.callbacks.ModelCheckpoint):
    """ModelCheckpoint that keeps the most RECENT N epoch checkpoints.

    Unlike the default monitor-based ranking (which keeps the BEST epochs and
    can therefore hold on to early checkpoints when the monitored metric
    degrades), this callback saves every epoch and deletes the oldest files
    once more than `keep_recent` epoch checkpoints exist in the directory.
    `last.ckpt` is never deleted.
    """

    def __init__(self, keep_recent: int = 10, **kwargs: Any) -> None:
        super().__init__(save_top_k=-1, **kwargs)
        self.keep_recent = keep_recent

    def _remove_oldest(self, trainer: "pl.Trainer") -> None:
        # Called after each save; prune oldest epoch checkpoints.
        try:
            dirpath = Path(self.dirpath)
            ckpts = [
                p
                for p in dirpath.glob("*.ckpt")
                if p.name not in ("last.ckpt", "last.ckpt.tmp")
            ]
            if len(ckpts) <= self.keep_recent:
                return
            # Oldest first: by epoch extracted from filename, fall back to mtime.
            def epoch_of(p: Path) -> int:
                stem = p.name.replace("epoch=epoch=", "epoch=").replace("-step=step=", "-step=")
                for part in stem.split("-"):
                    if part.startswith("epoch="):
                        try:
                            return int(part.split("=", 1)[1])
                        except ValueError:
                            break
                return -1

            sortable = [(epoch_of(p), p.stat().st_mtime, p) for p in ckpts]
            sortable.sort(key=lambda t: (t[0], t[1]))
            for _, _, p in sortable[: len(sortable) - self.keep_recent]:
                trainer.strategy.print(f"Removing old checkpoint: {p.name}")
                p.unlink(missing_ok=True)
        except Exception as exc:  # pragma: no cover - never kill training for cleanup
            trainer.strategy.print(f"WARNING: checkpoint cleanup failed: {exc}")

    def _save_checkpoint(self, trainer: "pl.Trainer", filepath: str) -> None:
        super()._save_checkpoint(trainer, filepath)
        self._remove_oldest(trainer)