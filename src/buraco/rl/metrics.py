"""CSV metric sinks (primary) with an optional TensorBoard mirror."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


class CsvLogger:
    """Append-only CSV with a fixed header; safe to reopen on resume."""

    def __init__(self, path: Path, fieldnames: list[str]):
        self.path = path
        self.fieldnames = fieldnames
        new = not path.exists() or path.stat().st_size == 0
        self._fh = open(path, "a", newline="")
        self._writer = csv.DictWriter(self._fh, fieldnames=fieldnames)
        if new:
            self._writer.writeheader()
            self._fh.flush()

    def log(self, row: dict[str, Any]) -> None:
        self._writer.writerow({k: row.get(k, "") for k in self.fieldnames})
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


class TensorBoardLogger:
    """Optional mirror; constructed only when --tensorboard is passed."""

    def __init__(self, log_dir: Path):
        from torch.utils.tensorboard import SummaryWriter  # noqa: PLC0415

        self._writer = SummaryWriter(str(log_dir))

    def log(self, step: int, scalars: dict[str, float], prefix: str = "") -> None:
        for name, value in scalars.items():
            if isinstance(value, (int, float)):
                self._writer.add_scalar(f"{prefix}{name}", value, step)

    def close(self) -> None:
        self._writer.close()
