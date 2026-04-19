import json
from pathlib import Path
from datetime import datetime


class JsonlLogger:
    def __init__(self, output_dir: Path, model_name: str):
        output_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = model_name.replace("/", "_").replace(":", "_")
        self._path = output_dir / f"{safe_name}_{ts}.jsonl"
        self._file = self._path.open("w", encoding="utf-8")

    def write(self, entry: dict) -> None:
        self._file.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()

    @property
    def path(self) -> Path:
        return self._path
