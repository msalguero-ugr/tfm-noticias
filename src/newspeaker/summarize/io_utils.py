from __future__ import annotations
import orjson
from typing import Iterable, Iterator, Dict, Any

def read_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    with open(path, "rb") as f:
        for line in f:
            if not line.strip():
                continue
            yield orjson.loads(line)

def write_jsonl(path: str, records: Iterable[Dict[str, Any]]) -> None:
    with open(path, "wb") as f:
        for rec in records:
            f.write(orjson.dumps(rec))
            f.write(b"\n")
