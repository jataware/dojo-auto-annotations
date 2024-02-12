from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Meta:
    path: Path
    name: str
    description: str

    def __init__(self, path: str, name: str, description: str):
        assert path.startswith('[') and path.endswith(']')
        self.path = Path('datasets', path[1:-1])
        assert name.startswith('Name:')
        self.name = name[5:].strip()
        assert description.startswith('Description:')
        self.description = description[12:].strip()


def get_meta() -> list[Meta]:
    meta = Path('meta.txt').read_text()
    meta = meta.split('\n\n')
    meta = [m.strip() for m in meta]
    meta = [m for m in meta if m]
    meta = [Meta(*m.split('\n')) for m in meta]

    return meta
