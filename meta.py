from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Meta:
    path: Path
    name: str
    description: str

    @staticmethod
    def from_meta_text_row(path: str, name: str, description: str):
        assert path.startswith('[') and path.endswith(']')
        path = Path('datasets', path[1:-1])
        assert name.startswith('Name:')
        name = name[5:].strip()
        assert description.startswith('Description:')
        description = description[12:].strip()
        return Meta(path, name, description)


def get_meta() -> list[Meta]:
    meta = Path('meta.txt').read_text()
    meta = meta.split('\n\n')
    meta = [m.strip() for m in meta]
    meta = [m for m in meta if m]
    meta = [Meta.from_meta_text_row(*m.split('\n')) for m in meta]

    return meta
