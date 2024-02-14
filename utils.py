from __future__ import annotations

from typing import TypeVar

# Could be more complicated, e.g. use UI to ask user
def ask_user(prompt: str) -> str:
    return input(prompt)


def enum_to_keys(enum):
    return [e.name for e in enum]

T = TypeVar('T')
def inplace_replace(l:list[T], old:T, new:T):
    """In place replacement of old with new"""
    i = l.index(old)
    l[i] = new