from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto


@dataclass(unsafe_hash=True)
class State:
    x: int
    y: int

    def up(self) -> State:
        self.x -= 1
        return self

    def down(self) -> State:
        self.x += 1
        return self

    def left(self) -> State:
        self.y -= 1
        return self

    def right(self) -> State:
        self.y += 1
        return self

    def __repr__(self) -> str:
        return f"({self.x},{self.y})"

    def __format__(self, __format_spec: str) -> str:
        return f"{repr(self):{__format_spec}}"


@dataclass
class Action:
    up: int
    right: int
    down: int
    left: int

    def __repr__(self) -> str:
        return f"{self.up:>3} {self.right:>3} {self.down:>3} {self.left:>3}"


class ActionType(Enum):
    UP = auto()
    RIGHT = auto()
    DOWN = auto()
    LEFT = auto()
