from dataclasses import dataclass


@dataclass(frozen=True)
class ArrayBuffer:
    address: int
    length: int = 0
