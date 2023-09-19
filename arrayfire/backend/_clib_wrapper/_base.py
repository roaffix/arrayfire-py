from __future__ import annotations

import ctypes

# AFArrayType = ctypes.c_void_p


class AFArrayType(ctypes.c_void_p):
    @classmethod
    def create_pointer(cls) -> AFArrayType:
        cls.value = 0
        return cls()
