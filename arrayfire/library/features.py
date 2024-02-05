from __future__ import annotations

from typing import cast

import arrayfire_wrapper.lib as wrapper

from arrayfire import Array
from arrayfire.array_object import afarray_as_array


class Features:
    def __init__(self, max_features: int | None = None) -> None:
        self._pointer = (
            wrapper.create_features(max_features) if max_features else wrapper.AFFeatures.create_null_pointer()
        )

    def __del__(self) -> None:
        if self._pointer.value == 0:
            return

        wrapper.release_features(self._pointer)

    @classmethod
    def from_affeatures(cls, features: wrapper.AFFeatures) -> Features:
        out = cls()
        out._pointer = features
        return out

    @property
    @afarray_as_array
    def x(self) -> Array:
        return cast(Array, wrapper.get_features_xpos(self._pointer))

    @property
    @afarray_as_array
    def y(self) -> Array:
        return cast(Array, wrapper.get_features_ypos(self._pointer))

    @property
    def num_features(self) -> int:
        return wrapper.get_features_num(self._pointer)

    @property
    @afarray_as_array
    def score(self) -> Array:
        return cast(Array, wrapper.get_features_score(self._pointer))

    @property
    @afarray_as_array
    def size(self) -> Array:
        return cast(Array, wrapper.get_features_size(self._pointer))
