__all__ = [
    "is_image_io_available",
    "read_array",
    "save_array",
    "load_image",
    "load_image_native",
    "load_image_memory",
    "delete_image_memory",
    "save_image",
    "save_image_native",
    "save_image_memory",
]

from pathlib import Path
from typing import cast

import arrayfire_wrapper.lib as wrapper
from arrayfire_wrapper.lib import is_image_io_available

from arrayfire import Array
from arrayfire.array_object import afarray_as_array
from arrayfire.library.constants import ImageFormat


@afarray_as_array
def read_array(filename: str | Path, index: None | int = None, key: None | str = None) -> Array:
    if not index and not key:
        raise ValueError("Can not read without specified index or key argument.")

    if index:
        return cast(Array, wrapper.read_array_index(str(filename), index))

    return cast(Array, wrapper.read_array_key(str(filename), key))  # type: ignore[arg-type]


def save_array(array: Array, filename: str | Path, key: str, /, *, to_append: bool = False) -> int:
    return wrapper.save_array(key, array.arr, str(filename), to_append)


@afarray_as_array
def load_image(filename: str | Path, /, *, is_color: bool = False) -> Array:
    return cast(Array, wrapper.load_image(str(filename), is_color))


@afarray_as_array
def load_image_native(filename: str | Path) -> Array:
    return cast(Array, wrapper.load_image_native(str(filename)))


@afarray_as_array
def load_image_memory(pointer: int) -> Array:
    return cast(Array, wrapper.load_image_memory(pointer))


def delete_image_memory(image: Array) -> None:
    return wrapper.delete_image_memory(image.arr)


def save_image(image: Array, filename: str | Path, /) -> None:
    wrapper.save_image(image.arr, str(filename))
    return None


def save_image_native(image: Array, filename: str | Path, /) -> None:
    wrapper.save_image_native(image.arr, str(filename))
    return None


def save_image_memory(image: Array, pointer: int, image_format: ImageFormat) -> None:
    wrapper.save_image_memory(pointer, image.arr, image_format)
    return None
