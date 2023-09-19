import ctypes

from arrayfire.backend._backend import _backend
from arrayfire.dtypes import CShape, Dtype

from ._base import AFArrayType
from ._error_handler import safe_call

AFRandomEngine = ctypes.c_void_p


def create_random_engine(engine_type: int, seed: int, /) -> AFRandomEngine:
    out = AFArrayType.create_pointer()
    safe_call(_backend.clib.af_create_random_engine(ctypes.pointer(out), engine_type, ctypes.c_longlong(seed)))
    return out


def release_random_engine(engine: AFRandomEngine, /) -> None:
    safe_call(_backend.clib.af_release_random_engine(engine))
    return None


def random_engine_set_type(engine: AFRandomEngine, engine_type: int, /) -> None:
    safe_call(_backend.clib.af_random_engine_set_type(ctypes.pointer(engine), engine_type))
    return None


def random_engine_get_type(engine: AFRandomEngine, /) -> int:
    out = ctypes.c_int(0)
    safe_call(_backend.clib.af_random_engine_get_type(ctypes.pointer(out), engine))
    return out.value


def random_engine_set_seed(engine: AFRandomEngine, seed: int, /) -> None:
    safe_call(_backend.clib.af_random_engine_set_seed(ctypes.pointer(engine), ctypes.c_longlong(seed)))
    return None


def random_engine_get_seed(engine: AFRandomEngine, /) -> int:
    out = ctypes.c_longlong(0)
    safe_call(_backend.clib.af_random_engine_get_seed(ctypes.pointer(out), engine))
    return out.value


def randu(shape: tuple[int, ...], dtype: Dtype, /) -> AFArrayType:
    """
    source: https://arrayfire.org/docs/group__random__func__randu.htm#ga412e2c2f5135bdda218c3487c487d3b5
    """
    out = AFArrayType.create_pointer()
    c_shape = CShape(*shape)
    safe_call(_backend.clib.af_randu(ctypes.pointer(out), 4, c_shape.c_array, dtype.c_api_value))
    return out


def random_uniform(shape: tuple[int, ...], dtype: Dtype, engine: AFRandomEngine, /) -> AFArrayType:
    """
    source: https://arrayfire.org/docs/group__random__func__randu.htm#ga2ca76d970cfac076f9006755582a4a4c
    """
    out = AFArrayType.create_pointer()
    c_shape = CShape(*shape)
    safe_call(_backend.clib.af_random_uniform(ctypes.pointer(out), 4, c_shape.c_array, dtype.c_api_value, engine))
    return out
