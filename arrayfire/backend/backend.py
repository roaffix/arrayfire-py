import ctypes
import enum
from dataclasses import dataclass
import os
import platform
import sys
import traceback
from typing import List, Optional, Tuple
import ctypes
from pathlib import Path

from arrayfire import config

from arrayfire.dtypes.helpers import c_dim_t, to_str, CShape
from arrayfire.dtypes import Dtype, float32

# HACK for osx
# backend_api = ctypes.CDLL("/opt/arrayfire//lib/libafcpu.3.dylib")
# HACK for windows
# backend_api = ctypes.CDLL("C:/Program Files/ArrayFire/v3/lib/afcpu.dll")


def safe_call(c_err: int) -> None:
    if c_err == _ErrorCodes.none.value:
        return

    err_str = ctypes.c_char_p(0)
    backend_api.af_get_last_error(ctypes.pointer(err_str), ctypes.pointer(c_dim_t(0)))
    raise RuntimeError(to_str(err_str))


class _ErrorCodes(enum.Enum):
    none = 0


@dataclass
class ArrayBuffer:
    address: int
    length: int = 0


class Backend:
    def __init__(self) -> None:
        self._clibs = {"cuda": None, "opencl": None, "cpu": None, "unified": None}

        self._backend_map = {0: "unified", 1: "cpu", 2: "cuda", 4: "opencl"}

        self._backend_name_map = {"default": 0, "unified": 0, "cpu": 1, "cuda": 2, "opencl": 4}

        more_info_str = "Please look at https://github.com/arrayfire/arrayfire-python/wiki for more information."
        self.setup_obj = config.setup()

        af_module = __import__(__name__)
        self.AF_PYMODULE_PATH = af_module.__path__[0] + "/" if af_module.__path__ else None

        self._name = None

        libnames = reversed(self._libname("forge", head="", ver_major=config.FORGE_VER_MAJOR))
        VERBOSE_LOADS = os.environ.get("AF_VERBOSE_LOADS") == "1"

        for libname in libnames:
            try:
                full_libname = libname[0] + libname[1]
                ctypes.cdll.LoadLibrary(full_libname)
                if VERBOSE_LOADS:
                    print("Loaded " + full_libname)
                break
            except OSError:
                if VERBOSE_LOADS:
                    traceback.print_exc()
                    print("Unable to load " + full_libname)
                pass

        out = ctypes.c_void_p(0)
        dims = CShape(10, 10, 1, 1)
        for name in ("cpu", "opencl", "cuda", ""):
            libnames = reversed(self._libname(name))
            for libname in libnames:
                try:
                    full_libname = Path(libname[0]) / Path(libname[1])
                    ctypes.cdll.LoadLibrary(str(full_libname))
                    _name = "unified" if name == "" else name
                    clib = ctypes.CDLL(str(full_libname))
                    self._clibs[_name] = clib
                    err = clib.af_randu(ctypes.pointer(out), 4, ctypes.pointer(dims.c_array), float32.c_api_value)
                    if err == _ErrorCodes.none.value:
                        self._name = _name
                        clib.af_release_array(out)
                        if VERBOSE_LOADS:
                            print("Loaded " + full_libname)

                        if name == "cuda":
                            nvrtc_name = self._find_nvrtc_builtins_libname(libname[0])
                            if nvrtc_name:
                                ctypes.cdll.LoadLibrary(libname[0] + nvrtc_name)
                                if VERBOSE_LOADS:
                                    print("Loaded " + libname[0] + nvrtc_name)
                            else:
                                if VERBOSE_LOADS:
                                    print("Could not find local nvrtc-builtins library")
                        break
                except OSError:
                    if VERBOSE_LOADS:
                        traceback.print_exc()
                        print("Unable to load " + full_libname)
                    pass

        # if self._name is None:
        #     raise RuntimeError("Could not load any ArrayFire libraries.\n" + more_info_str)

    def _libname(self, name, head="af", ver_major=config.AF_VER_MAJOR) -> List[str]:
        post = self.setup_obj.post.replace(config._VER_MAJOR_PLACEHOLDER, ver_major)
        libname = self.setup_obj.pre + head + name + post

        if self.setup_obj.af_path:
            if (self.setup_obj.af_path / "lib64").is_dir():
                path_search = self.setup_obj.af_path / "lib64/"
            else:
                path_search = self.setup_obj.af_path / "lib/"
        else:
            if (self.setup_obj.af_path / "lib64").is_dir():
                path_search = self.setup_obj.af_path / "lib64/"
            else:
                path_search = self.setup_obj.af_path / "lib/"

        if platform.architecture()[0][:2] == "64":
            path_site = sys.prefix + "/lib64/"
        else:
            path_site = sys.prefix + "/lib/"

        path_local = self.AF_PYMODULE_PATH
        libpaths = [("", libname), (str(path_site), libname), (str(path_local), libname)]
        if self.setup_obj.af_path:  # prefer specified AF_PATH if exists
            libpaths.append((str(path_search), libname))
        else:
            libpaths.insert(2, (str(path_search), libname))
        return libpaths

    def _find_nvrtc_builtins_libname(self, search_path):
        filelist = os.listdir(search_path)
        for f in filelist:
            if "nvrtc-builtins" in f:
                return f
        return None

    def set_unsafe(self, name: str) -> None:
        lib = self._clibs.get(name)
        if lib is None:
            raise RuntimeError("Backend not found")
        self._name = name

    def get_id(self, name: str) -> int:
        return self._backend_name_map[name]

    def get_name(self, bk_id: int) -> str:
        return self._backend_map.get(bk_id, "unknown")

    def get(self):
        return self._clibs.get(self._name)

    def name(self) -> str:
        return self._name

    def is_unified(self) -> bool:
        return self._name == "unified"

    def parse(self, res: int) -> Tuple[str, ...]:
        lst = []
        for key, value in self._backend_name_map.items():
            if value & res:
                lst.append(key)
        return tuple(lst)

backend_api = Backend().get()
