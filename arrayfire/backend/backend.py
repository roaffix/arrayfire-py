import ctypes
import enum
import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from arrayfire.backend import config
from arrayfire.dtypes import Dtype, float32
from arrayfire.dtypes.helpers import CShape, c_dim_t, to_str
from arrayfire.version import ARRAYFIRE_VER_MAJOR, FORGE_VER_MAJOR

VERBOSE_LOADS = os.environ.get("AF_VERBOSE_LOADS") == "1"


class _ErrorCodes(enum.Enum):
    none = 0


@dataclass
class ArrayBuffer:
    address: int
    length: int = 0


class BackendDevices(enum.Enum):
    unified = 0  # NOTE It is set as Default value on Arrayfire backend
    cpu = 1
    cuda = 2
    opencl = 4


class Backend:
    def __init__(self) -> None:
        self._clibs = {device.name: None for device in BackendDevices}

        more_info_str = "Please look at https://github.com/arrayfire/arrayfire-python/wiki for more information."
        self.setup = config.setup()

        af_module = __import__(__name__)
        self.AF_PYMODULE_PATH = af_module.__path__[0] + "/" if af_module.__path__ else None

        self._name = None

        for libname in self._libnames(config.SupportedLibs.forge, FORGE_VER_MAJOR):
            full_libname = libname[0] + libname[1]
            try:
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
        for device in BackendDevices:
            _name = device.name if device != BackendDevices.unified else ""
            for libname in self._libnames(config.SupportedLibs.arrayfire):
                full_libname = Path(libname[0]) / Path(libname[1])
                try:
                    ctypes.cdll.LoadLibrary(str(full_libname))
                    clib = ctypes.CDLL(str(full_libname))
                    self._clibs[_name] = clib
                    err = clib.af_randu(ctypes.pointer(out), 4, ctypes.pointer(dims.c_array), float32.c_api_value)
                    if err == _ErrorCodes.none.value:
                        self._name = _name
                        clib.af_release_array(out)
                        if VERBOSE_LOADS:
                            print("Loaded " + full_libname)

                        if device == BackendDevices.cuda:
                            nvrtc_name = self._find_nvrtc_builtins_libname(Path(libname[0]))
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

    def _libnames(self, lib: config.SupportedLibs, ver_major: Optional[str] = None) -> List[Tuple[str, str]]:
        post = self.setup.post if ver_major is None else ver_major
        libname = self.setup.pre + lib.value + post

        lib64_path = self.setup.af_path / "lib64"
        search_path = lib64_path if lib64_path.is_dir() else self.setup.af_path / "lib"

        site_path = Path(sys.prefix) / "lib64" if not config.is_arch_x86() else Path(sys.prefix) / "lib"

        # prefer locally packaged arrayfire libraries if they exist
        af_module = __import__(__name__)
        local_path = af_module.__path__[0] + "/" if af_module.__path__ else None

        libpaths = [("", libname), (str(site_path), libname), (str(local_path), libname)]

        if self.setup.af_path:  # prefer specified AF_PATH if exists
            libpaths.append((str(search_path), libname))
        else:
            libpaths.insert(2, (str(search_path), libname))
        return libpaths

    def _find_nvrtc_builtins_libname(self, search_path: Path) -> Optional[str]:
        for f in search_path.iterdir():
            if "nvrtc-builtins" in f.name:
                return f.name
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


# HACK for osx
# backend_api = ctypes.CDLL("/opt/arrayfire//lib/libafcpu.3.dylib")
# HACK for windows
# backend_api = ctypes.CDLL("C:/Program Files/ArrayFire/v3/lib/afcpu.dll")
backend_api = Backend().get()


def safe_call(c_err: int) -> None:
    if c_err == _ErrorCodes.none.value:
        return

    err_str = ctypes.c_char_p(0)
    backend_api.af_get_last_error(ctypes.pointer(err_str), ctypes.pointer(c_dim_t(0)))
    raise RuntimeError(to_str(err_str))
