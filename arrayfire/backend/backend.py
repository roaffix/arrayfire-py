import ctypes
import enum
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from arrayfire.backend import config
from arrayfire.dtypes.helpers import c_dim_t, to_str
from arrayfire.logger import logger
from arrayfire.version import FORGE_VER_MAJOR

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
        self._clibs: Dict[str, Optional[ctypes.CDLL]] = {device.name: None for device in BackendDevices}
        self.setup = config.setup()

        self._name: Optional[str] = None

        self.load_forge_lib()
        self.load_backend_libs()
        print(self._clibs)

    def load_forge_lib(self) -> None:
        for libname in self._libnames("forge", config.SupportedLibs.forge, FORGE_VER_MAJOR):
            full_libname = libname[0] + libname[1]
            try:
                ctypes.cdll.LoadLibrary(full_libname)
                logger.info(f"Loaded {full_libname}")
                break
            except OSError:
                logger.warning(f"Unable to load {full_libname}")
                pass

    def load_backend_libs(self) -> None:
        for device in BackendDevices:
            self.load_backend_lib(device)

    def load_backend_lib(self, device: BackendDevices) -> None:
        # NOTE we still set unified cdll to it's original name later, even if the path search is different
        name = device.name if device != BackendDevices.unified else ""

        for libname in self._libnames(name):
            full_libname = Path(libname[0]) / Path(libname[1])
            try:
                ctypes.cdll.LoadLibrary(str(full_libname))
                self._clibs[device.name] = ctypes.CDLL(str(full_libname))

                if device == BackendDevices.cuda:
                    self.load_nvrtc_builtins_lib(libname[0])

                logger.info(f"Loaded {full_libname}")
                break
            except OSError:
                logger.warning(f"Unable to load {full_libname}")
                pass

    def load_nvrtc_builtins_lib(self, lib_path: str) -> None:
        nvrtc_name = self._find_nvrtc_builtins_libname(Path(lib_path))
        if nvrtc_name:
            ctypes.cdll.LoadLibrary(lib_path + nvrtc_name)
            logger.info("Loaded " + lib_path + nvrtc_name)
        else:
            logger.warning("Could not find local nvrtc-builtins library")

        if all(value is None for value in self._clibs.values()):
            raise RuntimeError(
                "Could not load any ArrayFire libraries.\n"
                "Please look at https://github.com/arrayfire/arrayfire-python/wiki for more information.")

    def _libnames(
        self, name: str, lib: config.SupportedLibs = config.SupportedLibs.arrayfire, ver_major: Optional[str] = None
    ) -> List[Tuple[str, str]]:
        post = self.setup.post if ver_major is None else ver_major
        libname = self.setup.pre + lib.value + name + post

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
        return self._clibs.get("cpu")  # FIXME: should be self._name

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
