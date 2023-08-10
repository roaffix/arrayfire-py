import ctypes
import enum
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from arrayfire.backend import config
from arrayfire.dtypes.helpers import c_dim_t, to_str
from arrayfire.logger import logger

VERBOSE_LOADS = os.environ.get("AF_VERBOSE_LOADS") == "1"


class _ErrorCodes(enum.Enum):
    none = 0


@dataclass
class ArrayBuffer:
    address: int
    length: int = 0


class BackendDevices(enum.Enum):
    unified = 0  # NOTE It is set as Default value on Arrayfire backend
    cuda = 2
    opencl = 4
    cpu = 1  # NOTE It comes last because we want to keep this order on backend initialization


class Backend:
    device: BackendDevices
    library: ctypes.CDLL

    def __init__(self) -> None:
        self._setup = config.setup()

        self._load_forge_lib()
        self._load_backend_libs()

    def _load_forge_lib(self) -> None:
        for libname in self._libnames("forge", config.SupportedLibPrefixes.forge):
            try:
                ctypes.cdll.LoadLibrary(str(libname))
                logger.info(f"Loaded {libname}")
                break
            except OSError:
                logger.warning(f"Unable to load {libname}")
                pass

    def _load_backend_libs(self) -> None:
        for device in BackendDevices:
            self._load_backend_lib(device)

            if self.device:
                logger.info(f"Setting {device.name} as backend.")
                break

        if not self.device and not self.library:
            raise RuntimeError(
                "Could not load any ArrayFire libraries.\n"
                "Please look at https://github.com/arrayfire/arrayfire-python/wiki for more information."
            )

    def _load_backend_lib(self, device: BackendDevices) -> None:
        # NOTE we still set unified cdll to it's original name later, even if the path search is different
        name = device.name if device != BackendDevices.unified else ""

        for libname in self._libnames(name, config.SupportedLibPrefixes.arrayfire):
            try:
                ctypes.cdll.LoadLibrary(str(libname))
                self.device = device
                self.library = ctypes.CDLL(str(libname))

                if device == BackendDevices.cuda:
                    self._load_nvrtc_builtins_lib(libname.parent)

                logger.info(f"Loaded {libname}")
                break
            except OSError:
                logger.warning(f"Unable to load {libname}")
                pass

    def _load_nvrtc_builtins_lib(self, lib_path: Path) -> None:
        nvrtc_name = self._find_nvrtc_builtins_libname(lib_path)
        if nvrtc_name:
            ctypes.cdll.LoadLibrary(str(lib_path / nvrtc_name))
            logger.info(f"Loaded {lib_path / nvrtc_name}")
        else:
            logger.warning("Could not find local nvrtc-builtins library")

    def _libnames(self, name: str, lib: config.SupportedLibPrefixes, ver_major: Optional[str] = None) -> List[Path]:
        post = self._setup.post if ver_major is None else ver_major
        libname = self._setup.pre + lib.value + name + post

        lib64_path = self._setup.af_path / "lib64"
        search_path = lib64_path if lib64_path.is_dir() else self._setup.af_path / "lib"

        site_path = Path(sys.prefix) / "lib64" if not config.is_arch_x86() else Path(sys.prefix) / "lib"

        # prefer locally packaged arrayfire libraries if they exist
        af_module = __import__(__name__)
        local_path = Path(af_module.__path__[0]) if af_module.__path__ else Path("")

        libpaths = [Path("", libname), site_path / libname, local_path / libname]

        if self._setup.af_path:  # prefer specified AF_PATH if exists
            return [search_path / libname] + libpaths
        else:
            libpaths.insert(2, Path(str(search_path), libname))
            return libpaths

    def _find_nvrtc_builtins_libname(self, search_path: Path) -> Optional[str]:
        for f in search_path.iterdir():
            if "nvrtc-builtins" in f.name:
                return f.name
        return None


# HACK for osx
# backend_api = ctypes.CDLL("/opt/arrayfire//lib/libafcpu.3.dylib")
# HACK for windows
# backend_api = ctypes.CDLL("C:/Program Files/ArrayFire/v3/lib/afcpu.dll")
backend_api = Backend().library


def safe_call(c_err: int) -> None:
    if c_err == _ErrorCodes.none.value:
        return

    err_str = ctypes.c_char_p(0)
    backend_api.af_get_last_error(ctypes.pointer(err_str), ctypes.pointer(c_dim_t(0)))
    raise RuntimeError(to_str(err_str))
