__all__ = ["BackendPlatform"]

import ctypes
import enum
import sys
from enum import Enum
from pathlib import Path
from typing import Iterator, List, Optional

from arrayfire.logger import logger
from arrayfire.platform import get_platform_config, is_arch_x86


class _LibPrefixes(Enum):
    forge = ""
    arrayfire = "af"


class BackendPlatform(enum.Enum):
    unified = 0  # NOTE It is set as Default value on Arrayfire backend
    cpu = 1
    cuda = 2
    opencl = 4

    def __iter__(self) -> Iterator:
        # NOTE cpu comes last because we want to keep this order priorty during backend initialization
        return iter((self.unified, self.cuda, self.opencl, self.cpu))


class Backend:
    platform: BackendPlatform
    library: ctypes.CDLL

    def __init__(self) -> None:
        self._platform_config = get_platform_config()

        self._load_forge_lib()
        self._load_backend_libs()

    def _load_forge_lib(self) -> None:
        for libname in self._libnames("forge", _LibPrefixes.forge):
            try:
                ctypes.cdll.LoadLibrary(str(libname))
                logger.info(f"Loaded {libname}")
                break
            except OSError:
                logger.warning(f"Unable to load {libname}")
                pass

    def _load_backend_libs(self) -> None:
        for platform in BackendPlatform:
            self._load_backend_lib(platform)

            if self.platform:
                logger.info(f"Setting {platform.name} as backend.")
                break

        if not self.platform and not self.library:
            raise RuntimeError(
                "Could not load any ArrayFire libraries.\n"
                "Please look at https://github.com/arrayfire/arrayfire-python/wiki for more information."
            )

    def _load_backend_lib(self, platform: BackendPlatform) -> None:
        # NOTE we still set unified cdll to it's original name later, even if the path search is different
        name = platform.name if platform != BackendPlatform.unified else ""

        for libname in self._libnames(name, _LibPrefixes.arrayfire):
            try:
                ctypes.cdll.LoadLibrary(str(libname))
                self.platform = platform
                self.library = ctypes.CDLL(str(libname))

                if platform == BackendPlatform.cuda:
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

    def _libnames(self, name: str, lib: _LibPrefixes, ver_major: Optional[str] = None) -> List[Path]:
        post = self._platform_config.lib_postfix if ver_major is None else ver_major
        libname = self._platform_config.lib_prefix + lib.value + name + post

        lib64_path = self._platform_config.af_path / "lib64"
        search_path = lib64_path if lib64_path.is_dir() else self._platform_config.af_path / "lib"

        site_path = Path(sys.prefix) / "lib64" if not is_arch_x86() else Path(sys.prefix) / "lib"

        # prefer locally packaged arrayfire libraries if they exist
        af_module = __import__(__name__)
        local_path = Path(af_module.__path__[0]) if af_module.__path__ else Path("")

        libpaths = [Path("", libname), site_path / libname, local_path / libname]

        if self._platform_config.af_path:  # prefer specified AF_PATH if exists
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
backend = Backend()
backend_api = backend.library
