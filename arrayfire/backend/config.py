import ctypes
import os
import platform
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

from arrayfire.version import ARRAYFIRE_VER_MAJOR, FORGE_VER_MAJOR


class SupportedLibs(Enum):
    forge = "forge"
    arrayfire = "af"


class SupportedPlatforms(Enum):
    windows = "Windows"
    darwin = "Darwin"  # OSX
    linux = "Linux"

    @classmethod
    def is_cygwin(cls, name: str) -> bool:
        return "cyg" in name.lower()


def is_arch_x86() -> bool:
    machine = platform.machine()
    return platform.architecture()[0][0:2] == "32" and (machine[-2:] == "86" or machine[0:3] == "arm")


@dataclass
class Setup:
    pre: str
    post: str
    af_path: Path
    cuda_found: bool

    def __iter__(self) -> Iterator:
        return iter((self.pre, self.post, self.af_path, self.af_path, self.cuda_found))


def setup() -> Setup:
    platform_name = platform.system()
    cuda_found = False

    try:
        af_path = Path(os.environ["AF_PATH"])
    except KeyError:
        af_path = None

    try:
        cuda_path = Path(os.environ["CUDA_PATH"])
    except KeyError:
        cuda_path = None

    if platform_name == SupportedPlatforms.windows.value or SupportedPlatforms.is_cygwin(platform_name):
        if platform_name == SupportedPlatforms.windows.value:
            # HACK Supressing crashes caused by missing dlls
            # http://stackoverflow.com/questions/8347266/missing-dll-print-message-instead-of-launching-a-popup
            # https://msdn.microsoft.com/en-us/library/windows/desktop/ms680621.aspx
            ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002)  # type: ignore[attr-defined]

        if not af_path:
            af_path = _find_default_path(f"C:/Program Files/ArrayFire/v{ARRAYFIRE_VER_MAJOR}")

        if cuda_path and (cuda_path / "bin").is_dir() and (cuda_path / "nvvm/bin").is_dir():
            cuda_found = True

        return Setup("", ".dll", af_path, cuda_found)

    if platform_name == SupportedPlatforms.darwin.value:
        default_cuda_path = Path("/usr/local/cuda/")

        if not af_path:
            af_path = _find_default_path("/opt/arrayfire", "/usr/local")

        if not (cuda_path and default_cuda_path.exists()):
            cuda_found = (default_cuda_path / "lib").is_dir() and (default_cuda_path / "/nvvm/lib").is_dir()

        return Setup("lib", f".{ARRAYFIRE_VER_MAJOR}.dylib", af_path, cuda_found)

    if platform_name == SupportedPlatforms.linux.value:
        default_cuda_path = Path("/usr/local/cuda/")

        if not af_path:
            af_path = _find_default_path(f"/opt/arrayfire-{ARRAYFIRE_VER_MAJOR}", "/opt/arrayfire/", "/usr/local/")

        if not (cuda_path and default_cuda_path.exists()):
            if "64" in platform.architecture()[0]:  # Check either is 64 bit arch is selected
                cuda_found = (default_cuda_path / "lib64").is_dir() and (default_cuda_path / "nvvm/lib64").is_dir()
            else:
                cuda_found = (default_cuda_path / "lib").is_dir() and (default_cuda_path / "nvvm/lib").is_dir()

        return Setup("lib", f".so.{ARRAYFIRE_VER_MAJOR}", af_path, cuda_found)

    raise OSError(f"{platform_name} is not supported.")


def _find_default_path(*args: str) -> Path:
    for path in args:
        default_path = Path(path)
        if default_path.exists():
            return default_path
    raise ValueError("None of specified default paths were found.")
