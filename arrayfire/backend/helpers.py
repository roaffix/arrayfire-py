from typing import Union

from .backend import Backend, BackendPlatform, backend, backend_api
from .c_backend.unsorted import set_backend as c_set_backend


def set_backend(platform: Union[BackendPlatform, str]) -> None:
    current_active_platform = backend_api.platform

    if isinstance(platform, str):
        if platform not in [d.name for d in BackendPlatform]:
            raise ValueError(f"{platform} is not a valid name for backend platform.")
        platform = BackendPlatform[platform]

    if not isinstance(platform, BackendPlatform):
        raise TypeError(f"{platform} is not a valid name for backend platform.")

    if current_active_platform == platform:
        raise RuntimeError(f"{platform} is already the active backend platform.")

    if backend_api.platform == BackendPlatform.unified:
        c_set_backend(platform.value)

    backend_api._load_backend_lib(platform)  # FIXME should not access private API

    if current_active_platform == backend_api.platform:
        raise RuntimeError(f"Could not set {platform} as new backend platform. Consider checking logs.")


def get_backend() -> Backend:
    return backend
