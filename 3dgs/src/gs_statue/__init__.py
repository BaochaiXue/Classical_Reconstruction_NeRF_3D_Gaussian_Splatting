"""Pipeline helpers for statue-to-3DGS reconstruction."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("gs-statue")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = ["__version__"]
