"""MnemeBrain Lite — The belief layer for AI agents."""

try:
    from importlib.metadata import PackageNotFoundError, version

    __version__ = version("mnemebrain-lite")
except (ImportError, PackageNotFoundError):
    __version__ = "0.1.0a4"
