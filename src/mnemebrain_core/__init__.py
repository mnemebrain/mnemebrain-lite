"""MnemeBrain Lite — The belief layer for AI agents."""

try:
    from importlib.metadata import version

    __version__ = version("mnemebrain-lite")
except Exception:
    __version__ = "0.1.0a3"
