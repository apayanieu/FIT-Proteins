try:
    from importlib.metadata import version as _v, PackageNotFoundError  # py3.8+
except Exception:  # pragma: no cover
    _v = None
    class PackageNotFoundError(Exception): ...
try:
    __version__ = _v("fit-proteins")
except PackageNotFoundError:
    __version__ = "0.0.0"
