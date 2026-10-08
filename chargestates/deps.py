"""Locate collapse2d, the single-particle code of the companion paper (release 1.0.0 of
https://github.com/ehocine/dirac-collapse, doi:10.5281/zenodo.23123320).

Search order: an importable collapse2d; $COLLAPSE2D_ROOT; ./external/dirac-collapse (see the
README); ~/Desktop/dirac-collapse.
"""
import os
import sys

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CANDIDATES = [os.environ.get("COLLAPSE2D_ROOT"), os.path.join(_HERE, "external", "dirac-collapse"),
               os.path.expanduser("~/Desktop/dirac-collapse")]

try:
    import collapse2d  # noqa: F401
except ImportError:
    for _root in _CANDIDATES:
        if _root and os.path.isdir(os.path.join(_root, "collapse2d")):
            sys.path.insert(0, _root)
            break
    try:
        import collapse2d  # noqa: F401,E402
    except ImportError as err:
        raise ImportError(
            "collapse2d not found. Run\n"
            "  git clone --branch 1.0.0 https://github.com/ehocine/dirac-collapse external/dirac-collapse\n"
            "or set COLLAPSE2D_ROOT to a checkout of release 1.0.0.") from err
