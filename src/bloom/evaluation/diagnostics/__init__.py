"""Read-only classifiers and biological summaries of saved prediction exports.

The producer definitions are retained from the supplied downstream tools.
See docs/diagnostics.md for input contracts and the distinction between these
programme scores and the final figure engine's observed-reference scores.
"""

from __future__ import annotations

import sys


def read_direct_reporting(handle):
    """Read a legacy foundation export's verified axis from the packaged backend."""
    from bloom.evaluation.export import repository_root

    backend = str(repository_root() / "third_party" / "downstream")
    sys.path.insert(0, backend)
    try:
        from v4lib.foundation_reporting import read_direct_reporting as reader

        return reader(handle)
    finally:
        sys.path.pop(0)
