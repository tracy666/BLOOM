"""Paper-aligned biological figure generation from saved evaluation outputs.

The public CLI groups the retained analysis engine by Results 2.2--2.8.
No model training or inference runs during figure rendering.
"""

from .groups import GROUPS

__all__ = ["GROUPS"]
