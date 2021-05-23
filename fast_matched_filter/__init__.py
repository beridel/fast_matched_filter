"""
init file for fast_matched_filter library

:copyright:
    William B. Frank and Eric Beauce
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.en.html)
"""

from .fast_matched_filter import (
    matched_filter, test_matched_filter, CPU_LOADED, GPU_LOADED)

del fast_matched_filter

__all__ = [matched_filter, test_matched_filter]

__version__ = '1.4.0'
