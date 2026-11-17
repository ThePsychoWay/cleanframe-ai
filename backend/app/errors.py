"""Plain-language reasons for failures that nobody wrote a message for, shared by every worker job."""

from __future__ import annotations

import errno

RESOURCES_EXHAUSTED = (
    "The server ran out of memory or disk space while working on this video. "
    "Try a shorter clip or a smaller output size, then try again."
)
STORAGE_UNAVAILABLE = "The video file couldn't be read from or saved to storage. Please try again in a moment."
_RESOURCE_ERRNOS = frozenset({errno.ENOSPC, errno.ENOMEM, errno.EDQUOT})


def ran_out_of_resources(error: BaseException | None) -> bool:
    """True for memory or disk exhaustion, including when it's the cause of a more general error."""
    seen = 0
    while error is not None and seen < 8:
        if isinstance(error, MemoryError) or (isinstance(error, OSError) and error.errno in _RESOURCE_ERRNOS):
            return True
        error = error.__cause__ or error.__context__
        seen += 1
    return False
