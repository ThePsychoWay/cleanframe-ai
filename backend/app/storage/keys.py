"""Storage keys for files derived from an upload.

Built only from server-generated ids, never from anything the person typed. They're deterministic so
a retry overwrites the previous attempt and deleting a project can remove them even if analysis
stopped halfway.
"""

import uuid


def proxy_key(user_id: uuid.UUID, project_id: uuid.UUID) -> str:
    return f"derived/{user_id}/{project_id}/proxy.mp4"


def thumbnail_key(user_id: uuid.UUID, project_id: uuid.UUID) -> str:
    return f"derived/{user_id}/{project_id}/thumbnail.jpg"


def output_key(user_id: uuid.UUID, project_id: uuid.UUID, job_id: uuid.UUID, container: str) -> str:
    """The restored video of one render job. One key per job, so a failed or cancelled render never overwrites a good result."""
    return f"outputs/{user_id}/{project_id}/{job_id}.{container}"
