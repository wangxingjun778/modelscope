"""modelscope.hub — shim layer delegating to modelscope_hub."""

from .agent_idp_api import AgentIdpApi
from .callback import ProgressCallback, TqdmCallback
from .commit_scheduler import CommitScheduler
from .snapshot_download import snapshot_download


def _sync_config() -> None:
    """Kept as a no-op for backward compatibility.

    ``modelscope_hub.HubConfig`` reads the legacy environment variables
    itself: ``MS_CACHE_HOME`` as a deprecated alias of ``MODELSCOPE_CACHE``,
    and ``MODELSCOPE_CREDENTIALS_PATH`` as a deprecated alias of
    ``MODELSCOPE_HOME`` (the parent directory is used when it points at the
    ``credentials`` directory or a file inside it). An initialized default
    config is never modified here.
    """


__all__ = [
    'AgentIdpApi',
    'CommitScheduler',
    'ProgressCallback',
    'TqdmCallback',
    'snapshot_download',
]
