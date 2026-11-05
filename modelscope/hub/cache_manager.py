"""Utilities to inspect and clear the ModelScope cache.

Re-exported from ``modelscope_hub.compat.cache_manager``, which owns the cache
layout discovery:

- :func:`scan_cache_dir` keeps the legacy read-only, file-grain report built
  from the frozen ``Cached*Info`` classes; :func:`scan_cache` is the
  repo-grain report of ``modelscope_hub``.
- :func:`resolve_cached_snapshot` / :func:`resolve_cached_file` locate the
  actual local snapshot (or file) of a repository revision without network
  access, and :func:`identify_cached_repo` reports the repository id of a
  cached directory. Never derive these paths from a cache root.
- :func:`plan_cache_clear` returns an inspectable deletion plan limited to
  identified repositories, which :func:`execute_cache_clear` re-validates
  before deleting; :func:`clear_cache` does both in one call.
"""

from modelscope_hub.compat.cache_manager import *  # noqa: F401,F403

__all__ = [
    'CachedFileInfo',
    'CachedRevisionInfo',
    'CachedRepoInfo',
    'ModelScopeCacheInfo',
    'scan_cache_dir',
    'scan_cache',
    'clear_cache',
    'get_cache_root',
    'resolve_cached_snapshot',
    'resolve_cached_file',
    'identify_cached_repo',
    'plan_cache_clear',
    'execute_cache_clear',
]
