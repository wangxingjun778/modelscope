"""Snapshot download — shim preserving the legacy positional-arg signature.

Delegates to ``modelscope_hub.compat`` while keeping ``revision``, ``cache_dir``
and friends accessible as positional arguments for backward compatibility.
Arguments added after the legacy ``endpoint`` parameter are keyword-only.
Cache layout, legacy cache reuse, offline lookup and report/recovery modes are
implemented by ``modelscope_hub``; this module only adapts arguments.
"""
from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Type, Union

from modelscope_hub.compat.snapshot_download import \
    dataset_snapshot_download as _compat_dataset_snapshot_download
from modelscope_hub.compat.snapshot_download import \
    snapshot_download as _compat_snapshot_download

if TYPE_CHECKING:
    from modelscope_hub import DownloadRetryPolicy

    from .callback import ProgressCallback

__all__ = ['snapshot_download', 'dataset_snapshot_download']


def _to_str(path: Union[str, Path, None]) -> Optional[str]:
    return str(path) if path is not None else None


def snapshot_download(
    model_id: Optional[str] = None,
    revision: Optional[str] = None,
    cache_dir: Union[str, Path, None] = None,
    user_agent: Optional[Union[Dict, str]] = None,
    local_files_only: Optional[bool] = False,
    cookies=None,
    ignore_file_pattern: Optional[Union[str, List[str]]] = None,
    allow_file_pattern: Optional[Union[str, List[str]]] = None,
    local_dir: Union[str, Path, None] = None,
    allow_patterns: Optional[Union[List[str], str]] = None,
    ignore_patterns: Optional[Union[List[str], str]] = None,
    max_workers: Optional[int] = None,
    repo_id: Optional[str] = None,
    repo_type: Optional[str] = None,
    progress_callbacks: Optional[List[Type[ProgressCallback]]] = None,
    token: Optional[str] = None,
    endpoint: Optional[str] = None,
    *,
    continue_on_error: bool = False,
    tracker_path: Optional[str] = None,
    retry_failed: bool = False,
    retry_policy: Optional[DownloadRetryPolicy] = None,
    reuse_legacy_cache: bool = True,
) -> str:
    """Download a complete repo snapshot and return its local directory.

    Preserves the legacy positional-argument signature for backward
    compatibility while delegating to ``modelscope_hub.compat``.

    Args:
        model_id: The model id; ``repo_id`` takes precedence when both are set.
        revision: Branch, tag or commit. Defaults to ``master``.
        cache_dir: The total cache root. Repositories are stored below it in
            the ``modelscope_hub`` layout; it is not the repository directory.
        local_dir: Download the repository files directly into this directory.
        local_files_only: Return the cached snapshot without network access.
        progress_callbacks: A list of :class:`ProgressCallback` subclasses
            (not instances), each instantiated per file to report progress.
        continue_on_error: Keep downloading after individual file failures and
            persist a resumable report; the output directory is returned even
            when the report is partial.
        tracker_path: Location of the report state file. Requires
            ``continue_on_error=True``.
        retry_failed: Retry only the files recorded as failed by a previous
            report. Requires ``continue_on_error=True``.
        retry_policy: Retry policy of report-mode downloads. Requires
            ``continue_on_error=True``.
        reuse_legacy_cache: Reuse repositories cached by older SDK versions.
            Legacy folders are not isolated per revision; pass ``False`` to use
            revision-specific snapshots only.

    Returns:
        str: The local directory of the snapshot.
    """
    return _compat_snapshot_download(
        model_id=model_id,
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
        cache_dir=_to_str(cache_dir),
        local_dir=_to_str(local_dir),
        allow_file_pattern=allow_file_pattern,
        ignore_file_pattern=ignore_file_pattern,
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
        max_workers=max_workers if max_workers is not None else 4,
        cookies=cookies,
        token=token,
        endpoint=endpoint,
        local_files_only=bool(local_files_only),
        user_agent=user_agent,
        progress_callbacks=progress_callbacks,
        continue_on_error=continue_on_error,
        tracker_path=tracker_path,
        retry_failed=retry_failed,
        retry_policy=retry_policy,
        reuse_legacy_cache=reuse_legacy_cache,
    )


def dataset_snapshot_download(
    dataset_id: Optional[str] = None,
    revision: Optional[str] = None,
    cache_dir: Union[str, Path, None] = None,
    local_dir: Union[str, Path, None] = None,
    allow_file_pattern: Optional[Union[str, List[str]]] = None,
    ignore_file_pattern: Optional[Union[str, List[str]]] = None,
    allow_patterns: Optional[Union[List[str], str]] = None,
    ignore_patterns: Optional[Union[List[str], str]] = None,
    max_workers: Optional[int] = None,
    cookies=None,
    repo_id: Optional[str] = None,
    token: Optional[str] = None,
    endpoint: Optional[str] = None,
    *,
    local_files_only: bool = False,
    user_agent: Optional[Union[Dict, str]] = None,
    progress_callbacks: Optional[List[Type[ProgressCallback]]] = None,
    continue_on_error: bool = False,
    tracker_path: Optional[str] = None,
    retry_failed: bool = False,
    retry_policy: Optional[DownloadRetryPolicy] = None,
    reuse_legacy_cache: bool = True,
) -> str:
    """Download a dataset repo snapshot (legacy positional-arg signature).

    ``dataset_id`` takes precedence over its ``repo_id`` alias. The other
    arguments have the same meaning as in :func:`snapshot_download`.

    Returns:
        str: The local directory of the snapshot.
    """
    return _compat_dataset_snapshot_download(
        dataset_id=dataset_id or repo_id,
        revision=revision,
        cache_dir=_to_str(cache_dir),
        local_dir=_to_str(local_dir),
        allow_file_pattern=allow_file_pattern,
        ignore_file_pattern=ignore_file_pattern,
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
        max_workers=max_workers if max_workers is not None else 4,
        cookies=cookies,
        token=token,
        endpoint=endpoint,
        local_files_only=bool(local_files_only),
        user_agent=user_agent,
        progress_callbacks=progress_callbacks,
        continue_on_error=continue_on_error,
        tracker_path=tracker_path,
        retry_failed=retry_failed,
        retry_policy=retry_policy,
        reuse_legacy_cache=reuse_legacy_cache,
    )
