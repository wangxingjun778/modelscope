"""File download — delegates Hub and URL downloads to ``modelscope_hub``.

Hub single-file downloads (``model_file_download``, ``dataset_file_download``)
and the legacy URL helpers (``http_get_file``, ``http_get_model_file``,
``get_file_download_url``) keep their legacy signatures here, while transfer,
resume, integrity checks and the cache layout live in ``modelscope_hub.compat``.
This module only supplies the SDK policy inputs: the release timestamp used to
select a model revision, the endpoint selection and the transfer constants of
:mod:`modelscope.hub.constants`.
"""
import os
import time
from http.cookiejar import CookieJar
from typing import Dict, List, Optional, Type

from modelscope_hub.compat.file_download import \
    dataset_file_download as _compat_dataset_file_download
from modelscope_hub.compat.file_download import \
    get_file_download_url as _compat_get_file_download_url
from modelscope_hub.compat.file_download import \
    http_get_file as _compat_http_get_file
from modelscope_hub.compat.file_download import \
    http_get_model_file as _compat_http_get_model_file
from modelscope_hub.compat.file_download import \
    model_file_download as _compat_model_file_download

from modelscope.hub.constants import (API_FILE_DOWNLOAD_CHUNK_SIZE,
                                      API_FILE_DOWNLOAD_RETRY_TIMES,
                                      API_FILE_DOWNLOAD_TIMEOUT,
                                      MODELSCOPE_SDK_DEBUG)
from .callback import ProgressCallback, TqdmCallback  # noqa: F401
from .errors import FileDownloadError  # noqa: F401
from .utils.utils import get_endpoint


def _get_release_timestamp():
    """Compute the release timestamp for revision resolution.

    Returns None (dev-mode) when MODELSCOPE_SDK_DEBUG is set.
    """
    if os.environ.get(MODELSCOPE_SDK_DEBUG):
        return None
    try:
        from modelscope import version
        dt = getattr(version, '__release_datetime__', None)
        if not dt:
            return None
        return int(time.mktime(time.strptime(dt, '%Y-%m-%d %H:%M:%S')))
    except Exception:
        return None


def model_file_download(
    model_id: str,
    file_path: str,
    revision: str = None,
    *,
    cache_dir: str = None,
    local_dir: str = None,
    cookies: dict = None,
    token: str = None,
    endpoint: str = None,
    local_files_only: bool = False,
    user_agent=None,
    progress_callbacks: Optional[List[Type[ProgressCallback]]] = None,
    reuse_legacy_cache: bool = True,
) -> str:
    """Download a single model file and return its local path.

    Without an explicit ``revision``, an online download selects the model
    revision released before this SDK version (dev-mode when
    ``MODELSCOPE_SDK_DEBUG`` is set); offline lookups never query the hub.

    Args:
        progress_callbacks: A list of :class:`ProgressCallback` subclasses
            (not instances), each instantiated to report download progress.
        reuse_legacy_cache: Reuse repositories cached by older SDK versions.
            Legacy folders are not isolated per revision; pass ``False`` to use
            revision-specific snapshots only.
    """
    return _compat_model_file_download(
        model_id,
        file_path,
        revision=revision,
        cache_dir=cache_dir,
        local_dir=local_dir,
        cookies=cookies,
        token=token,
        endpoint=endpoint,
        local_files_only=local_files_only,
        user_agent=user_agent,
        progress_callbacks=progress_callbacks,
        reuse_legacy_cache=reuse_legacy_cache,
        release_timestamp=_get_release_timestamp(),
    )


def dataset_file_download(
    dataset_id: str,
    file_path: str,
    *,
    cache_dir: str = None,
    local_dir: str = None,
    revision: str = None,
    cookies: dict = None,
    token: str = None,
    endpoint: str = None,
    local_files_only: bool = False,
    user_agent=None,
    progress_callbacks: Optional[List[Type[ProgressCallback]]] = None,
    reuse_legacy_cache: bool = True,
) -> str:
    """Download a single dataset file and return its local path.

    ``progress_callbacks`` and ``reuse_legacy_cache`` have the same meaning as
    in :func:`model_file_download`.
    """
    return _compat_dataset_file_download(
        dataset_id,
        file_path,
        cache_dir=cache_dir,
        local_dir=local_dir,
        revision=revision,
        cookies=cookies,
        token=token,
        endpoint=endpoint,
        local_files_only=local_files_only,
        user_agent=user_agent,
        progress_callbacks=progress_callbacks,
        reuse_legacy_cache=reuse_legacy_cache,
    )


# --- Direct HTTP downloads (any URL, no repository semantics) ---


def get_file_download_url(model_id: str,
                          file_path: str,
                          revision: str,
                          endpoint: Optional[str] = None):
    """Format file download url according to `model_id`, `revision` and `file_path`.
    e.g., Given `model_id=john/bert`, `revision=master`, `file_path=README.md`,
    the resulted download url is: https://modelscope.cn/api/v1/models/john/bert/repo?Revision=master&FilePath=README.md

    Args:
        model_id (str): The model_id.
        file_path (str): File path
        revision (str): File revision.
        endpoint (str): The remote endpoint, defaults to the SDK endpoint.

    Returns:
        str: The file url.
    """
    return _compat_get_file_download_url(
        model_id, file_path, revision, endpoint=endpoint or get_endpoint())


def http_get_model_file(
    url: str,
    local_dir: str,
    file_name: str,
    file_size: int,
    cookies: CookieJar,
    headers: Optional[Dict[str, str]] = None,
    disable_tqdm: bool = False,
    progress_callbacks: List[Type[ProgressCallback]] = None,
):
    """Download remote file, will retry 5 times before giving up on errors.

    Args:
        url(str):
            actual download url of the file
        local_dir(str):
            local directory where the downloaded file stores
        file_name(str):
            name of the file stored in `local_dir`
        file_size(int):
            The file size.
        cookies(CookieJar):
            cookies used to authentication the user, which is used for downloading private repos
        headers(Dict[str, str], optional):
            http headers to carry necessary info when requesting the remote file
        disable_tqdm(bool, optional): Disable the progress bar with tqdm.
        progress_callbacks(List[Type[ProgressCallback]], optional):
            progress callbacks to track the download progress.

    Returns:
        str: The sha256 of a file downloaded in one pass, None when the
            download was resumed or retried.

    Raises:
        MaxRetryError: The download kept failing after all retries.
    """
    return _compat_http_get_model_file(
        url,
        local_dir,
        file_name,
        file_size,
        cookies,
        headers=headers,
        disable_tqdm=disable_tqdm,
        progress_callbacks=progress_callbacks,
        retry_times=API_FILE_DOWNLOAD_RETRY_TIMES,
        timeout=API_FILE_DOWNLOAD_TIMEOUT,
        chunk_size=API_FILE_DOWNLOAD_CHUNK_SIZE,
    )


def http_get_file(
    url: str,
    local_dir: str,
    file_name: str,
    cookies: CookieJar,
    headers: Optional[Dict[str, str]] = None,
):
    """Download remote file, will retry 5 times before giving up on errors.

    The file only appears in `local_dir` after the download is complete.

    Args:
        url(str):
            actual download url of the file
        local_dir(str):
            local directory where the downloaded file stores
        file_name(str):
            name of the file stored in `local_dir`
        cookies(CookieJar):
            cookies used to authentication the user, which is used for downloading private repos
        headers(Dict[str, str], optional):
            http headers to carry necessary info when requesting the remote file

    Raises:
        MaxRetryError: The download kept failing after all retries.
    """
    _compat_http_get_file(
        url,
        local_dir,
        file_name,
        cookies,
        headers=headers,
        retry_times=API_FILE_DOWNLOAD_RETRY_TIMES,
        timeout=API_FILE_DOWNLOAD_TIMEOUT,
        chunk_size=API_FILE_DOWNLOAD_CHUNK_SIZE,
    )


__all__ = [
    'model_file_download',
    'dataset_file_download',
    'http_get_file',
    'http_get_model_file',
    'get_file_download_url',
]
