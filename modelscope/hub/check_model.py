# Copyright (c) Alibaba, Inc. and its affiliates.

import os
from typing import Dict, Optional, Union
from urllib.parse import urlparse

from modelscope.hub.api import HubApi
from modelscope.hub.cache_manager import identify_cached_repo
from modelscope.hub.git import GitCommandWrapper
from modelscope.hub.utils.caching import enable_default_hash_validation
from modelscope.hub.utils.utils import compute_hash
from modelscope.utils.logger import get_logger

logger = get_logger()


def get_model_id_from_cache(model_root_path: str, ) -> str:
    """Get the model id of a local model directory.

    Git checkouts report their remote; downloaded snapshots report the
    identity recorded in the cache. The cache is only read, never created.

    Returns:
        str: The model id, or 'unknown' when it cannot be identified.
    """
    # download with git
    if os.path.exists(os.path.join(model_root_path, '.git')):
        git_cmd_wrapper = GitCommandWrapper()
        git_url = git_cmd_wrapper.get_repo_remote_url(model_root_path)
        if git_url.endswith('.git'):
            git_url = git_url[:-4]
        u_parse = urlparse(git_url)
        model_id = u_parse.path[1:]
    else:  # snapshot_download
        model_id = identify_cached_repo(model_root_path) or 'unknown'
    return model_id


def _is_same_as_remote(local_path: str, remote_file, verify_hash: bool):
    if not os.path.isfile(local_path):
        return False
    if os.path.getsize(local_path) != remote_file.size:
        return False
    if verify_hash and remote_file.sha256:
        return compute_hash(local_path) == remote_file.sha256
    return True


def check_local_model_is_latest(
    model_root_path: str,
    user_agent: Optional[Union[Dict, str]] = None,
    token: Optional[str] = None,
):
    """Check local model repo is latest.
    Check local model repo is same as hub latest version.

    The local files are compared with the latest hub revision by size, and by
    sha256 for git checkouts or when default hash validation is enabled. This
    check is advisory: a difference is logged and any failure is ignored.
    Directories whose model id cannot be identified are skipped without any
    request; ``user_agent`` is accepted for backward compatibility.
    """
    try:
        model_id = get_model_id_from_cache(model_root_path)
        if model_id == 'unknown':
            return
        _api = HubApi(timeout=20, token=token)
        try:
            _, revisions = _api.get_model_branches_and_tags(model_id=model_id)
            latest_revision = revisions[0] if revisions else 'master'
        except Exception:
            latest_revision = 'master'

        verify_hash = enable_default_hash_validation or os.path.exists(
            os.path.join(model_root_path, '.git'))
        for remote_file in _api.list_repo_files(
                model_id, 'model', revision=latest_revision, recursive=True):
            if remote_file.is_dir:
                continue
            local_path = os.path.join(model_root_path, remote_file.path)
            if not _is_same_as_remote(local_path, remote_file, verify_hash):
                logger.info(
                    f'Model file {remote_file.path} is different from the latest version `{latest_revision}`,'
                    f'This is because you are using an older version or the file is updated manually.'
                )
                break
    except Exception:
        pass  # ignore


def check_model_is_id(model_id: str, token: Optional[str] = None):
    if model_id is None or os.path.exists(model_id):
        return False
    else:
        _api = HubApi()
        _api.login(token)
        try:
            _api.get_model(model_id=model_id, )
            return True
        except Exception:
            return False
