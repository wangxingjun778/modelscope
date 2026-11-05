# Copyright (c) Alibaba, Inc. and its affiliates.
"""Legacy file cache classes, used internally only.

Re-exported from ``modelscope_hub.compat.caching`` to keep the historical
import path. Downloads and cache scans no longer use these writable classes;
their destructive methods only act on identified repositories beneath the
active cache root.
"""
from modelscope_hub.compat.caching import (FileSystemCache,
                                           ModelFileSystemCache,
                                           enable_default_hash_validation)

from modelscope.hub.constants import (  # noqa: F401
    FILE_HASH, MODELSCOPE_ENABLE_DEFAULT_HASH_VALIDATION)
