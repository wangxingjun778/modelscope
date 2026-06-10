# Copyright (c) Alibaba, Inc. and its affiliates.
"""Hub constants — delegates to modelscope_hub where possible.

Constants that have equivalents in modelscope_hub are imported from there.
Constants unique to this project (enum classes, legacy aliases) are retained locally.
"""
import os
from pathlib import Path

# --- Delegated constants (from modelscope_hub.compat.constants) ---
from modelscope_hub.compat.constants import (  # noqa: F401
    DEFAULT_DATASET_REVISION, DEFAULT_MAX_WORKERS, FILE_HASH,
    MODELSCOPE_DOMAIN, MODELSCOPE_PREFER_AI_SITE, REPO_TYPE_DATASET,
    REPO_TYPE_MODEL, REPO_TYPE_STUDIO, REPO_TYPE_SUPPORT,
    TEMPORARY_FOLDER_NAME, ModelVisibility_INTERNAL, ModelVisibility_PRIVATE,
    ModelVisibility_PUBLIC)

# Endpoint
# Upload
# Download
# --- Delegated constants (from modelscope_hub.constants) ---
# Network
from modelscope_hub.constants import (  # noqa: F401; noqa: F401; noqa: F401; noqa: F401
    API_CONNECT_TIMEOUT, API_MAX_RETRIES, API_TIMEOUT, DEFAULT_ENDPOINT,
    DEFAULT_INTL_ENDPOINT, DOWNLOAD_CHUNK_SIZE, DOWNLOAD_PARALLEL_THRESHOLD,
    DOWNLOAD_PARALLELS, DOWNLOAD_PART_SIZE, DOWNLOAD_RETRY_TIMES,
    DOWNLOAD_TIMEOUT, ENV_FILE_LOCK, ENV_INTRA_CLOUD_ACCELERATION,
    ENV_INTRA_CLOUD_REGION, UPLOAD_ADAPTIVE_BATCH_SIZE,
    UPLOAD_BLOB_CONNECT_TIMEOUT, UPLOAD_BLOB_MAX_RETRIES,
    UPLOAD_BLOB_READ_TIMEOUT, UPLOAD_BLOB_RETRY_BACKOFF,
    UPLOAD_BLOB_RETRY_MAX_WAIT, UPLOAD_BLOB_TQDM_DISABLE_THRESHOLD,
    UPLOAD_COMMIT_BATCH_SIZE, UPLOAD_FAILED_FILE_MAX_RETRIES,
    UPLOAD_LFS_ENFORCE_THRESHOLD, UPLOAD_LFS_THRESHOLD, UPLOAD_MAX_FILE_COUNT,
    UPLOAD_MAX_FILE_COUNT_IN_DIR, UPLOAD_MAX_FILE_SIZE,
    UPLOAD_NORMAL_FILE_SIZE_TOTAL_LIMIT, UPLOAD_REACT_BACKOFF_MAX_EXPONENT,
    UPLOAD_REACT_ENABLED, UPLOAD_REACT_MAX_DELAY,
    UPLOAD_REACT_ROUND2_BASE_DELAY, UPLOAD_REACT_ROUND3_FILE_DELAY,
    UPLOAD_RETRY_ALLOWED_METHODS, UPLOAD_USE_CACHE,
    UPLOAD_VALIDATE_BLOB_BATCH_SIZE)

# --- Legacy aliases (map old names to new modelscope_hub constants) ---
API_HTTP_CLIENT_TIMEOUT = API_TIMEOUT
API_HTTP_CLIENT_CONNECT_TIMEOUT = API_CONNECT_TIMEOUT
API_HTTP_CLIENT_MAX_RETRIES = API_MAX_RETRIES
API_FILE_DOWNLOAD_RETRY_TIMES = DOWNLOAD_RETRY_TIMES
API_FILE_DOWNLOAD_TIMEOUT = DOWNLOAD_TIMEOUT
API_FILE_DOWNLOAD_CHUNK_SIZE = DOWNLOAD_CHUNK_SIZE
MODELSCOPE_PARALLEL_DOWNLOAD_THRESHOLD_MB = DOWNLOAD_PARALLEL_THRESHOLD // (
    1024 * 1024)
MODELSCOPE_DOWNLOAD_PARALLELS = DOWNLOAD_PARALLELS
UPLOAD_BLOB_TIMEOUT = (UPLOAD_BLOB_CONNECT_TIMEOUT, UPLOAD_BLOB_READ_TIMEOUT)
UPLOAD_SIZE_THRESHOLD_TO_ENFORCE_LFS = UPLOAD_LFS_ENFORCE_THRESHOLD

# --- Local constants (not in modelscope_hub) ---
MODELSCOPE_URL_SCHEME = 'https://'
DEFAULT_MODELSCOPE_DOMAIN = 'www.modelscope.cn'
DEFAULT_MODELSCOPE_INTL_DOMAIN = 'www.modelscope.ai'
DEFAULT_MODELSCOPE_DATA_ENDPOINT = MODELSCOPE_URL_SCHEME + DEFAULT_MODELSCOPE_DOMAIN
DEFAULT_MODELSCOPE_INTL_DATA_ENDPOINT = MODELSCOPE_URL_SCHEME + DEFAULT_MODELSCOPE_INTL_DOMAIN
DEFAULT_MODELSCOPE_GROUP = 'damo'
MODEL_ID_SEPARATOR = '/'
LOGGER_NAME = 'ModelScopeHub'
DEFAULT_CREDENTIALS_PATH = Path.home().joinpath('.modelscope', 'credentials')
MODELSCOPE_CREDENTIALS_PATH = os.environ.get(
    'MODELSCOPE_CREDENTIALS_PATH', DEFAULT_CREDENTIALS_PATH.as_posix())
REQUESTS_API_HTTP_METHOD = ['get', 'head', 'post', 'put', 'patch', 'delete']

CREATE_TAG_MAX_RETRIES = int(
    os.environ.get('MODELSCOPE_CREATE_TAG_MAX_RETRIES', 3))
CREATE_TAG_RETRY_BACKOFF = int(
    os.environ.get('MODELSCOPE_CREATE_TAG_RETRY_BACKOFF', 2))

API_RESPONSE_FIELD_DATA = 'Data'
API_RESPONSE_FIELD_GIT_ACCESS_TOKEN = 'AccessToken'
API_RESPONSE_FIELD_USERNAME = 'Username'
API_RESPONSE_FIELD_EMAIL = 'Email'
API_RESPONSE_FIELD_MESSAGE = 'Message'
MODELSCOPE_CLOUD_ENVIRONMENT = 'MODELSCOPE_ENVIRONMENT'
MODELSCOPE_CLOUD_USERNAME = 'MODELSCOPE_USERNAME'
MODELSCOPE_SDK_DEBUG = 'MODELSCOPE_SDK_DEBUG'
MODELSCOPE_ENABLE_DEFAULT_HASH_VALIDATION = 'MODELSCOPE_ENABLE_DEFAULT_HASH_VALIDATION'
ONE_YEAR_SECONDS = 24 * 365 * 60 * 60
MODELSCOPE_REQUEST_ID = 'X-Request-ID'
DEFAULT_SKILLS_DIR = os.path.join(os.path.expanduser('~'), '.agents', 'skills')


MODELSCOPE_ASCII = r"""
 _   .-')                _ .-') _     ('-.             .-')                              _ (`-.    ('-.
( '.( OO )_             ( (  OO) )  _(  OO)           ( OO ).                           ( (OO  ) _(  OO)
 ,--.   ,--.).-'),-----. \     .'_ (,------.,--.     (_)---\_)   .-----.  .-'),-----.  _.`     \(,------.
 |   `.'   |( OO'  .-.  ',`'--..._) |  .---'|  |.-') /    _ |   '  .--./ ( OO'  .-.  '(__...--'' |  .---'
 |         |/   |  | |  ||  |  \  ' |  |    |  | OO )\  :` `.   |  |('-. /   |  | |  | |  /  | | |  |
 |  |'.'|  |\_) |  |\|  ||  |   ' |(|  '--. |  |`-' | '..`''.) /_) |OO  )\_) |  |\|  | |  |_.' |(|  '--.
 |  |   |  |  \ |  | |  ||  |   / : |  .--'(|  '---.'.-._)   \ ||  |`-'|   \ |  | |  | |  .___.' |  .--'
 |  |   |  |   `'  '-'  '|  '--'  / |  `---.|      | \       /(_'  '--'\    `'  '-'  ' |  |      |  `---.
 `--'   `--'     `-----' `-------'  `------'`------'  `-----'    `-----'      `-----'  `--'      `------'
"""# noqa


class Licenses(object):
    APACHE_V2 = 'Apache License 2.0'
    GPL_V2 = 'GPL-2.0'
    GPL_V3 = 'GPL-3.0'
    LGPL_V2_1 = 'LGPL-2.1'
    LGPL_V3 = 'LGPL-3.0'
    AFL_V3 = 'AFL-3.0'
    ECL_V2 = 'ECL-2.0'
    MIT = 'MIT'

    @classmethod
    def to_list(cls):
        return [
            cls.APACHE_V2,
            cls.GPL_V2,
            cls.GPL_V3,
            cls.LGPL_V2_1,
            cls.LGPL_V3,
            cls.AFL_V3,
            cls.ECL_V2,
            cls.MIT,
        ]


class ModelVisibility(object):
    PRIVATE = 1
    INTERNAL = 3
    PUBLIC = 5


class DatasetVisibility(object):
    PRIVATE = 1
    INTERNAL = 3
    PUBLIC = 5


class Visibility(object):
    PRIVATE = 'private'
    INTERNAL = 'internal'
    PUBLIC = 'public'


class GatedMode(object):
    """Gated mode for private repositories.

    Only effective when Visibility is PRIVATE.
    API payload key: ``ProtectedMode``.
    Values: True = gated (application-based download),
            False = off (normal private).
    """
    GATED = True
    OFF = False


VisibilityMap = {
    ModelVisibility.PRIVATE: Visibility.PRIVATE,
    ModelVisibility.INTERNAL: Visibility.INTERNAL,
    ModelVisibility.PUBLIC: Visibility.PUBLIC
}


class SortKey(object):
    DEFAULT = 'default'
    DOWNLOADS = 'downloads'
    LIKES = 'likes'
    LAST_MODIFIED = 'last_modified'


VALID_SORT_KEYS = {
    SortKey.DEFAULT,
    SortKey.DOWNLOADS,
    SortKey.LIKES,
    SortKey.LAST_MODIFIED,
}
