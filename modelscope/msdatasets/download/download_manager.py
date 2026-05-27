# Copyright (c) Alibaba, Inc. and its affiliates.

from datasets.download.download_manager import DownloadManager
from datasets.download.streaming_download_manager import \
    StreamingDownloadManager
from datasets.utils.file_utils import cached_path

from modelscope.msdatasets.download.download_config import DataDownloadConfig
from modelscope.msdatasets.utils.oss_utils import OssUtilities
from modelscope.utils.file_utils import is_relative_path


class DataDownloadManager(DownloadManager):

    def __init__(self, download_config: DataDownloadConfig):
        super().__init__(
            dataset_name=download_config.dataset_name,
            data_dir=download_config.data_dir,
            download_config=download_config,
            record_checksums=True)

    def _download(self, url_or_filename: str,
                  download_config: DataDownloadConfig) -> str:
        url_or_filename = str(url_or_filename)

        oss_utilities = OssUtilities(
            dataset_name=download_config.dataset_name,
            namespace=download_config.namespace,
            revision=download_config.version)

        if is_relative_path(url_or_filename):
            # fetch oss files
            return oss_utilities.download(
                url_or_filename, download_config=download_config)
        else:
            return cached_path(
                url_or_filename, download_config=download_config)

    def _download_single(self, url_or_filename: str,
                         download_config: DataDownloadConfig) -> str:
        # Note: _download_single function is available for datasets>=2.19.0
        return self._download(url_or_filename, download_config)


class DataStreamingDownloadManager(StreamingDownloadManager):
    """Streaming download manager that returns remote URLs instead of downloading.

    This enables true streaming: HF's StreamingDownloadManager._extract() will
    produce fsspec-compatible chained URLs (e.g. "zip://::{https_url}") that
    support Range Request-based partial reads.
    """

    def __init__(self, download_config: DataDownloadConfig):
        super().__init__(
            dataset_name=download_config.dataset_name,
            data_dir=download_config.data_dir,
            download_config=download_config,
            base_path=download_config.cache_dir)
        self._oss_utilities = None

    @property
    def oss_utilities(self) -> 'OssUtilities':
        """Lazily initialize OssUtilities to avoid unnecessary API calls."""
        if self._oss_utilities is None:
            self._oss_utilities = OssUtilities(
                dataset_name=self.download_config.dataset_name,
                namespace=self.download_config.namespace,
                revision=self.download_config.version)
        return self._oss_utilities

    def _download_single(self, url_or_filename: str) -> str:
        """Return a remote URL for streaming access instead of downloading.

        For relative paths (OSS files), generates a presigned URL that supports
        HTTP Range headers. For absolute URLs, returns them as-is.
        """
        url_or_filename = str(url_or_filename)
        if is_relative_path(url_or_filename):
            return self.oss_utilities.get_signed_url(url_or_filename)
        return url_or_filename
