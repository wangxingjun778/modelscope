# Copyright (c) Alibaba, Inc. and its affiliates.
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from modelscope_hub import HubConfig

from modelscope.hub.cache_manager import scan_cache_dir
from modelscope.hub.errors import CacheNotFound
from modelscope.utils.file_utils import get_hub_cache_root
from modelscope.utils.logger import get_logger

logger = get_logger()


class HubScanCacheTest(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'cache'
        snapshot = self.root / 'models/org--repo/snapshots/master'
        snapshot.mkdir(parents=True)
        (snapshot / 'README.md').write_text('fixture readme')
        cfg = HubConfig(
            cache_dir=self.root,
            config_dir=Path(self.temp.name) / 'auth',
            token='')
        patcher = patch('modelscope_hub.config._default_config', cfg)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_scan_default_dir(self):
        """Test scanning the default cache directory."""
        try:
            res_info = scan_cache_dir()
            table_str = res_info.export_as_table()
            lines = table_str.split('\n')
            if len(lines) <= 20:
                logger.info(table_str)
            else:
                limited_table = '\n'.join(
                    lines[:10]
                    + [f'... ({len(lines) - 20} lines omitted) ...']
                    + lines[-10:])
                logger.info(limited_table)
        except Exception as e:
            self.fail(f'Scanning default cache directory failed: {e}')

    def test_scan_given_dir(self):
        """Test scanning a given cache directory."""
        try:
            scan_cache_dir(get_hub_cache_root())
            logger.info('Done')
        except Exception as e:
            self.fail(f'Scanning given cache directory failed: {e}')

    def test_scan_not_exist_dir(self):
        """Test scanning a non-existent cache directory."""
        with self.assertRaises(CacheNotFound):
            scan_cache_dir(Path(self.temp.name) / 'missing')


if __name__ == '__main__':
    unittest.main()
