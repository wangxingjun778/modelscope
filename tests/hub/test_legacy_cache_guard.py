# Copyright (c) Alibaba, Inc. and its affiliates.
"""Guards that caches written by older SDK versions stay reusable.

``modelscope_hub`` owns every historical cache layout; the legacy entry
points must reuse them without probing hub internals and must keep the
``reuse_legacy_cache=False`` opt-out on the way down to the transfer.
"""
import os
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from modelscope_hub._download import DownloadManager
from modelscope_hub.errors import CacheNotFound

from modelscope.hub.file_download import (dataset_file_download,
                                          model_file_download)
from modelscope.hub.snapshot_download import (dataset_snapshot_download,
                                              snapshot_download)

ENDPOINT = 'https://unit.test'
LAYOUTS = ('{kind}s/org/repo___v1', 'hub/{kind}s/org/repo___v1', 'org/repo.v1',
           'hub/org/repo.v1')


class LegacyCacheGuardTest(unittest.TestCase):

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.tmp = Path(temp.name).resolve()
        patchers = [
            mock.patch.dict(
                os.environ, {
                    'HOME': str(self.tmp / 'home'),
                    'MODELSCOPE_CACHE': str(self.tmp / 'cache'),
                    'MODELSCOPE_HOME': str(self.tmp / 'auth'),
                    'MODELSCOPE_API_TOKEN': '',
                }),
            mock.patch('modelscope_hub.config._default_config', None),
        ]
        for patcher in patchers:
            patcher.start()
            self.addCleanup(patcher.stop)
        for name in ('MS_CACHE_HOME', 'MODELSCOPE_ENDPOINT',
                     'MODELSCOPE_DOMAIN', 'MODELSCOPE_SDK_DEBUG'):
            os.environ.pop(name, None)
        patcher = mock.patch(
            'requests.sessions.Session.send',
            side_effect=AssertionError('network'))
        self.send = patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self):
        self.send.assert_not_called()

    @staticmethod
    def legacy(path: Path, repo_id: str) -> Path:
        """Create a legacy cache folder with its primitive pickle identity."""
        path.mkdir(parents=True)
        (path / 'config.json').write_bytes(b'payload')
        (path / '.mdl').write_bytes(
            pickle.dumps({'id': repo_id.replace('.', '___')}))
        return path

    def test_offline_lookups_reuse_every_legacy_layout(self):
        downloads = {
            'model': (snapshot_download, model_file_download),
            'dataset': (dataset_snapshot_download, dataset_file_download),
        }
        for kind, (snapshot, single_file) in downloads.items():
            for index, layout in enumerate(LAYOUTS):
                with self.subTest(kind=kind, layout=layout):
                    root = self.tmp / f'{kind}-{index}'
                    old = self.legacy(root / layout.format(kind=kind),
                                      'org/repo.v1')
                    self.assertEqual(
                        snapshot(
                            'org/repo.v1',
                            cache_dir=root,
                            local_files_only=True), str(old))
                    self.assertEqual(
                        single_file(
                            'org/repo.v1',
                            'config.json',
                            cache_dir=str(root),
                            local_files_only=True), str(old / 'config.json'))
                    with self.assertRaises(CacheNotFound):
                        snapshot(
                            'org/repo.v1',
                            cache_dir=root,
                            local_files_only=True,
                            reuse_legacy_cache=False)
                    with self.assertRaises(CacheNotFound):
                        single_file(
                            'org/repo.v1',
                            'config.json',
                            cache_dir=str(root),
                            local_files_only=True,
                            reuse_legacy_cache=False)

    def test_online_download_reuses_legacy_file(self):
        old = self.legacy(self.tmp / 'cache/hub/models/org/repo', 'org/repo')
        with self.assertWarnsRegex(UserWarning, 'not verified'):
            path = model_file_download(
                'org/repo', 'config.json', 'v1', endpoint=ENDPOINT)
        self.assertEqual(path, str(old / 'config.json'))

    def test_online_opt_out_downloads_revision_snapshot(self):
        old = self.legacy(self.tmp / 'cache/hub/models/org/repo', 'org/repo')

        def transfer(self, repo_id, repo_type, file_path, revision, target,
                     **kwargs):
            target.write_bytes(b'fresh')

        with mock.patch.object(
                DownloadManager,
                '_download_with_resume',
                autospec=True,
                side_effect=transfer):
            path = model_file_download(
                'org/repo',
                'config.json',
                'v1',
                endpoint=ENDPOINT,
                reuse_legacy_cache=False)
        expected = (
            self.tmp / 'cache/models/org--repo/snapshots/v1/config.json')
        self.assertEqual(path, str(expected))
        self.assertEqual(expected.read_bytes(), b'fresh')
        self.assertEqual((old / 'config.json').read_bytes(), b'payload')


if __name__ == '__main__':
    unittest.main()
