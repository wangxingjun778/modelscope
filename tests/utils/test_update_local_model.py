# Copyright (c) Alibaba, Inc. and its affiliates.
"""Network-free tests of ``update_local_model`` used by audio pipelines."""
import inspect
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from modelscope_hub._download import DownloadManager

from modelscope.utils.audio.audio_utils import update_local_model


class UpdateLocalModelTest(unittest.TestCase):

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.tmp = Path(temp.name).resolve()
        self.cache = self.tmp / 'cache'
        patchers = [
            mock.patch.dict(
                os.environ, {
                    'HOME': str(self.tmp / 'home'),
                    'MODELSCOPE_CACHE': str(self.cache),
                    'MODELSCOPE_HOME': str(self.tmp / 'auth'),
                    'MODELSCOPE_API_TOKEN': '',
                    'MODELSCOPE_ENDPOINT': 'https://unit.test',
                }),
            mock.patch('modelscope_hub.config._default_config', None),
        ]
        for patcher in patchers:
            patcher.start()
            self.addCleanup(patcher.stop)
        for name in ('MS_CACHE_HOME', 'MODELSCOPE_DOMAIN'):
            os.environ.pop(name, None)
        patcher = mock.patch(
            'requests.sessions.Session.send',
            side_effect=AssertionError('network'))
        self.send = patcher.start()
        self.addCleanup(patcher.stop)
        patcher = mock.patch.object(
            DownloadManager,
            'download_repo',
            autospec=True,
            return_value=self.tmp / 'downloaded')
        self.download = patcher.start()
        self.addCleanup(patcher.stop)
        self.config = {'model': 'org/model'}
        self.local = self.tmp / 'local'
        self.local.mkdir()
        (self.local / 'configuration.json').write_text('{}')

    def tearDown(self):
        self.send.assert_not_called()

    def download_args(self):
        call = self.download.call_args
        return inspect.signature(DownloadManager.download_repo).bind(
            *call.args, **call.kwargs).arguments

    def test_disabled_update_does_nothing(self):
        self.assertIsNone(
            update_local_model(self.config, str(self.local),
                               {'update_model': False}))
        self.download.assert_not_called()

    def test_explicit_update_refreshes_model_path_in_place(self):
        for update_model, revision in (('latest', None), ('v1.0', 'v1.0')):
            with self.subTest(update_model=update_model):
                result = update_local_model(self.config, str(self.local),
                                            {'update_model': update_model})
                self.assertEqual(result, str(self.tmp / 'downloaded'))
                args = self.download_args()
                self.assertEqual(args['repo_id'], 'org/model')
                self.assertEqual(args['revision'], revision)
                self.assertEqual(args['local_dir'], self.local)

    def test_downloaded_snapshot_is_left_as_is(self):
        snapshot = self.cache / 'models/org--model/snapshots/master'
        snapshot.mkdir(parents=True)
        (snapshot / 'configuration.json').write_text('{}')
        self.assertIsNone(update_local_model(self.config, str(snapshot), {}))
        self.download.assert_not_called()

    def test_external_model_only_syncs_regular_cache(self):
        before = sorted(self.local.rglob('*'))
        update_local_model(self.config, str(self.local), {})
        args = self.download_args()
        self.assertIsNone(args['local_dir'])
        self.assertEqual(args['cache_dir'], self.cache)
        # No bypass symlink or nested ``models/models`` is created.
        self.assertEqual(sorted(self.local.rglob('*')), before)
        self.assertEqual(list(self.cache.rglob('*')), [])

    def test_missing_model_name_or_directory(self):
        self.assertIsNone(update_local_model({}, str(self.local), {}))
        self.assertIsNone(
            update_local_model(self.config, 'org/model',
                               {'update_model': 'latest'}))
        self.download.assert_not_called()


if __name__ == '__main__':
    unittest.main()
