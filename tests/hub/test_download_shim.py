# Copyright (c) Alibaba, Inc. and its affiliates.
"""Network-free contract tests of the legacy download entry points.

Each legacy entry point of ``modelscope.hub`` must forward every argument
through ``modelscope_hub.compat`` and the facade down to the download manager,
which is the only layer replaced here. Offline lookups must neither send a
request nor construct a client, and must leave the cache tree untouched.
"""
import inspect
import io
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock
from urllib.parse import urlparse

import json
import requests
from modelscope_hub import DownloadRetryPolicy, ProgressCallback
from modelscope_hub._download import DownloadManager
from modelscope_hub.compat.hub_api import LegacyHubApi
from modelscope_hub.errors import InvalidParameter

from modelscope.hub.file_download import (dataset_file_download,
                                          model_file_download)
from modelscope.hub.snapshot_download import (dataset_snapshot_download,
                                              snapshot_download)

ENDPOINT = 'https://unit.test'


def put(path: Path, data: bytes = b'payload') -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def tree(root: Path):
    return sorted(str(path.relative_to(root)) for path in root.rglob('*'))


def bound_args(call, func):
    """Return the arguments of a recorded ``autospec`` call by name."""
    return inspect.signature(func).bind(*call.args, **call.kwargs).arguments


def json_response(request, payload, status=200):
    """Build a real response with byte-stream semantics."""
    content = json.dumps(payload).encode('utf-8')
    response = requests.Response()
    response.status_code = status
    response.url = request.url
    response.request = request
    response.headers['Content-Type'] = 'application/json'
    response.headers['Content-Length'] = str(len(content))
    response._content = content
    response._content_consumed = True
    response.raw = io.BytesIO(content)
    return response


class Callback(ProgressCallback):
    pass


class IsolatedHubTestCase(unittest.TestCase):
    """Use a private cache root, config dir and process default config."""

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.tmp = Path(temp.name).resolve()
        self.cache = self.tmp / 'cache'
        self.auth = self.tmp / 'auth'
        self._start(
            mock.patch.dict(
                os.environ, {
                    'HOME': str(self.tmp / 'home'),
                    'MODELSCOPE_CACHE': str(self.cache),
                    'MODELSCOPE_HOME': str(self.auth),
                    'MODELSCOPE_API_TOKEN': '',
                }))
        for name in ('MS_CACHE_HOME', 'MODELSCOPE_CREDENTIALS_PATH',
                     'MODELSCOPE_ENDPOINT', 'MODELSCOPE_DOMAIN',
                     'MODELSCOPE_PREFER_AI_SITE', 'MODELSCOPE_SDK_DEBUG'):
            os.environ.pop(name, None)
        self._start(mock.patch('modelscope_hub.config._default_config', None))

    def _start(self, patcher):
        result = patcher.start()
        self.addCleanup(patcher.stop)
        return result


class OfflineLookupTest(IsolatedHubTestCase):

    def setUp(self):
        super().setUp()
        self.send = self._start(
            mock.patch(
                'requests.sessions.Session.send',
                side_effect=AssertionError('network')))
        for module in ('snapshot_download', 'file_download'):
            self._start(
                mock.patch(
                    f'modelscope_hub.compat.{module}.HubApi',
                    side_effect=AssertionError('client')))

    def tearDown(self):
        self.send.assert_not_called()
        self.assertFalse(self.auth.exists())

    def test_snapshot_keeps_legacy_positional_args(self):
        expected = put(self.cache
                       / 'models/owner--repo/snapshots/master/config.json')
        before = tree(self.tmp)
        # model_id, revision, cache_dir, user_agent, local_files_only
        path = snapshot_download('owner/repo', None, self.cache, None, True)
        self.assertIsInstance(path, str)
        self.assertEqual(path, str(expected.parent))
        self.assertEqual(
            snapshot_download(repo_id='owner/repo', local_files_only=True),
            str(expected.parent))
        self.assertEqual(
            snapshot_download(
                'owner/repo', 'master', str(self.cache),
                local_files_only=True), str(expected.parent))
        self.assertEqual(tree(self.tmp), before)

    def test_local_dir_accepts_path(self):
        local = put(self.tmp / 'local' / 'config.json').parent
        self.assertEqual(
            snapshot_download(
                'owner/repo', local_dir=local, local_files_only=True),
            str(local))

    def test_dataset_alias_and_positional_order(self):
        expected = put(self.cache
                       / 'datasets/owner--data/snapshots/master/train.csv')
        before = tree(self.tmp)
        # dataset_id, revision, cache_dir
        self.assertEqual(
            dataset_snapshot_download(
                'owner/data', 'master', self.cache, local_files_only=True),
            str(expected.parent))
        self.assertEqual(
            dataset_snapshot_download(
                repo_id='owner/data', local_files_only=True),
            str(expected.parent))
        # dataset_id takes precedence over the repo_id alias
        self.assertEqual(
            dataset_snapshot_download(
                'owner/data', repo_id='owner/other', local_files_only=True),
            str(expected.parent))
        self.assertEqual(tree(self.tmp), before)

    def test_single_file_skips_revision_resolution(self):
        model_file = put(self.cache
                         / 'models/owner--repo/snapshots/master/config.json')
        dataset_file = put(self.cache
                           / 'datasets/owner--data/snapshots/v1/train.csv')
        before = tree(self.tmp)
        with mock.patch.object(
                LegacyHubApi,
                'get_valid_revision_detail',
                side_effect=AssertionError('revision query')):
            self.assertEqual(
                model_file_download(
                    'owner/repo', 'config.json', local_files_only=True),
                str(model_file))
            self.assertEqual(
                dataset_file_download(
                    'owner/data', 'train.csv', local_files_only=True),
                str(dataset_file))
        self.assertEqual(tree(self.tmp), before)


class OnlineForwardingTest(IsolatedHubTestCase):
    """Arguments must reach the download manager through every layer."""

    def setUp(self):
        super().setUp()
        self.send = self._start(
            mock.patch(
                'requests.sessions.Session.send',
                side_effect=AssertionError('network')))

    def tearDown(self):
        self.send.assert_not_called()

    def assert_client(self, manager, token='secret'):
        self.assertEqual(manager._config.endpoint, ENDPOINT)
        self.assertEqual(manager._config.token, token)

    def test_snapshot_options_reach_download_manager(self):
        output = self.tmp / 'output'
        for download, repo_type in ((snapshot_download, 'model'),
                                    (dataset_snapshot_download, 'dataset')):
            with self.subTest(repo_type=repo_type), mock.patch.object(
                    DownloadManager,
                    'download_repo',
                    autospec=True,
                    return_value=output) as call:
                result = download(
                    'owner/repo',
                    'v1',
                    self.tmp / 'root',
                    local_dir=self.tmp / 'local',
                    allow_patterns='*.json',
                    ignore_file_pattern=['*.bin'],
                    max_workers=2,
                    token='secret',
                    endpoint=ENDPOINT,
                    user_agent={'app': 'demo'},
                    progress_callbacks=[Callback],
                    reuse_legacy_cache=False)
                self.assertEqual(result, str(output))
                args = bound_args(call.call_args,
                                  DownloadManager.download_repo)
                self.assert_client(args['self'])
                self.assertEqual(args['repo_id'], 'owner/repo')
                self.assertEqual(args['repo_type'], repo_type)
                self.assertEqual(args['revision'], 'v1')
                self.assertEqual(args['cache_dir'], self.tmp / 'root')
                self.assertEqual(args['local_dir'], self.tmp / 'local')
                self.assertEqual(args['allow_patterns'], ['*.json'])
                self.assertEqual(args['ignore_patterns'], ['*.bin'])
                self.assertEqual(args['max_workers'], 2)
                self.assertEqual(args['user_agent'], {'app': 'demo'})
                self.assertEqual(args['progress_callbacks'], [Callback])
                self.assertIs(args['reuse_legacy_cache'], False)

    def test_snapshot_defaults_reach_download_manager(self):
        with mock.patch.object(
                DownloadManager,
                'download_repo',
                autospec=True,
                return_value=self.tmp) as call:
            snapshot_download('owner/repo', endpoint=ENDPOINT)
        args = bound_args(call.call_args, DownloadManager.download_repo)
        self.assertIsNone(args['revision'])
        self.assertEqual(args['cache_dir'], self.cache)
        self.assertIsNone(args['local_dir'])
        self.assertEqual(args['max_workers'], 4)
        self.assertIsNone(args['progress_callbacks'])
        self.assertIs(args['reuse_legacy_cache'], True)

    def test_report_options_reach_download_manager(self):
        policy = DownloadRetryPolicy(max_retries=1)
        report = mock.Mock(output_dir=str(self.tmp / 'output'))
        tracker = self.tmp / 'state.json'
        for download, repo_type in ((snapshot_download, 'model'),
                                    (dataset_snapshot_download, 'dataset')):
            with self.subTest(repo_type=repo_type), mock.patch.object(
                    DownloadManager,
                    'download_repo_report',
                    autospec=True,
                    return_value=report) as call:
                result = download(
                    'owner/repo',
                    token='secret',
                    endpoint=ENDPOINT,
                    user_agent='app/demo',
                    progress_callbacks=[Callback],
                    continue_on_error=True,
                    tracker_path=str(tracker),
                    retry_failed=True,
                    retry_policy=policy,
                    reuse_legacy_cache=False)
                self.assertEqual(result, report.output_dir)
                args = bound_args(call.call_args,
                                  DownloadManager.download_repo_report)
                self.assert_client(args['self'])
                self.assertEqual(args['repo_type'], repo_type)
                self.assertEqual(args['tracker_path'], tracker)
                self.assertIs(args['retry_failed'], True)
                self.assertIs(args['retry_policy'], policy)
                self.assertEqual(args['user_agent'], 'app/demo')
                self.assertEqual(args['progress_callbacks'], [Callback])
                self.assertIs(args['reuse_legacy_cache'], False)

    def test_recovery_options_require_report_mode(self):
        for download in (snapshot_download, dataset_snapshot_download):
            with self.subTest(download=download.__name__), \
                    self.assertRaises(InvalidParameter):
                download(
                    'owner/repo', endpoint=ENDPOINT, tracker_path='state.json')

    def test_file_options_reach_download_manager(self):
        output = self.tmp / 'file'
        for download, repo_type in ((model_file_download, 'model'),
                                    (dataset_file_download, 'dataset')):
            with self.subTest(repo_type=repo_type), mock.patch.object(
                    DownloadManager,
                    'download_file',
                    autospec=True,
                    return_value=output) as call:
                result = download(
                    'owner/repo',
                    'sub/config.json',
                    revision='v1',
                    cache_dir=str(self.tmp / 'root'),
                    local_dir=str(self.tmp / 'local'),
                    token='secret',
                    endpoint=ENDPOINT,
                    user_agent={'app': 'demo'},
                    progress_callbacks=[Callback],
                    reuse_legacy_cache=False)
                self.assertEqual(result, str(output))
                args = bound_args(call.call_args,
                                  DownloadManager.download_file)
                self.assert_client(args['self'])
                self.assertEqual(args['repo_type'], repo_type)
                self.assertEqual(args['file_path'], 'sub/config.json')
                self.assertEqual(args['revision'], 'v1')
                self.assertEqual(args['cache_dir'], self.tmp / 'root')
                self.assertEqual(args['local_dir'], self.tmp / 'local')
                self.assertEqual(args['user_agent'], {'app': 'demo'})
                self.assertEqual(args['progress_callbacks'], [Callback])
                self.assertIs(args['reuse_legacy_cache'], False)


class ReleaseRevisionTest(IsolatedHubTestCase):
    """Only online model file downloads select the release-time revision."""

    RELEASE = '2026-01-01 00:00:00'

    def setUp(self):
        super().setUp()
        release = int(
            time.mktime(time.strptime(self.RELEASE, '%Y-%m-%d %H:%M:%S')))
        day = 24 * 3600
        revisions = {
            'Code': 200,
            'Success': True,
            'Data': {
                'RevisionMap': {
                    'Branches': [{
                        'Revision': 'master',
                        'CreatedAt': release - 300 * day
                    }],
                    'Tags': [{
                        'Revision': 'v1.0',
                        'CreatedAt': release - day
                    }, {
                        'Revision': 'v2.0',
                        'CreatedAt': release + day
                    }],
                }
            }
        }
        self.requests = []

        def send(request, **kwargs):
            self.requests.append(request.url)
            path = urlparse(request.url).path
            if path == '/api/v1/models/owner/repo/revisions':
                return json_response(request, revisions)
            return json_response(request, {
                'Code': 404,
                'Message': 'not found'
            }, 404)

        self._start(
            mock.patch('requests.sessions.Session.send', side_effect=send))
        self._start(
            mock.patch('modelscope.version.__release_datetime__',
                       self.RELEASE))
        self.download_file = self._start(
            mock.patch.object(
                DownloadManager,
                'download_file',
                autospec=True,
                return_value=self.tmp / 'file'))
        self.download_repo = self._start(
            mock.patch.object(
                DownloadManager,
                'download_repo',
                autospec=True,
                return_value=self.tmp / 'snapshot'))

    def revision(self, mocked, func):
        return bound_args(mocked.call_args, func)['revision']

    def test_release_mode_selects_latest_tag_before_release(self):
        model_file_download('owner/repo', 'config.json', endpoint=ENDPOINT)
        self.assertEqual(
            self.revision(self.download_file, DownloadManager.download_file),
            'v1.0')
        self.assertEqual(len(self.requests), 1)

    def test_debug_mode_skips_revision_query(self):
        with mock.patch.dict(os.environ, {'MODELSCOPE_SDK_DEBUG': '1'}):
            model_file_download('owner/repo', 'config.json', endpoint=ENDPOINT)
        self.assertIsNone(
            self.revision(self.download_file, DownloadManager.download_file))
        self.assertEqual(self.requests, [])

    def test_explicit_revision_skips_revision_query(self):
        model_file_download(
            'owner/repo', 'config.json', 'v2.0', endpoint=ENDPOINT)
        self.assertEqual(
            self.revision(self.download_file, DownloadManager.download_file),
            'v2.0')
        self.assertEqual(self.requests, [])

    def test_missing_repo_keeps_http_error_contract(self):
        with self.assertRaises(requests.exceptions.HTTPError):
            model_file_download(
                'owner/missing', 'config.json', endpoint=ENDPOINT)
        self.download_file.assert_not_called()

    def test_dataset_file_and_snapshot_keep_default_revision(self):
        dataset_file_download('owner/data', 'train.csv', endpoint=ENDPOINT)
        snapshot_download('owner/repo', endpoint=ENDPOINT)
        self.assertIsNone(
            self.revision(self.download_file, DownloadManager.download_file))
        self.assertIsNone(
            self.revision(self.download_repo, DownloadManager.download_repo))
        self.assertEqual(self.requests, [])


if __name__ == '__main__':
    unittest.main()
