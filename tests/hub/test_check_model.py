# Copyright (c) Alibaba, Inc. and its affiliates.
"""Network-free tests of the local model identity and freshness check."""
import hashlib
import io
import os
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from urllib.parse import parse_qs, urlparse

import json
import requests

from modelscope.hub.check_model import (check_local_model_is_latest,
                                        get_model_id_from_cache)

ENDPOINT = 'https://unit.test'


def put(path: Path, data: bytes = b'payload') -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def tree(root: Path):
    return sorted(str(path.relative_to(root)) for path in root.rglob('*'))


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


def remote_file(path, data):
    return {
        'Path': path,
        'Size': len(data),
        'Sha256': hashlib.sha256(data).hexdigest(),
        'Type': 'blob',
    }


class CheckModelTest(unittest.TestCase):

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
                    'MODELSCOPE_ENDPOINT': ENDPOINT,
                }),
            mock.patch('modelscope_hub.config._default_config', None),
        ]
        for patcher in patchers:
            patcher.start()
            self.addCleanup(patcher.stop)
        for name in ('MS_CACHE_HOME', 'MODELSCOPE_DOMAIN',
                     'MODELSCOPE_PREFER_AI_SITE'):
            os.environ.pop(name, None)
        self.snapshot = self.cache / 'models/org--model/snapshots/master'
        put(self.snapshot / 'config.json', b'config')
        put(self.snapshot / 'sub/weights.bin', b'weights')

    def serve(self, files):
        """Serve revisions and the file listing of ``org/model``."""
        revisions = {
            'Code': 200,
            'Success': True,
            'Data': {
                'RevisionMap': {
                    'Branches': [{
                        'Revision': 'master'
                    }],
                    'Tags': [{
                        'Revision': 'v1.0'
                    }],
                }
            }
        }
        listing = {'Code': 200, 'Success': True, 'Data': {'Files': files}}
        self.urls = []

        def send(request, **kwargs):
            self.urls.append(request.url)
            path = urlparse(request.url).path
            if path == '/api/v1/models/org/model/revisions':
                return json_response(request, revisions)
            if path == '/api/v1/models/org/model/repo/files':
                return json_response(request, listing)
            return json_response(request, {'Code': 404}, 404)

        return mock.patch('requests.sessions.Session.send', side_effect=send)

    def check(self, files, model_root=None):
        with self.serve(files), mock.patch(
                'modelscope.hub.check_model.logger') as logger:
            check_local_model_is_latest(str(model_root or self.snapshot))
        return [call.args[0] for call in logger.info.call_args_list]

    def test_model_id_of_standard_snapshot(self):
        snapshot = put(self.cache / 'models/org--model.v1/snapshots/master'
                       / 'config.json').parent
        before = tree(self.tmp)
        self.assertEqual(
            get_model_id_from_cache(str(snapshot)), 'org/model.v1')
        self.assertEqual(tree(self.tmp), before)

    def test_model_id_of_legacy_cache_is_decoded(self):
        legacy = put(self.cache
                     / 'hub/models/org/model___v1/config.json').parent
        (legacy / '.mdl').write_bytes(pickle.dumps({'id': 'org/model___v1'}))
        self.assertEqual(get_model_id_from_cache(str(legacy)), 'org/model.v1')

    def test_unknown_dir_is_only_read(self):
        local = put(self.tmp / 'local/config.json').parent
        before = tree(self.tmp)
        self.assertEqual(get_model_id_from_cache(str(local)), 'unknown')
        self.assertEqual(tree(self.tmp), before)

    def test_unknown_identity_sends_no_request(self):
        local = put(self.tmp / 'local/config.json').parent
        with mock.patch('requests.sessions.Session.send') as send, \
                mock.patch('modelscope.hub.check_model.HubApi') as api:
            check_local_model_is_latest(str(local))
        send.assert_not_called()
        api.assert_not_called()

    def test_latest_snapshot_is_not_reported(self):
        files = [
            remote_file('config.json', b'config'),
            {
                'Path': 'sub',
                'Type': 'tree'
            },
            remote_file('sub/weights.bin', b'weights'),
        ]
        self.assertEqual(self.check(files), [])
        self.assertEqual(len(self.urls), 2)
        self.assertEqual(
            parse_qs(urlparse(self.urls[1]).query)['Revision'], ['v1.0'])

    def test_changed_or_missing_file_is_reported(self):
        for files in ([remote_file('config.json', b'new config')],
                      [remote_file('tokenizer.json', b'{}')]):
            with self.subTest(path=files[0]['Path']):
                messages = self.check(files)
                self.assertEqual(len(messages), 1)
                self.assertIn(files[0]['Path'], messages[0])

    def test_same_size_content_needs_hash_validation(self):
        files = [remote_file('config.json', b'CONFIG')]
        self.assertEqual(self.check(files), [])
        with mock.patch(
                'modelscope.hub.check_model.enable_default_hash_validation',
                True):
            self.assertEqual(len(self.check(files)), 1)


if __name__ == '__main__':
    unittest.main()
