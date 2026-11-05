# Copyright (c) Alibaba, Inc. and its affiliates.
import hashlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import json
import requests
from modelscope_hub import HubConfig
from modelscope_hub.constants import REPO_TREE_PAGE_MAX_ATTEMPTS
from urllib3.exceptions import MaxRetryError

from modelscope.hub.api import HubApi
from modelscope.hub.constants import API_FILE_DOWNLOAD_RETRY_TIMES
from modelscope.hub.errors import ServerError
from modelscope.hub.file_download import (get_file_download_url, http_get_file,
                                          http_get_model_file)


def _response(request, status=200, body=b''):
    """Build a real response with byte-stream semantics for transport fixtures."""
    if isinstance(body, bytes):
        content = body
    else:
        content = json.dumps(body).encode('utf-8')
    response = requests.Response()
    response.status_code = status
    response.url = request.url
    response.request = request
    response.headers['Content-Length'] = str(len(content))
    response._content = content
    response._content_consumed = True
    response.raw = io.BytesIO(content)
    return response


class HubRetryTest(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        cfg = HubConfig(
            cache_dir=Path(self.temp.name) / 'cache',
            config_dir=Path(self.temp.name) / 'auth',
            token='')
        patcher = patch('modelscope_hub.config._default_config', cfg)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.api = HubApi(endpoint='https://modelscope.test', token='')
        self.model_id = 'test/retry-model'

    def test_retry_exception(self):

        def send(request, **kwargs):
            return _response(request, 502, {
                'Code': 502,
                'Message': 'fixture failure'
            })

        with patch('requests.sessions.Session.send', side_effect=send) as transport, \
                patch('modelscope_hub._legacy_api.time.sleep'), \
                self.assertRaises(ServerError) as caught:
            self.api.get_model_files(model_id=self.model_id, recursive=True)
        self.assertEqual(transport.call_count, REPO_TREE_PAGE_MAX_ATTEMPTS)
        self.assertEqual(caught.exception.response.status_code, 502)

    def test_retry_and_success(self):
        statuses = iter([500, 502, 200])
        body = {
            'Code': 200,
            'Success': True,
            'Data': {
                'Files': [{
                    'Path': 'README.md',
                    'Size': 16,
                    'Type': 'blob'
                }]
            }
        }

        def send(request, **kwargs):
            status = next(statuses)
            payload = body if status == 200 else {
                'Code': status,
                'Message': 'fixture failure'
            }
            return _response(request, status, payload)

        with patch('requests.sessions.Session.send', side_effect=send) as transport, \
                patch('modelscope_hub._legacy_api.time.sleep'):
            files = self.api.get_model_files(
                model_id=self.model_id, recursive=True)
        self.assertEqual(files, [{'Path': 'README.md', 'Size': 16}])
        self.assertEqual(transport.call_count, 3)
        self.assertTrue(
            all('/repo/files?' in call.args[0].url
                for call in transport.call_args_list))

    def test_retry_broken_continue(self):
        content = b'0123456789abcdef'
        ranges = []

        def send(request, **kwargs):
            ranges.append(request.headers['Range'])
            if len(ranges) == 1:
                response = _response(request, body=content)

                def chunks(chunk_size=1, **kwargs):
                    yield content[:4]
                    raise requests.ConnectionError(
                        'fixture interrupted stream')

                response.iter_content = chunks
                return response
            response = _response(request, 206, content[4:])
            response.headers['Content-Range'] = 'bytes 4-15/16'
            return response

        with patch('requests.sessions.Session.send', side_effect=send), \
                patch('urllib3.util.retry.Retry.sleep'):
            checksum = http_get_model_file(
                url='https://modelscope.test/download',
                local_dir=self.temp.name,
                file_name='payload.bin',
                file_size=len(content),
                cookies=None,
                disable_tqdm=True)
        self.assertEqual((Path(self.temp.name) / 'payload.bin').read_bytes(),
                         content)
        self.assertEqual(ranges, ['bytes=0-15', 'bytes=4-15'])
        self.assertIsNone(checksum)

    def test_retry_broken_continue_retry_failed(self):

        def send(request, **kwargs):
            return _response(request, 502, b'fixture failure')

        with patch('requests.sessions.Session.send', side_effect=send) as transport, \
                patch('urllib3.util.retry.Retry.sleep'), self.assertRaises(MaxRetryError):
            http_get_model_file(
                url='https://modelscope.test/download',
                local_dir=self.temp.name,
                file_name='payload.bin',
                file_size=16,
                cookies=None,
                disable_tqdm=True)
        self.assertEqual(transport.call_count,
                         API_FILE_DOWNLOAD_RETRY_TIMES + 1)
        self.assertFalse((Path(self.temp.name) / 'payload.bin').exists())

    def test_url_helpers_keep_returns_headers_and_constant_overrides(self):
        content = b'url data'
        chunks = []
        headers = {'user-agent': 'application/example'}

        def send(request, **kwargs):
            self.assertIn('invoked_by/compat', request.headers['User-Agent'])
            self.assertIn('application/example', request.headers['User-Agent'])
            self.assertNotIn('Authorization', request.headers)
            self.assertNotIn('Cookie', request.headers)
            self.assertTrue(all(value <= 7 for value in kwargs['timeout']))
            response = _response(request, body=content)
            response.headers.pop('Content-Length')
            original = response.iter_content

            def iter_content(chunk_size=1, **options):
                chunks.append(chunk_size)
                yield from original(chunk_size=chunk_size, **options)

            response.iter_content = iter_content
            return response

        with patch('requests.sessions.Session.send', side_effect=send), \
                patch('modelscope.hub.file_download.API_FILE_DOWNLOAD_TIMEOUT', 7), \
                patch('modelscope.hub.file_download.API_FILE_DOWNLOAD_CHUNK_SIZE', 3):
            checksum = http_get_model_file(
                'https://external.test/video',
                self.temp.name,
                'known.bin',
                len(content),
                None,
                headers=headers,
                disable_tqdm=True)
            self.assertEqual(checksum, hashlib.sha256(content).hexdigest())
            self.assertIsNone(
                http_get_file(
                    'https://external.test/video',
                    self.temp.name,
                    'atomic.bin',
                    None,
                    headers=headers))
        self.assertEqual(chunks, [3, 3])
        self.assertEqual(headers, {'user-agent': 'application/example'})
        self.assertEqual((Path(self.temp.name) / 'atomic.bin').read_bytes(),
                         content)

    def test_url_retry_budget_is_patchable_at_sdk_path(self):

        def send(request, **kwargs):
            return _response(request, 502, b'fixture failure')

        with patch('requests.sessions.Session.send', side_effect=send) as transport, \
                patch('urllib3.util.retry.Retry.sleep'), \
                patch('modelscope.hub.file_download.API_FILE_DOWNLOAD_RETRY_TIMES', 1), \
                self.assertRaises(MaxRetryError):
            http_get_model_file(
                'https://external.test/video',
                self.temp.name,
                'file.bin',
                8,
                None,
                disable_tqdm=True)
        self.assertEqual(transport.call_count, 2)

    def test_download_url_keeps_sdk_endpoint_and_encoding(self):
        self.assertEqual(
            get_file_download_url(
                'org/model',
                'sub/a b.json',
                'feature/x',
                endpoint='https://endpoint.test'),
            'https://endpoint.test/api/v1/models/org/model/repo?'
            'Revision=feature%2Fx&FilePath=sub%2Fa+b.json')


if __name__ == '__main__':
    unittest.main()
