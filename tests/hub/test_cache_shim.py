# Copyright (c) Alibaba, Inc. and its affiliates.
"""Read-only cache, configuration and import contracts of the SDK shim."""
import ast
import importlib
import os
import pickle
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest import mock

import json
from modelscope_hub import HubConfig, get_cache_root, set_default_config
from modelscope_hub.compat import cache_manager as compat_cache
from modelscope_hub.compat import caching as compat_caching
from tests.hub.test_download_shim import IsolatedHubTestCase, put, tree

from modelscope.hub import cache_manager
from modelscope.hub.check_model import get_model_id_from_cache
from modelscope.hub.errors import (CacheNotFound, FileDownloadError,
                                   NetworkError)
from modelscope.hub.snapshot_download import snapshot_download
from modelscope.utils import file_utils


class CacheShimTest(IsolatedHubTestCase):

    def test_exports_preserve_type_identity(self):
        for name in cache_manager.__all__:
            self.assertIs(
                getattr(cache_manager, name), getattr(compat_cache, name))
        caching = importlib.import_module('modelscope.hub.utils.caching')
        for name in ('FileSystemCache', 'ModelFileSystemCache',
                     'enable_default_hash_validation'):
            self.assertIs(
                getattr(caching, name), getattr(compat_caching, name))
        self.assertEqual(caching.FILE_HASH, 'Sha256')

    def test_frozen_scan_and_queries_do_not_write(self):
        target = put(self.cache / 'models/org--repo/snapshots/master/file.txt')
        legacy = put(self.cache / 'hub/models/org/legacy/file.txt').parent
        (legacy / '.mdl').write_bytes(pickle.dumps({'id': 'org/legacy'}))
        # A malformed old index is diagnosed, not renamed or reconstructed.
        (legacy / '.msc').write_bytes(b'invalid pickle')
        before = tree(self.tmp)
        with mock.patch(
                'requests.Session', side_effect=AssertionError('session')):
            report = cache_manager.scan_cache_dir(self.cache)
            self.assertIsInstance(report.repos, frozenset)
            self.assertEqual(len(report.repos), 1)
            repo = next(iter(report.repos))
            self.assertEqual(repo.repo_id, 'org/repo')
            self.assertIsInstance(repo.revisions, frozenset)
            self.assertTrue(report.warnings)
            with self.assertRaises(FrozenInstanceError):
                repo.repo_id = 'other/repo'
            self.assertIn('org/repo', report.export_as_table())
            self.assertEqual(
                cache_manager.resolve_cached_snapshot(
                    'org/repo', cache_dir=self.cache), target.parent)
            self.assertEqual(
                cache_manager.resolve_cached_file(
                    'org/repo', 'file.txt', cache_dir=self.cache), target)
            self.assertEqual(
                get_model_id_from_cache(str(target.parent)), 'org/repo')
        self.assertEqual(tree(self.tmp), before)
        self.assertFalse(self.auth.exists())

    def test_precise_identity_preserves_encoding_and_slash_revision(self):
        repo_id = 'org--team/model___v1.0'
        repo = self.cache / 'models' / repo_id.replace('/', '--')
        target = put(repo / 'snapshots/feature/x/config.json')
        (repo / '.ms_repo.json').write_text(
            json.dumps({
                'schema': 1,
                'repo_id': repo_id,
                'repo_type': 'model',
                'revisions': ['feature/x'],
            }))
        self.assertEqual(get_model_id_from_cache(str(target.parent)), repo_id)
        self.assertEqual(
            cache_manager.resolve_cached_snapshot(
                repo_id, revision='feature/x', cache_dir=self.cache),
            target.parent)
        revisions = next(iter(cache_manager.scan_cache_dir(
            self.cache).repos)).revisions
        self.assertEqual([entry.commit_hash for entry in revisions],
                         ['feature/x'])

    def test_ambiguous_revisions_and_partial_cache(self):
        for revision in ('v1', 'v2'):
            put(self.cache / f'models/org--repo/snapshots/{revision}/file.txt')
        with self.assertRaisesRegex(CacheNotFound, 'ambiguous'):
            snapshot_download('org/repo', local_files_only=True)
        self.assertTrue(
            Path(
                snapshot_download(
                    'org/repo', revision='v1',
                    local_files_only=True)).is_dir())
        put(self.cache
            / 'models/org--partial/snapshots/master/file.incomplete')
        with self.assertRaises(CacheNotFound):
            snapshot_download('org/partial', local_files_only=True)

    def test_cache_helpers_keep_legacy_values_and_active_root(self):
        self.assertEqual(file_utils.get_modelscope_cache_dir(),
                         str(self.cache))
        self.assertEqual(file_utils.get_model_cache_root(),
                         str(self.cache / 'models'))
        self.assertEqual(file_utils.get_dataset_cache_root(),
                         str(self.cache / 'datasets'))
        self.assertEqual(
            file_utils.get_model_cache_dir('org/repo'),
            str(self.cache / 'models/org/repo') + '/')
        self.assertEqual(
            file_utils.get_dataset_cache_dir('org/data'),
            str(self.cache / 'datasets/org/data') + '/')
        cfg = HubConfig(
            cache_dir=self.tmp / 'active', config_dir=self.auth, token='')
        set_default_config(cfg)
        before = tree(self.tmp)
        self.assertEqual(file_utils.get_hub_cache_root(), str(cfg.cache_dir))
        self.assertEqual(get_cache_root(), cfg.cache_dir)
        # Importing the SDK cannot replace an already initialized config.
        hub = importlib.import_module('modelscope.hub')
        with mock.patch.dict(os.environ,
                             {'MS_CACHE_HOME': str(self.tmp / 'alias')}):
            importlib.reload(hub)
            hub._sync_config()
        self.assertEqual(file_utils.get_hub_cache_root(), str(cfg.cache_dir))
        self.assertEqual(tree(self.tmp), before)

    def test_default_and_deprecated_environment_paths(self):
        os.environ.pop('MODELSCOPE_CACHE')
        os.environ.pop('MODELSCOPE_HOME')
        self.assertEqual(file_utils.get_modelscope_cache_dir(),
                         str(self.tmp / 'home/.cache/modelscope/hub'))
        self.assertEqual(file_utils.get_hub_cache_root(),
                         str(self.tmp / 'home/.cache/modelscope'))
        with mock.patch.dict(
                os.environ, {
                    'MS_CACHE_HOME':
                    str(self.tmp / 'alias-cache'),
                    'MODELSCOPE_CREDENTIALS_PATH':
                    str(self.tmp / 'alias-home/credentials'),
                }):
            cfg = HubConfig(token='')
            self.assertEqual(cfg.cache_dir, self.tmp / 'alias-cache')
            self.assertEqual(cfg.config_dir, self.tmp / 'alias-home')
            os.environ['MODELSCOPE_HOME'] = str(self.auth)
            self.assertEqual(HubConfig(token='').config_dir, self.auth)

    def test_error_alias_keeps_both_catch_contracts(self):
        from modelscope_hub.errors import FileDownloadError as CoreError
        generic = FileDownloadError('failed')
        self.assertIsInstance(generic, NetworkError)
        self.assertTrue(generic.retryable)
        core = CoreError(
            'partial',
            failed_files={'file': NetworkError('failed')},
            total=2,
            output_dir='/output')
        wrapped = FileDownloadError(core.message, source=core)
        self.assertIsInstance(wrapped, CoreError)
        self.assertIsInstance(wrapped, NetworkError)
        self.assertEqual(wrapped.failed_files, core.failed_files)
        self.assertEqual(wrapped.total, 2)
        self.assertEqual(wrapped.output_dir, '/output')
        self.assertEqual(wrapped.retryable, core.retryable)

    def test_hf_config_consumes_real_offline_snapshot(self):
        from modelscope import AutoConfig

        snapshot = self.cache / 'models/org--bert/snapshots/master'
        put(snapshot / 'config.json',
            b'{"model_type": "bert", "hidden_size": 8}')
        with mock.patch(
                'requests.Session', side_effect=AssertionError('session')):
            config = AutoConfig.from_pretrained(
                'org/bert', cache_dir=str(self.cache), local_files_only=True)
        self.assertEqual(config.model_type, 'bert')
        self.assertEqual(config.hidden_size, 8)
        self.assertEqual(Path(config.name_or_path), snapshot)

    def test_all_hub_modules_and_source_import_contracts(self):
        source_root = Path(__file__).resolve().parents[2] / 'modelscope'
        checked = set()
        with mock.patch(
                'requests.sessions.Session.send',
                side_effect=AssertionError('network')):
            for source in (source_root / 'hub').rglob('*.py'):
                relative = source.relative_to(
                    source_root.parent).with_suffix('')
                module_name = '.'.join(
                    relative.parts).removesuffix('.__init__')
                importlib.import_module(module_name)
            # Check imports across ALL SDK source, without importing AI models
            # that need optional native dependencies or perform downloads.
            for source in source_root.rglob('*.py'):
                parsed = ast.parse(source.read_text(encoding='utf-8'))
                for node in ast.walk(parsed):
                    if not isinstance(
                            node,
                            ast.ImportFrom) or node.level or not node.module:
                        continue
                    is_hub_import = node.module.startswith(
                        ('modelscope.hub', 'modelscope_hub'))
                    if not is_hub_import:
                        continue
                    for alias in node.names:
                        if alias.name == '*' or (node.module,
                                                 alias.name) in checked:
                            continue
                        with self.subTest(
                                source=str(source), symbol=alias.name):
                            module = importlib.import_module(node.module)
                            if not hasattr(module, alias.name):
                                importlib.import_module(
                                    f'{node.module}.{alias.name}')
                        checked.add((node.module, alias.name))
        self.assertGreater(len(checked), 40)


if __name__ == '__main__':
    unittest.main()
