# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from modelscope_hub import get_cache_root

from modelscope.hub.utils.utils import (_modelscope_hub_cache_root,
                                        find_reusable_legacy_repo_dir)


class LegacyCacheReuseTest(unittest.TestCase):
    """Old flat/hub cache layouts should be reusable without re-download.

    A legacy directory only yields to a standard ``snapshots`` structure of
    the same repository, not to an arbitrary non-empty directory.
    """

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.cache = Path(self._tmpdir.name).resolve()
        self.model_id = 'iic/nlp_xlmr_named-entity-recognition_eng-ecommerce-query'
        self.owner, self.name = self.model_id.split('/', 1)
        patcher = mock.patch('modelscope_hub.config._default_config', None)
        patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self):
        self._tmpdir.cleanup()

    def _touch_model_dir(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        (path / 'configuration.json').write_text('{}', encoding='utf-8')
        return path

    def _touch_snapshot(self, repo_id: str, revision: str = 'master'):
        return self._touch_model_dir(self.cache / 'models'
                                     / repo_id.replace('/', '--') / 'snapshots'
                                     / revision)

    def test_finds_flat_legacy_cache(self):
        legacy = self._touch_model_dir(self.cache / self.owner / self.name)
        found = find_reusable_legacy_repo_dir(
            self.model_id, cache_dir=self.cache)
        self.assertEqual(found, str(legacy))

    def test_finds_hub_legacy_cache(self):
        legacy = self._touch_model_dir(self.cache / 'hub' / self.owner
                                       / self.name)
        found = find_reusable_legacy_repo_dir(
            self.model_id, cache_dir=self.cache)
        self.assertEqual(found, str(legacy))

    def test_finds_dataset_flat_legacy_cache(self):
        legacy = self._touch_model_dir(self.cache / 'org' / 'data')
        found = find_reusable_legacy_repo_dir(
            'org/data', repo_type='dataset', cache_dir=self.cache)
        self.assertEqual(found, str(legacy))

    def test_reuses_unsafed_models_slash_layout(self):
        # Hub only probes safe_name (dots -> ___); unsafed path is reusable.
        dotted_id = 'org/model.with.dots'
        owner, name = dotted_id.split('/', 1)
        slash = self._touch_model_dir(self.cache / 'models' / owner / name)
        found = find_reusable_legacy_repo_dir(dotted_id, cache_dir=self.cache)
        self.assertEqual(found, str(slash))

    def test_prefers_hub_known_safe_slash_layout(self):
        dotted_id = 'org/model.with.dots'
        owner, name = dotted_id.split('/', 1)
        self._touch_model_dir(self.cache / 'models' / owner
                              / name.replace('.', '___'))
        self._touch_model_dir(self.cache / owner / name)
        found = find_reusable_legacy_repo_dir(dotted_id, cache_dir=self.cache)
        self.assertIsNone(found)

    def test_prefers_standard_snapshot_layout(self):
        self._touch_snapshot(self.model_id)
        self._touch_model_dir(self.cache / self.owner / self.name)
        found = find_reusable_legacy_repo_dir(
            self.model_id, cache_dir=self.cache)
        self.assertIsNone(found)

    def test_non_snapshot_standard_dir_does_not_hide_legacy(self):
        # Files directly below ``models/owner--name`` are no snapshot.
        self._touch_model_dir(self.cache / 'models'
                              / self.model_id.replace('/', '--'))
        legacy = self._touch_model_dir(self.cache / self.owner / self.name)
        found = find_reusable_legacy_repo_dir(
            self.model_id, cache_dir=self.cache)
        self.assertEqual(found, str(legacy))

    def test_empty_or_metadata_only_legacy_dir_ignored(self):
        legacy = self.cache / self.owner / self.name
        legacy.mkdir(parents=True)
        self.assertIsNone(
            find_reusable_legacy_repo_dir(self.model_id, cache_dir=self.cache))
        (legacy / '.mdl').write_bytes(pickle.dumps({'id': self.model_id}))
        (legacy / 'model.bin.incomplete').write_bytes(b'partial')
        self.assertIsNone(
            find_reusable_legacy_repo_dir(self.model_id, cache_dir=self.cache))

    def test_invalid_repo_id_returns_none(self):
        for repo_id in ('', None, 'no-owner'):
            with self.subTest(repo_id=repo_id):
                self.assertIsNone(
                    find_reusable_legacy_repo_dir(
                        repo_id, cache_dir=self.cache))

    def test_uses_modelscope_cache_env(self):
        legacy = self._touch_model_dir(self.cache / self.owner / self.name)
        with mock.patch.dict(os.environ,
                             {'MODELSCOPE_CACHE': str(self.cache)}):
            found = find_reusable_legacy_repo_dir(self.model_id)
        self.assertEqual(found, str(legacy))

    def test_default_root_matches_hub_not_sdk_hub_suffix(self):
        # Without MODELSCOPE_CACHE, hub uses ~/.cache/modelscope (no /hub).
        env = {
            k: v
            for k, v in os.environ.items()
            if k not in ('MODELSCOPE_CACHE', 'MS_CACHE_HOME')
        }
        env['HOME'] = str(self.cache / 'home')
        with mock.patch.dict(os.environ, env, clear=True):
            root = _modelscope_hub_cache_root()
            self.assertEqual(root, get_cache_root())
            self.assertEqual(root,
                             self.cache / 'home' / '.cache' / 'modelscope')
            legacy = self._touch_model_dir(root / 'hub' / self.owner
                                           / self.name)
            self.assertEqual(
                find_reusable_legacy_repo_dir(self.model_id), str(legacy))
            self._touch_model_dir(root / 'models'
                                  / self.model_id.replace('/', '--')
                                  / 'snapshots' / 'master')
            self.assertIsNone(find_reusable_legacy_repo_dir(self.model_id))


if __name__ == '__main__':
    unittest.main()
