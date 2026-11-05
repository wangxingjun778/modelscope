import io
import os
import pickle
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

from modelscope_hub.errors import CacheError, InvalidParameter

from modelscope.hub.constants import TEMPORARY_FOLDER_NAME


class TestScanCacheCommand(unittest.TestCase):
    """Test scan-cache and dry-run aliases using local identified fixtures."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve() / 'cache'
        snapshot = self.root / 'models/org--repo/snapshots/master'
        snapshot.mkdir(parents=True)
        (snapshot / 'README.md').write_text('fixture readme')
        self.env = dict(
            os.environ,
            MODELSCOPE_CACHE=str(self.root),
            MODELSCOPE_HOME=str(self.root.parent / 'auth'),
            MODELSCOPE_API_TOKEN='')

    def _run(self, arguments):
        result = subprocess.run(
            [sys.executable, '-m', 'modelscope.cli.cli'] + arguments,
            env=self.env,
            capture_output=True,
            text=True,
            timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout

    def test_scan_default_dir(self):
        output = self._run(['scan-cache'])
        self.assertIn('cache_dir', output)
        self.assertIn('org/repo', output)

    def test_scan_given_dir(self):
        output = self._run(['scan-cache', '--dir', str(self.root)])
        self.assertIn('cache_dir', output)
        self.assertIn('org/repo', output)

    def test_scan_not_exist_dir(self):
        output = self._run(
            ['scan-cache', '--dir',
             str(self.root.parent / 'missing')])
        self.assertIn('0 repo(s)', output)

    def test_all_console_aliases_support_clear_dry_run(self):
        repo = self.root / 'models/org--repo'
        for alias in ('modelscope', 'ms', 'modelscope-hub', 'ms-hub'):
            with self.subTest(alias=alias):
                arguments = [
                    str(Path(sys.executable).parent / alias), 'clear-cache',
                    '--model', 'org/repo', '--dry-run'
                ]
                result = subprocess.run(
                    arguments,
                    env=self.env,
                    capture_output=True,
                    text=True,
                    timeout=30)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(str(repo), result.stdout)
                self.assertIn('Dry run', result.stdout)
                self.assertTrue(repo.exists())


class TestClearCacheCommand(unittest.TestCase):
    """clear-cache only deletes identified repository caches below the active
    modelscope_hub root, after the exact paths have been confirmed:
      * MODELSCOPE_CACHE set   -> $MODELSCOPE_CACHE
      * MODELSCOPE_CACHE unset -> ~/.cache/modelscope, which also covers the
        legacy ~/.cache/modelscope/hub layouts
    SDK-owned files, unidentified folders and other repositories are kept.
    """

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.tmp = Path(temp.name).resolve()
        patchers = [
            mock.patch.dict(
                os.environ, {
                    'HOME': str(self.tmp / 'home'),
                    'MODELSCOPE_HOME': str(self.tmp / 'auth'),
                }),
            mock.patch('modelscope_hub.config._default_config', None),
        ]
        for patcher in patchers:
            patcher.start()
            self.addCleanup(patcher.stop)
        for name in ('MODELSCOPE_CACHE', 'MS_CACHE_HOME'):
            os.environ.pop(name, None)

    @staticmethod
    def _put(path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'payload')
        return path

    def _standard(self, root: Path, kind: str, repo_id: str) -> Path:
        repo = root / f'{kind}s' / repo_id.replace('/', '--')
        self._put(repo / 'snapshots' / 'master' / 'config.json')
        return repo

    def _legacy(self, path: Path, repo_id: str) -> Path:
        self._put(path / 'config.json')
        (path / '.mdl').write_bytes(pickle.dumps({'id': repo_id}))
        return path

    def _clear(self, model=None, dataset=None, answer='Y', dry_run=False):
        from modelscope.cli.clearcache import ClearCacheCMD
        args = Namespace(model=model, dataset=dataset, dry_run=dry_run)
        with mock.patch('builtins.input', return_value=answer) as prompt, \
                mock.patch('sys.stdout', new_callable=io.StringIO) as out:
            ClearCacheCMD(args).execute()
        return prompt, out.getvalue()

    def test_env_cache(self):
        root = self.tmp / 'cache'
        os.environ['MODELSCOPE_CACHE'] = str(root)
        model = self._standard(root, 'model', 'org/model')
        temp = self._put(root / 'models' / TEMPORARY_FOLDER_NAME / 'org'
                         / 'model' / 'part.bin').parent
        dataset = self._legacy(root / 'datasets' / 'org' / 'dataset',
                               'org/dataset')
        kept = [
            self._standard(root, 'model', 'org/other'),
            self._put(root / 'ast_indexer'),
            self._put(root / 'models' / 'org' / 'unknown' / 'weights.bin'),
        ]
        _, output = self._clear(model='org/model')
        self.assertIn(str(model), output)
        self.assertFalse(model.exists())
        self.assertFalse(temp.exists())
        self.assertTrue(dataset.exists())
        self._clear(dataset='org/dataset')
        self.assertFalse(dataset.exists())
        for path in kept:
            self.assertTrue(path.exists(), path)

    def test_default_cache(self):
        root = self.tmp / 'home' / '.cache' / 'modelscope'
        legacy = self._legacy(root / 'hub' / 'models' / 'org' / 'repo',
                              'org/repo')
        kept = [
            self._put(root / 'hub' / 'ast_indexer'),
            self._put(root / 'hub' / 'models' / 'org' / 'other' / 'w.bin'),
        ]
        from modelscope.cli.clearcache import ClearCacheCMD
        self.assertEqual(ClearCacheCMD(Namespace()).cache_dir, str(root))
        self._clear(model='org/repo')
        self.assertFalse(legacy.exists())
        for path in kept:
            self.assertTrue(path.exists(), path)

    def test_clear_all_only_removes_identified_repos(self):
        root = self.tmp / 'cache'
        os.environ['MODELSCOPE_CACHE'] = str(root)
        removed = [
            self._standard(root, 'model', 'org/model'),
            self._legacy(root / 'hub' / 'datasets' / 'org' / 'data',
                         'org/data'),
        ]
        kept = [
            self._put(root / 'hub' / 'ast_indexer'),
            self._put(root / 'models' / 'org' / 'unknown' / 'weights.bin'),
            self._put(root / 'datasets' / 'downloads' / 'archive.zip'),
        ]
        self._clear()
        for path in removed:
            self.assertFalse(path.exists(), path)
        for path in kept:
            self.assertTrue(path.exists(), path)

    def test_dry_run_and_abort_delete_nothing(self):
        root = self.tmp / 'cache'
        os.environ['MODELSCOPE_CACHE'] = str(root)
        model = self._standard(root, 'model', 'org/model')
        prompt, output = self._clear(model='org/model', dry_run=True)
        prompt.assert_not_called()
        self.assertIn(str(model), output)
        self.assertTrue(model.exists())
        _, output = self._clear(model='org/model', answer='n')
        self.assertIn('Operation aborted.', output)
        self.assertTrue(model.exists())

    def test_missing_cache_is_reported(self):
        os.environ['MODELSCOPE_CACHE'] = str(self.tmp / 'cache')
        prompt, output = self._clear(dataset='org/missing')
        prompt.assert_not_called()
        self.assertIn('not found', output)

    def test_empty_id_never_clears_everything(self):
        root = self.tmp / 'cache'
        os.environ['MODELSCOPE_CACHE'] = str(root)
        model = self._standard(root, 'model', 'org/model')
        for kwargs in ({'model': ''}, {'dataset': ''}):
            with self.subTest(**kwargs), \
                    self.assertRaises(InvalidParameter):
                self._clear(**kwargs)
        self.assertTrue(model.exists())

    def test_unsafe_repo_and_external_link_abort_before_deletion(self):
        root = self.tmp / 'cache'
        os.environ['MODELSCOPE_CACHE'] = str(root)
        kept = self._standard(root, 'model', 'org/kept')
        sentinel = self._put(self.tmp / 'external/sentinel')
        alias = root / 'models/org/link'
        alias.parent.mkdir(parents=True)
        alias.symlink_to(sentinel.parent, target_is_directory=True)
        with self.assertRaises(InvalidParameter):
            self._clear(model='../external')
        with self.assertRaises(CacheError):
            self._clear()
        self.assertEqual(sentinel.read_bytes(), b'payload')
        self.assertTrue(kept.exists())

    def test_untyped_flat_cache_rejects_whole_batch(self):
        root = self.tmp / 'cache'
        os.environ['MODELSCOPE_CACHE'] = str(root)
        kept = self._standard(root, 'model', 'org/kept')
        flat = self._legacy(root / 'owner/repo', 'owner/repo')
        with self.assertRaises(CacheError):
            self._clear()
        self.assertTrue(kept.exists())
        self.assertTrue(flat.exists())

    def test_confirmation_precedes_execution_and_changed_plan_is_rejected(
            self):
        from modelscope.cli.clearcache import ClearCacheCMD
        root = self.tmp / 'cache'
        os.environ['MODELSCOPE_CACHE'] = str(root)
        model = self._standard(root, 'model', 'org/model')
        args = Namespace(model='org/model', dataset=None)

        def confirm(_):
            self.assertTrue(model.exists())
            self._put(model / 'snapshots/v2/config.json')
            return 'Y'

        with mock.patch('builtins.input', side_effect=confirm), \
                mock.patch('sys.stdout', new_callable=io.StringIO), \
                self.assertRaises(CacheError):
            ClearCacheCMD(args).execute()
        self.assertTrue(model.exists())


if __name__ == '__main__':
    unittest.main()
