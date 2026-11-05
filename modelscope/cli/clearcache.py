# Copyright (c) Alibaba, Inc. and its affiliates.
"""Clear-cache CLI command — retained for backward compatibility.

The ``modelscope_hub`` CLI now owns ``clear-cache`` as an alias for
``cache clear``, but this module preserves the legacy :class:`ClearCacheCMD`
class so that existing tests and callers that import it directly continue
to work. Both rely on the same ``modelscope_hub`` planner: only identified
repository caches are deleted, after the exact paths have been confirmed.
"""
from argparse import ArgumentParser

from modelscope.cli.base import CLICommand
from modelscope.hub.cache_manager import execute_cache_clear, plan_cache_clear
from modelscope.hub.utils.utils import convert_readable_size
from modelscope.utils.file_utils import get_hub_cache_root


def subparser_func(args):
    """ Function which will be called for a specific sub parser.
    """
    return ClearCacheCMD(args)


class ClearCacheCMD(CLICommand):
    name = 'clear-cache'

    def __init__(self, args):
        self.args = args
        self.cache_dir = get_hub_cache_root()

    @staticmethod
    def register(subparsers) -> None:
        """Register clear-cache subcommand (CLICommand ABC contract)."""
        ClearCacheCMD.define_args(subparsers)

    @staticmethod
    def define_args(parsers: ArgumentParser):
        """ define args for clear-cache command.
        """
        parser = parsers.add_parser(ClearCacheCMD.name)
        group = parser.add_mutually_exclusive_group()
        group.add_argument(
            '--model',
            type=str,
            help='The id of the model whose cache will be cleared. '
            'If neither model or dataset id is provided, the caches of all '
            'identified models and datasets will be cleared.')
        group.add_argument(
            '--dataset',
            type=str,
            help='The id of the dataset whose cache will be cleared. '
            'If neither model or dataset id is provided, the caches of all '
            'identified models and datasets will be cleared.')
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Show the cache paths that would be deleted, delete nothing.')

        parser.set_defaults(func=subparser_func)

    def execute(self):
        self._execute_with_confirmation()

    def _execute_with_confirmation(self):
        model = getattr(self.args, 'model', None)
        dataset = getattr(self.args, 'dataset', None)
        # Compare with None: an empty id is rejected by the planner instead of
        # widening the scope to the whole cache.
        if model is not None:
            repo_type, repo_id, scope = 'model', model, f'model {model}'
        elif dataset is not None:
            repo_type, repo_id, scope = 'dataset', dataset, f'dataset {dataset}'
        else:
            repo_type, repo_id, scope = None, None, 'all models and datasets'

        plan = plan_cache_clear(self.cache_dir, repo_type, repo_id)
        if not plan.targets:
            print(f'Cache for {scope} not found in {plan.cache_dir}. '
                  'Nothing to do.')
            return
        print(f'\nYou are about to delete the local cache for {scope} '
              f'in {plan.cache_dir}:')
        for target in plan.targets:
            print(f'  {target.path} ({convert_readable_size(target.size)})')
        if getattr(self.args, 'dry_run', False):
            print('Dry run, nothing deleted.')
            return

        user_input = input(
            '\nPlease press Y or y to proceed, any other key to abort.\n'
        ).strip().upper()
        if user_input != 'Y':
            print('Operation aborted.')
            return
        freed = execute_cache_clear(plan)
        print(f'Cache cleared, {convert_readable_size(freed)} freed.')
