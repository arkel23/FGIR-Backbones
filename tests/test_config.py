"""Tests for fgir_backbones/other_utils/config.py. Same bar as tests/test_exemplar.py: flat tests,
one behavior each, no I/O beyond a temp dir, runs in seconds.

    python -m unittest tests.test_config -v
"""
import sys
import tempfile
import unittest
from pathlib import Path
from argparse import ArgumentParser

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fgir_backbones.other_utils.config import (expand_defaults, load_config_from_yaml,
                                               explicit_cli_args, adjust_config)


def write(dirpath, name, text):
    path = Path(dirpath) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return str(path)


def cfg_parser():
    parser = ArgumentParser()
    parser.add_argument('--cfg', type=str, nargs='+', default=None)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--dataset_name', type=str, default='aircraft')
    parser.add_argument('--pretrained', action='store_true')
    return parser


class TestLoading(unittest.TestCase):
    def test_single_config_loads_its_keys(self):
        with tempfile.TemporaryDirectory() as d:
            path = write(d, 'a.yaml', 'lr: 0.5\ndataset_name: cotton\n')
            self.assertEqual(load_config_from_yaml(path), {'lr': 0.5, 'dataset_name': 'cotton'})

    def test_defaults_block_merges_and_the_including_file_wins(self):
        with tempfile.TemporaryDirectory() as d:
            write(d, 'sub/ds.yaml', 'dataset_name: soygene\nlr: 0.9\n')
            path = write(d, 'top.yaml', "defaults:\n    dataset: 'sub/ds.yaml'\nlr: 0.25\n")
            self.assertEqual(load_config_from_yaml(path), {'dataset_name': 'soygene', 'lr': 0.25})

    def test_several_configs_merge_left_to_right(self):
        with tempfile.TemporaryDirectory() as d:
            first = write(d, 'a.yaml', 'lr: 0.1\ndataset_name: cotton\n')
            second = write(d, 'b.yaml', 'lr: 0.2\n')
            merged = load_config_from_yaml([first, second])
            self.assertEqual(merged, {'lr': 0.2, 'dataset_name': 'cotton'})

    def test_empty_config_is_not_an_error(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(load_config_from_yaml(write(d, 'empty.yaml', '')), {})

    def test_defaults_key_never_reaches_the_result(self):
        with tempfile.TemporaryDirectory() as d:
            write(d, 'sub/ds.yaml', 'dataset_name: cub\n')
            path = write(d, 'top.yaml', "defaults:\n    dataset: 'sub/ds.yaml'\n")
            self.assertNotIn('defaults', expand_defaults(path))


class TestPrecedence(unittest.TestCase):
    def test_explicit_args_are_detected_even_when_equal_to_the_default(self):
        parser = cfg_parser()
        sys.argv = ['prog', '--lr', '0.0001']  # 0.0001 is also the default
        self.assertIn('lr', explicit_cli_args(parser))
        self.assertNotIn('dataset_name', explicit_cli_args(parser))

    def test_command_line_beats_the_config(self):
        with tempfile.TemporaryDirectory() as d:
            path = write(d, 'a.yaml', 'lr: 0.5\ndataset_name: cotton\n')
            parser = cfg_parser()
            sys.argv = ['prog', '--cfg', path, '--lr', '0.9']
            args = adjust_config(parser, parser.parse_args())
            self.assertEqual(args.lr, 0.9)          # given on the command line
            self.assertEqual(args.dataset_name, 'cotton')  # left to the config

    def test_a_cli_value_equal_to_the_default_still_beats_the_config(self):
        # The bug this mechanism exists for: comparing against defaults cannot tell
        # "not passed" from "passed a value that happens to equal the default".
        with tempfile.TemporaryDirectory() as d:
            path = write(d, 'a.yaml', 'lr: 0.5\n')
            parser = cfg_parser()
            sys.argv = ['prog', '--cfg', path, '--lr', '0.0001']
            self.assertEqual(adjust_config(parser, parser.parse_args()).lr, 0.0001)

    def test_store_true_flag_is_not_reverted_by_the_config(self):
        with tempfile.TemporaryDirectory() as d:
            path = write(d, 'a.yaml', 'pretrained: false\n')
            parser = cfg_parser()
            sys.argv = ['prog', '--cfg', path, '--pretrained']
            self.assertTrue(adjust_config(parser, parser.parse_args()).pretrained)

    def test_config_fills_what_the_command_line_left_alone(self):
        with tempfile.TemporaryDirectory() as d:
            path = write(d, 'a.yaml', 'pretrained: true\nlr: 0.5\n')
            parser = cfg_parser()
            sys.argv = ['prog', '--cfg', path]
            args = adjust_config(parser, parser.parse_args())
            self.assertTrue(args.pretrained)
            self.assertEqual(args.lr, 0.5)

    def test_keys_the_parser_does_not_know_are_ignored(self):
        with tempfile.TemporaryDirectory() as d:
            path = write(d, 'a.yaml', 'not_a_flag: 1\nlr: 0.5\n')
            parser = cfg_parser()
            sys.argv = ['prog', '--cfg', path]
            args = adjust_config(parser, parser.parse_args())
            self.assertFalse(hasattr(args, 'not_a_flag'))

    def test_no_config_leaves_args_untouched(self):
        parser = cfg_parser()
        sys.argv = ['prog', '--lr', '0.7']
        self.assertEqual(adjust_config(parser, parser.parse_args()).lr, 0.7)

    def test_parser_defaults_survive_the_explicit_arg_probe(self):
        # explicit_cli_args mutates defaults to SUPPRESS; it must put them back.
        parser = cfg_parser()
        sys.argv = ['prog']
        explicit_cli_args(parser)
        self.assertEqual(parser.parse_args().lr, 1e-4)


class TestRepoConfigs(unittest.TestCase):
    def test_a_real_repo_config_composes(self):
        root = Path(__file__).resolve().parents[1]
        cfg = load_config_from_yaml(str(root / 'configs' / 'aircraft_ft_medaugs.yaml'))
        self.assertEqual(cfg['dataset_name'], 'aircraft')
        self.assertNotIn('defaults', cfg)


if __name__ == '__main__':
    unittest.main()
