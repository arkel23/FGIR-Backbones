"""Config loading: several YAMLs at once, with the command line winning over all of them."""
import os
import argparse

import yaml
from omegaconf import OmegaConf


def expand_defaults(path):
    """One config as a flat dict; its `defaults:` includes merge first, so the file wins."""
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}

    merged = {}
    for rel in (cfg.pop('defaults', None) or {}).values():
        with open(os.path.join(os.path.dirname(path), rel)) as f:
            merged.update(yaml.safe_load(f) or {})
    merged.update(cfg)
    return merged


def load_config_from_yaml(paths):
    """Merge one or more config files, left to right."""
    if isinstance(paths, str):
        paths = [paths]

    cfg = OmegaConf.create({})
    for path in paths:
        cfg = OmegaConf.merge(cfg, OmegaConf.create(expand_defaults(os.path.abspath(path))))
    return OmegaConf.to_object(cfg)


def explicit_cli_args(parser):
    """The dests the user actually passed. Re-parsing with every default suppressed is the
    only way to tell "not given" from "given a value that equals the default"."""
    saved = {a.dest: a.default for a in parser._actions}
    try:
        for action in parser._actions:
            action.default = argparse.SUPPRESS
        return set(vars(parser.parse_args()))
    finally:
        for action in parser._actions:
            action.default = saved[action.dest]


def adjust_config(parser, args):
    """Fill args from --cfg, leaving anything given on the command line untouched."""
    if not args.cfg:
        return args

    explicit = explicit_cli_args(parser)
    for key, value in load_config_from_yaml(args.cfg).items():
        if hasattr(args, key) and key not in explicit:
            setattr(args, key, value)
    return args
