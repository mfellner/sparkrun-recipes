#!/usr/bin/env python3
"""Translate SparkRun 0.3.6's API alias flag, then exec the real TensorFold CLI."""
import os
import sys


def translate(args):
    if args.count('--served-model-name') != 1 or '--name' in args:
        raise ValueError('exactly one --served-model-name and no --name required')
    i = args.index('--served-model-name')
    if i + 1 >= len(args) or not args[i + 1] or args[i + 1].startswith('-'):
        raise ValueError('missing served model alias')
    result = list(args)
    result[i] = '--name'
    return result


if __name__ == '__main__':
    try:
        args = translate(sys.argv[1:])
    except ValueError as exc:
        sys.exit(str(exc))
    os.execvp('tensorfold', ['tensorfold', 'serve', *args])
