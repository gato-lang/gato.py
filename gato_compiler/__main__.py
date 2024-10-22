from argparse import ArgumentParser

from .source import Module


argparser = ArgumentParser()
argparser.add_argument(
    'source',
    help="a file or directory containing the source code of the module to compile",
)


if __name__ == '__main__':
    args = argparser.parse_args()
    Module.parse(args.source).compile()
