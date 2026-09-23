"""YOLOv8 Traffic Analysis - giriş noktası.

Yapılandırma config.yaml'dan okunur, CLI argümanlarıyla override edilir.
Ayrıntılar için: python main.py --help
"""

import logging

from src.cli import parse_args
from src.config import load_config, merge_cli
from src.pipeline import run


def main(argv=None):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    args = parse_args(argv)
    config = load_config(args.config)
    config = merge_cli(config, args)

    run(config)


if __name__ == "__main__":
    main()
