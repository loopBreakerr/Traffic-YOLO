"""Yapılandırma yükleme: YAML dosyası + CLI override birleştirme.

Öncelik sırası (yüksekten düşüğe):
    1. CLI argümanları
    2. config.yaml
    3. Kod içi varsayılanlar (DEFAULT_CONFIG)
"""

import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = {
    "model": {"path": "models/best.pt"},
    "source": "input/test_video.mp4",
    "output": "output/result.mp4",
    "inference": {"confidence_threshold": 0.50},
    "display": {
        "enabled": True,
        "window_name": "YOLOv8 Traffic Analysis",
        "quit_key": "q",
    },
}


def deep_merge(base, override):
    """`override` içindeki değerleri `base` üzerine iç içe (recursive) uygular.

    src.train tarafından da (train_config.yaml birleştirmesi için) kullanılır.
    """
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path):
    """YAML config dosyasını yükleyip varsayılanların üzerine birleştirir.

    `path` verilmişse ve dosya yoksa açıklayıcı bir hata fırlatır.
    """
    if path is None:
        return deep_merge(DEFAULT_CONFIG, {})

    cfg_path = Path(path)
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config dosyası bulunamadı: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh) or {}

    if not isinstance(loaded, dict):
        raise ValueError(f"Config dosyası bir sözlük (mapping) içermeli: {cfg_path}")

    logger.info("Yapılandırma yüklendi: %s", cfg_path)
    return deep_merge(DEFAULT_CONFIG, loaded)


def merge_cli(config, args):
    """argparse Namespace'indeki None olmayan değerleri config üzerine uygular."""
    overrides = {}
    if getattr(args, "model", None) is not None:
        overrides["model"] = {"path": args.model}
    if getattr(args, "source", None) is not None:
        overrides["source"] = args.source
    if getattr(args, "output", None) is not None:
        overrides["output"] = args.output
    if getattr(args, "conf", None) is not None:
        overrides["inference"] = {"confidence_threshold": args.conf}
    if getattr(args, "no_display", False):
        overrides["display"] = {"enabled": False}

    if overrides:
        logger.info("CLI override'ları uygulanıyor: %s", sorted(overrides))
    return deep_merge(config, overrides)
