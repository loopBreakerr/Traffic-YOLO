"""VisDrone üzerinde YOLO fine-tuning (Colab GPU'da çalıştırılmak üzere).

Kullanım:
    python -m src.train
    python -m src.train --epochs 5 --patience 20   # duman testi

train_config.yaml'dan okur, CLI argümanlarıyla override edilebilir.
Öncelik sırası: CLI argümanı > train_config.yaml > DEFAULT_TRAIN_CONFIG.

VARSAYIM: Orijinal Colab notebook'u kayıp; `batch` ve `imgsz` için gerçekte
kullanılan değerler bilinmiyor. DEFAULT_TRAIN_CONFIG'teki batch=16,
imgsz=640 makul varsayılanlardır - bkz. train_config.yaml'daki not.
"""

import argparse
import logging
from pathlib import Path

import yaml
from ultralytics import YOLO

from src.config import deep_merge
from src.visdrone_convert import verify_class_order

logger = logging.getLogger(__name__)

DEFAULT_TRAIN_CONFIG = {
    "base_model": "models/best.pt",
    "data": "visdrone_full.yaml",
    "epochs": 150,
    "patience": 20,
    "batch": 16,  # VARSAYIM - orijinal Colab notebook kayıp
    "imgsz": 640,  # VARSAYIM - orijinal Colab notebook kayıp
    "device": None,
    "project": "runs/train",
    "name": "exp1_more_epochs",
    "seed": 0,
}


def load_train_config(path):
    if path is None:
        return dict(DEFAULT_TRAIN_CONFIG)

    cfg_path = Path(path)
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config dosyası bulunamadı: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh) or {}

    logger.info("Eğitim yapılandırması yüklendi: %s", cfg_path)
    return deep_merge(DEFAULT_TRAIN_CONFIG, loaded)


def build_parser():
    parser = argparse.ArgumentParser(
        prog="train.py",
        description="VisDrone üzerinde YOLO fine-tuning.",
    )
    parser.add_argument("--config", default="train_config.yaml", help="YAML yapılandırma dosyası (varsayılan: train_config.yaml).")
    parser.add_argument("--base-model", default=None, help="Devam edilecek checkpoint. Config'teki base_model'i override eder.")
    parser.add_argument("--data", default=None, help="Dataset YAML yolu. Config'teki data'yı override eder.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--device", default=None, help="Ör. 0, 0,1 veya cpu.")
    parser.add_argument("--project", default=None, help="Çıktı kök klasörü (ör. Drive yolu).")
    parser.add_argument("--name", default=None, help="Bu koşunun alt klasör adı.")
    parser.add_argument("--seed", type=int, default=None)
    return parser


def merge_cli(config, args):
    overrides = {}
    if args.base_model is not None:
        overrides["base_model"] = args.base_model
    if args.data is not None:
        overrides["data"] = args.data
    for key in ("epochs", "patience", "batch", "imgsz", "device", "project", "name", "seed"):
        value = getattr(args, key)
        if value is not None:
            overrides[key] = value

    if overrides:
        logger.info("CLI override'ları uygulanıyor: %s", sorted(overrides))
    return deep_merge(config, overrides)


def main(argv=None):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    args = build_parser().parse_args(argv)
    config = load_train_config(args.config)
    config = merge_cli(config, args)

    base_model = config["base_model"]
    if not Path(base_model).is_file():
        raise FileNotFoundError(f"Base model bulunamadı: {base_model}")

    logger.info("Sınıf sırası doğrulanıyor: %s", base_model)
    verify_class_order(base_model)

    logger.info("Model yükleniyor: %s", base_model)
    model = YOLO(base_model)

    logger.info(
        "Eğitim başlıyor: data=%s epochs=%s patience=%s batch=%s imgsz=%s device=%s project=%s name=%s seed=%s",
        config["data"], config["epochs"], config["patience"], config["batch"],
        config["imgsz"], config["device"], config["project"], config["name"], config["seed"],
    )
    results = model.train(
        data=config["data"],
        epochs=config["epochs"],
        patience=config["patience"],
        batch=config["batch"],
        imgsz=config["imgsz"],
        device=config["device"],
        project=config["project"],
        name=config["name"],
        seed=config["seed"],
    )

    logger.info("Eğitim tamamlandı. Sonuçlar: %s", results.save_dir)
    return results


if __name__ == "__main__":
    main()
