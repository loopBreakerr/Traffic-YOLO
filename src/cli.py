"""Komut satırı arayüzü tanımı."""

import argparse


def build_parser():
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="YOLOv8 trafik / nesne tespiti video inference pipeline'ı.",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="YAML yapılandırma dosyası yolu (varsayılan: config.yaml).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model ağırlık dosyası yolu. Config'teki model.path değerini override eder.",
    )
    parser.add_argument(
        "--source",
        default=None,
        help="Video dosyası yolu veya webcam indeksi (ör. 0). Config'teki source'u override eder.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Çıktı video dosyası yolu. Config'teki output'u override eder.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=None,
        help="Confidence threshold (0-1). Config'teki inference.confidence_threshold'u override eder.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Canlı önizleme penceresini kapatır; kareler yalnızca dosyaya yazılır.",
    )
    return parser


def parse_args(argv=None):
    return build_parser().parse_args(argv)
