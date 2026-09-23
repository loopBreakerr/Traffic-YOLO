"""VisDrone2019-DET ham etiketlerini YOLO formatına dönüştürür.

VisDrone'un resmi kategori sırası ile `models/best.pt`'nin eğitildiği sıra
(alfabetik) FARKLI. Bu yüzden ham kategori id'leri burada modelin gerçek
sınıf sırasına eşlenir (RAW_ID_TO_MODEL_IDX). Bu eşleme bozulursa train/val
etiketleri modelin çıktı sınıflarıyla uyuşmaz; metrikler ve eğitim sonuçları
sessizce anlamsız hale gelir.

Kullanım (CLI):
    python -m src.visdrone_convert --root data/VisDrone --split val
    python -m src.visdrone_convert --root data/VisDrone --split train --cleanup
"""

import argparse
import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

# Ham VisDrone kategori id'si (annotation dosyasının 6. sütunu) -> sınıf adı.
RAW_ID_TO_NAME = {
    1: "pedestrian",
    2: "people",
    3: "bicycle",
    4: "car",
    5: "van",
    6: "truck",
    7: "tricycle",
    8: "awning-tricycle",
    9: "bus",
    10: "motor",
}

# models/best.pt'nin gerçek eğitim sınıf sırası (alfabetik).
# visdrone_val.yaml / visdrone_full.yaml'daki `names` bloğuyla BİREBİR aynı
# olmalı: biri değişirse diğeri de güncellenmeli.
MODEL_CLASS_NAMES = {
    0: "awning-tricycle",
    1: "bicycle",
    2: "bus",
    3: "car",
    4: "motor",
    5: "pedestrian",
    6: "people",
    7: "tricycle",
    8: "truck",
    9: "van",
}

_NAME_TO_MODEL_IDX = {v: k for k, v in MODEL_CLASS_NAMES.items()}
RAW_ID_TO_MODEL_IDX = {raw: _NAME_TO_MODEL_IDX[name] for raw, name in RAW_ID_TO_NAME.items()}


def verify_class_order(model_path):
    """`model_path`'teki modelin sınıf sırasının MODEL_CLASS_NAMES ile aynı olduğunu doğrular.

    Eşleşmezse RuntimeError fırlatır; aksi halde train/eval sırasında
    sessizce yanlış sınıflarla eğitim/karşılaştırma yapılır.
    """
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    actual = dict(model.names)
    if actual != MODEL_CLASS_NAMES:
        raise RuntimeError(
            f"Model sınıf sırası beklenenden farklı!\n"
            f"  Model ({model_path}): {actual}\n"
            f"  Beklenen (MODEL_CLASS_NAMES): {MODEL_CLASS_NAMES}\n"
            "src/visdrone_convert.py'deki MODEL_CLASS_NAMES ve "
            "visdrone_*.yaml'daki names bloğunu bu modele göre güncelleyin."
        )


def convert_split(root, split, source_name=None, cleanup=False):
    """Bir VisDrone split'ini (train/val/test) YOLO formatına çevirir.

    Args:
        root: VisDrone kök klasörü (ör. data/VisDrone). Ham
            `VisDrone2019-DET-{split}/` klasörünün içinde bulunduğu yer.
        split: 'train', 'val' veya 'test'.
        source_name: Ham klasör adı farklıysa override eder
            (varsayılan: f"VisDrone2019-DET-{split}").
        cleanup: True ise dönüştürme sonrası ham kaynak klasörünü siler.

    Dönüş:
        (n_images, n_labels, n_boxes)
    """
    from PIL import Image

    root = Path(root)
    source_dir = root / (source_name or f"VisDrone2019-DET-{split}")
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Ham VisDrone klasörü bulunamadı: {source_dir}")

    images_dir = root / "images" / split
    labels_dir = root / "labels" / split
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    if (source_images_dir := source_dir / "images").is_dir():
        for img in source_images_dir.glob("*.jpg"):
            img.rename(images_dir / img.name)

    n_labels = 0
    n_boxes = 0
    for ann_file in (source_dir / "annotations").glob("*.txt"):
        img_path = images_dir / ann_file.with_suffix(".jpg").name
        w_img, h_img = Image.open(img_path).size
        dw, dh = 1.0 / w_img, 1.0 / h_img

        lines = []
        with open(ann_file, encoding="utf-8") as fh:
            for row in [x.split(",") for x in fh.read().strip().splitlines()]:
                if row[4] != "0":  # ignored region değil
                    x, y, w, h = map(int, row[:4])
                    cls = RAW_ID_TO_MODEL_IDX[int(row[5])]
                    xc, yc = (x + w / 2) * dw, (y + h / 2) * dh
                    wn, hn = w * dw, h * dh
                    lines.append(f"{cls} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")
                    n_boxes += 1

        (labels_dir / ann_file.name).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        n_labels += 1

    n_images = sum(1 for _ in images_dir.glob("*.jpg"))
    logger.info(
        "[%s] %d görüntü, %d etiket dosyası, %d kutu -> %s / %s",
        split, n_images, n_labels, n_boxes, images_dir, labels_dir,
    )

    if cleanup:
        shutil.rmtree(source_dir, ignore_errors=True)
        logger.info("Ham klasör temizlendi: %s", source_dir)

    return n_images, n_labels, n_boxes


def build_parser():
    parser = argparse.ArgumentParser(description="VisDrone ham etiketlerini YOLO formatına çevirir.")
    parser.add_argument("--root", default="data/VisDrone", help="VisDrone kök klasörü (varsayılan: data/VisDrone).")
    parser.add_argument("--split", required=True, choices=["train", "val", "test"], help="Dönüştürülecek split.")
    parser.add_argument("--source-name", default=None, help="Ham klasör adı (varsayılan: VisDrone2019-DET-<split>).")
    parser.add_argument("--cleanup", action="store_true", help="Dönüştürme sonrası ham klasörü siler.")
    return parser


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    convert_split(args.root, args.split, source_name=args.source_name, cleanup=args.cleanup)


if __name__ == "__main__":
    main()
