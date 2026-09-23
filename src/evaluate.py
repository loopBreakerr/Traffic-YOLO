"""VisDrone val split üzerinde model değerlendirmesi.

Kullanım:
    python -m src.evaluate

models/best.pt modelini visdrone_val.yaml'daki val split'i üzerinde
çalıştırır, genel + sınıf bazlı P/R/mAP50/mAP50-95 tablosunu konsola
yazdırır ve Ultralytics'in ürettiği confusion_matrix.png / PR_curve.png
dosyalarını docs/eval/ altına kopyalar.
"""

import logging
import shutil
from pathlib import Path

import yaml
from ultralytics import YOLO

logger = logging.getLogger(__name__)

MODEL_PATH = "models/best.pt"
DATA_YAML = "visdrone_val.yaml"
DOCS_EVAL_DIR = Path("docs/eval")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")

# (kaynak dosya adı, docs/eval/ altındaki hedef ad).
# Ultralytics detect görevi için PR eğrisini "BoxPR_curve.png" olarak
# kaydediyor; README'de kullanılacak sade isme (PR_curve.png) kopyalanır.
PLOT_FILES = (
    ("confusion_matrix.png", "confusion_matrix.png"),
    ("BoxPR_curve.png", "PR_curve.png"),
)


def _count_val_images(data_yaml):
    """visdrone_val.yaml'daki val split'indeki toplam görüntü sayısını döner."""
    with open(data_yaml, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    val_dir = Path(data["path"]) / data["val"]
    return sum(1 for f in val_dir.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS)


def _print_results_table(metrics, total_images):
    names = metrics.names
    box = metrics.box

    header = f"{'Sınıf':<18}{'Görüntü':>9}{'Kutu':>8}{'P':>8}{'R':>8}{'mAP50':>9}{'mAP50-95':>11}"
    print()
    print(header)
    print("-" * len(header))
    print(
        f"{'all':<18}{total_images:>9}{int(metrics.nt_per_class.sum()):>8}"
        f"{box.mp:>8.3f}{box.mr:>8.3f}{box.map50:>9.3f}{box.map:>11.3f}"
    )
    for i, cls_idx in enumerate(box.ap_class_index):
        cls_idx = int(cls_idx)
        n_img = int(metrics.nt_per_image[cls_idx]) if cls_idx < len(metrics.nt_per_image) else 0
        n_box = int(metrics.nt_per_class[cls_idx]) if cls_idx < len(metrics.nt_per_class) else 0
        print(
            f"{names[cls_idx]:<18}{n_img:>9}{n_box:>8}"
            f"{box.p[i]:>8.3f}{box.r[i]:>8.3f}{box.ap50[i]:>9.3f}{box.ap[i]:>11.3f}"
        )
    print()


def _copy_plots(save_dir):
    DOCS_EVAL_DIR.mkdir(parents=True, exist_ok=True)
    copied = []
    for src_name, dst_name in PLOT_FILES:
        src = Path(save_dir) / src_name
        if not src.is_file():
            logger.warning("Beklenen grafik bulunamadı, atlanıyor: %s", src)
            continue
        dst = DOCS_EVAL_DIR / dst_name
        shutil.copy2(src, dst)
        copied.append(dst)
    return copied


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    model_file = Path(MODEL_PATH)
    if not model_file.is_file():
        raise FileNotFoundError(f"Model dosyası bulunamadı: {model_file}")

    logger.info("Model yükleniyor: %s", model_file)
    model = YOLO(str(model_file))

    logger.info("Değerlendirme başlıyor: %s (split=val)", DATA_YAML)
    metrics = model.val(data=DATA_YAML, split="val")

    _print_results_table(metrics, total_images=_count_val_images(DATA_YAML))

    copied = _copy_plots(metrics.save_dir)
    for path in copied:
        logger.info("Kopyalandı: %s", path)


if __name__ == "__main__":
    main()
