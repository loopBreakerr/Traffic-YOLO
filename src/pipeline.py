"""Video inference pipeline: kaynak açma, kare döngüsü, çizim, kayıt."""

import logging
from pathlib import Path

import cv2

from src.detector import Detector

logger = logging.getLogger(__name__)

DEFAULT_FPS = 30


def _resolve_source(source):
    """Webcam indeksini int'e çevirir; dosya yollarını olduğu gibi bırakır."""
    if isinstance(source, bool):
        return source
    if isinstance(source, int):
        return source
    if isinstance(source, str) and source.isdigit():
        return int(source)
    return source


def run(config):
    """Verilen yapılandırmayla inference pipeline'ını çalıştırır.

    Dönüş: işlenen kare sayısı (int).
    """
    model_path = config["model"]["path"]
    source = _resolve_source(config["source"])
    output_path = config["output"]
    conf = config["inference"]["confidence_threshold"]

    display_cfg = config.get("display", {})
    display_enabled = bool(display_cfg.get("enabled", True))
    window_name = display_cfg.get("window_name", "YOLOv8 Traffic Analysis")
    quit_key = (str(display_cfg.get("quit_key", "q")) or "q")[0]

    detector = Detector(model_path)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Video kaynağı açılamadı: {source}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        logger.warning("Kaynak FPS okunamadı; varsayılan %d kullanılıyor.", DEFAULT_FPS)
        fps = DEFAULT_FPS

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    if display_enabled:
        logger.info("Analiz başladı. Çıkmak için '%s' tuşuna basınız.", quit_key)
    else:
        logger.info("Analiz başladı (önizleme kapalı; kareler yalnızca dosyaya yazılıyor).")

    frame_count = 0
    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            annotated_frame, _result = detector.predict(frame, conf)

            writer.write(annotated_frame)
            frame_count += 1

            # display kapalıysa hiçbir HighGUI çağrısı yapılmaz.
            if display_enabled:
                cv2.imshow(window_name, annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord(quit_key):
                    logger.info("Kullanıcı '%s' tuşuyla durdurdu.", quit_key)
                    break
    finally:
        cap.release()
        writer.release()
        if display_enabled:
            cv2.destroyAllWindows()

    logger.info(
        "İşlem tamamlandı. %d kare işlendi (%dx%d @ %d fps). Çıktı: %s",
        frame_count, width, height, fps, out_path,
    )
    return frame_count
