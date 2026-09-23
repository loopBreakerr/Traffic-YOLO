"""YOLO model sarmalayıcı."""

import logging
from pathlib import Path

from ultralytics import YOLO

logger = logging.getLogger(__name__)


class Detector:
    """Bir YOLO ağırlık dosyasını yükler ve kare başına tahmin üretir."""

    def __init__(self, model_path):
        model_file = Path(model_path)
        if not model_file.is_file():
            raise FileNotFoundError(f"Model dosyası bulunamadı: {model_file}")

        logger.info("Model yükleniyor: %s", model_file)
        self.model = YOLO(str(model_file))

    def predict(self, frame, conf):
        """Tek bir kare için tahmin yapar.

        Dönüş:
            annotated_frame (np.ndarray): Kutu/etiketlerin çizildiği BGR kare.
            result (ultralytics.engine.results.Results): Ham sonuç nesnesi.
                İleride eval script'i (per-class AP, confusion matrix) bu ham
                veriyi kullanacak.
        """
        results = self.model(frame, conf=conf, verbose=False)
        result = results[0]
        annotated_frame = result.plot()
        return annotated_frame, result
