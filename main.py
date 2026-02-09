import cv2
from ultralytics import YOLO

# 1. Konfigürasyonlar
MODEL_PATH = 'models/best.pt'
SOURCE_PATH = 'input/test_video.mp4'
OUTPUT_PATH = 'output/result.mp4'
CONFIDENCE_THRESHOLD = 0.50

def run_inference():
    # Model Yükleme
    print(f"🚀 Model yükleniyor: {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)

    # Video Kaynağını Aç
    cap = cv2.VideoCapture(SOURCE_PATH)
    
    # Video Özelliklerini Al
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Kayıtçı (VideoWriter) Oluştur
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

    print("Analiz başladı. Çıkmak için 'q' tuşuna basınız.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Tahmin Yap (Inference)
        results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)

        # Sonuçları Kareye Çiz
        annotated_frame = results[0].plot()

        # Ekrana Göster (Canlı İzleme)
        cv2.imshow("YOLOv8 Traffic Analysis", annotated_frame)
        
        # Dosyaya Yaz
        out.write(annotated_frame)

        # 'q' tuşu ile çıkış
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Kaynakları Serbest Bırak
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"✅ İşlem tamamlandı. Çıktı: {OUTPUT_PATH}")

if __name__ == '__main__':
    run_inference()