#  YOLOv8 Aerial Traffic & Object Detection

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![YOLOv8](https://img.shields.io/badge/YOLO-v8-green)
![Status](https://img.shields.io/badge/Status-Prototype-orange)

##  Proje Hakkında (Executive Summary)

Bu proje, İnsansız Hava Araçları (İHA) ve otonom sistemler için geliştirilmiş, gerçek zamanlı bir nesne tespit (Object Detection) modülüdür. **YOLOv8-Nano** mimarisi kullanılarak, **VisDrone** veri seti üzerinde eğitilmiştir.

Projenin temel amacı, havadan alınan görüntülerde araç, yaya ve diğer trafik unsurlarını yüksek hızda ve optimize edilmiş doğrulukla tespit etmektir. Geliştirilen algoritma, "Class Confusion" (sınıf karışıklığı) ve "Flickering" (titreme) problemlerini minimize etmek için **Confidence Thresholding** teknikleriyle güçlendirilmiştir.

##  Temel Yetenekler (Key Features)

* **Aerial Surveillance:** Havadan çekim (Drone/İHA) açısına uygun optimize edilmiş tespit yeteneği.
* **Edge AI Ready:** YOLOv8n (Nano) modeli kullanılarak, kısıtlı donanımlarda (Gömülü Sistemler) çalışabilecek hafiflikte tasarlanmıştır.
* **Multi-Class Detection:** Araç, Yaya, Motosiklet, Kamyon gibi farklı sınıfların eş zamanlı tespiti.
* **Video Inference Pipeline:** Hem kayıtlı videolarda hem de canlı kamera akışında (Webcam) çalışabilen modüler yapı.

##  Performans Metrikleri (Model Metrics)

Model, Google Colab üzerinde T4 GPU kullanılarak 20 Epoch boyunca eğitilmiştir. Eğitim sonuçlarına göre elde edilen değerler:

| Metrik | Değer | Açıklama |
| :--- | :--- | :--- |
| **Model Tipi** | YOLOv8n (Nano) | Hız/Performans optimizasyonu için seçildi. |
| **mAP50** | ~%30 | VisDrone gibi yoğun (dense) veri setleri için başlangıç seviyesi. |
| **Precision** | ~%40 | Hatalı pozitif (False Positive) oranını düşük tutar. |
| **Recall** | ~%31 | Nesne kaçırma oranını minimize eder. |
| **Eğitim Süresi** | 20 Epochs | Early stopping uygulanmıştır. |

##  Kurulum (Installation)

Projeyi yerel ortamınızda (Local Environment) çalıştırmak için aşağıdaki adımları izleyin.

### 1. Repoyu Klonlayın
```bash
git clone [https://github.com/loopBreakerr/Traffic-YOLO.git](https://github.com/loopBreakerr/Traffic-YOLO.git)
cd Traffic-YOLO
```
### 2. Gerekli Kütüphaneleri Yükleyin
pip install -r requirements.txt
(Not: requirements.txt dosyası ultralytics ve opencv-python kütüphanelerini içerir.)

### Kullanım (Usage)

Tüm ayarlar `config.yaml` dosyasından okunur ve komut satırı argümanlarıyla override edilebilir. Öncelik sırası: **CLI argümanı > config.yaml > kod içi varsayılan**.

Varsayılan yapılandırmayla çalıştırma:
```bash
python main.py
```

Yaygın override'lar:
```bash
python main.py --source input/baska_video.mp4 --output output/sonuc.mp4
python main.py --conf 0.35 --model models/best.pt
python main.py --config deneme.yaml
```

Webcam (canlı test) için `config.yaml` içindeki `source` değerini `0` yapın ya da:
```bash
python main.py --source 0
```

Önizleme penceresi olmadan (yalnızca dosyaya yazma):
```bash
python main.py --no-display
```

Tüm seçenekler: `python main.py --help`

### Proje Yapısı (Project Structure)

```
Traffic-YOLO/
├── config.yaml        # Varsayılan yapılandırma (path'ler, threshold, display)
├── main.py            # İnce giriş noktası: argparse + config + pipeline
├── src/
│   ├── cli.py         # argparse tanımı
│   ├── config.py      # YAML yükleme + CLI override birleştirme
│   ├── detector.py    # YOLO model sarmalayıcı (annotated frame + ham Results)
│   └── pipeline.py    # Video capture/writer + kare döngüsü
├── models/best.pt
├── input/
└── output/
```



### Geliştirici (Developer)
Geliştirici: Furkan (loopBreakerr)

Üniversite: Ankara Üniversitesi - Bilgisayar Mühendisliği

İletişim: [https://www.linkedin.com/in/furkangunel063431/]

