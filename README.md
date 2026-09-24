# Traffic-YOLO

VisDrone2019-DET veri seti üzerinde eğitilmiş bir YOLOv8n modeliyle trafik sahnelerinde (araç, yaya, bisiklet vb.) nesne tespiti yapan uçtan uca bir bilgisayarlı görü projesi. Modülerleştirilmiş bir inference pipeline'ı, reproducible bir değerlendirme script'i ve Colab üzerinde çalışan bir eğitim pipeline'ı içerir.

## Genel Bakış

Proje üç ana bileşenden oluşuyor:

- **Inference** (`main.py` + `src/`) — bir video üzerinde nesne tespiti çalıştırır, sonucu kaydeder/gösterir.
- **Değerlendirme** (`src/evaluate.py`) — modelin VisDrone val split'i üzerindeki gerçek performansını (mAP, precision, recall, confusion matrix) ölçer.
- **Eğitim** (`src/train.py` + `colab_train.ipynb`) — modeli VisDrone train split'i üzerinde (yeniden) eğitmek için reproducible bir pipeline.

## Sonuçlar

İki model karşılaştırılıyor: **v1** (orijinal, ~20 epoch eğitilmiş baseline) ve **v2** (v1 üzerinden 111 ek epoch daha eğitilmiş, Colab T4 GPU'da).

### Genel metrikler (VisDrone val split, 548 görüntü)

| Model | Precision | Recall | mAP50 | mAP50-95 |
|---|--:|--:|--:|--:|
| v1 (baseline) | 0.423 | 0.310 | 0.311 | 0.179 |
| v2 (+111 epoch) | 0.448 | 0.352 | **0.352** | 0.203 |
| Δ | +0.025 | +0.042 | **+0.041** | +0.024 |

### Sınıf bazlı mAP50

| Sınıf | Örnek sayısı | v1 | v2 | Δ |
|---|--:|--:|--:|--:|
| car | 14064 | 0.730 | 0.774 | +0.044 |
| motor | 4886 | 0.332 | 0.400 | +0.068 |
| pedestrian | 8844 | 0.323 | 0.382 | +0.059 |
| van | 1975 | 0.341 | 0.399 | +0.058 |
| people | 5125 | 0.256 | 0.305 | +0.049 |
| tricycle | 1045 | 0.213 | 0.260 | +0.047 |
| bus | 251 | 0.422 | 0.459 | +0.037 |
| truck | 750 | 0.309 | 0.322 | +0.013 |
| **awning-tricycle** | 532 | 0.116 | 0.134 | +0.018 |
| **bicycle** | 1287 | 0.066 | 0.086 | +0.020 |

**Bulgu:** Daha fazla epoch genel performansı anlamlı şekilde artırdı, ama kazanç eşit dağılmadı. `bicycle` ve `awning-tricycle` — modelin en zayıf iki sınıfı — en az iyileşen sınıflar arasında kaldı. Bu tamamen "az örnek = az iyileşme" ile de açıklanamıyor: `tricycle` (1045 örnek, bicycle'dan bile az) +0.047 ile çok daha fazla iyileşti. Yani örnek sayısı önemli bir faktör ama tek faktör değil — nesnenin görsel olarak ayırt edilebilirliği de rol oynuyor. Sonuç: **epoch artırmak tavan yapmış zayıf sınıfları kurtarmıyor, bir sonraki adım veri/augmentation odaklı olmalı** (bkz. Gelecek Çalışmalar).

Confusion matrix ve PR eğrileri: [`docs/eval/`](docs/eval/) (v1 ve v2 için ayrı ayrı).

## Proje Yapısı

```
Traffic-YOLO/
├── main.py                # İnference giriş noktası
├── config.yaml            # İnference varsayılanları
├── train_config.yaml      # Eğitim varsayılanları
├── visdrone_val.yaml      # Val split config (class-order remap dahil)
├── visdrone_full.yaml     # Train+val split config
├── colab_train.ipynb      # Colab'da çalışan eğitim notebook'u
├── src/
│   ├── cli.py             # main.py argparse tanımı
│   ├── config.py          # YAML + CLI config birleştirme
│   ├── detector.py        # YOLO model sarmalayıcı
│   ├── pipeline.py        # Video capture/writer döngüsü
│   ├── train.py           # Eğitim script'i
│   ├── evaluate.py        # Değerlendirme script'i
│   └── visdrone_convert.py # VisDrone → YOLO format dönüştürücü
├── models/
│   ├── best.pt             # v2 (güncel)
│   └── best_v1_baseline.pt # v1 (referans)
├── docs/eval/              # Confusion matrix + PR eğrisi görselleri
├── input/                  # Örnek test videosu
├── output/                 # Üretilen çıktılar (gitignored)
└── data/                   # Ham VisDrone verisi (gitignored, script'le indirilir)
```

## Kurulum

```bash
git clone https://github.com/loopBreakerr/Traffic-YOLO.git
cd Traffic-YOLO
pip install -r requirements.txt
```

## Kullanım

### İnference

```bash
python main.py --model models/best.pt --source input/test_video.mp4 --output output/result.mp4 --conf 0.5
```

Webcam için `--source 0`, pencere açmadan (headless) çalıştırmak için `--no-display` ekleyin. Tüm ayarlar `config.yaml` üzerinden de değiştirilebilir; CLI argümanları config'i geçersiz kılar.

### Değerlendirme

```bash
python -m src.evaluate
```

`models/best.pt`'yi VisDrone val split'i üzerinde çalıştırıp mAP/precision/recall tablosunu ve `docs/eval/` altına confusion matrix + PR eğrisi görsellerini üretir. Val verisi (`data/VisDrone/images/val`) yerelde bulunmalıdır; yoksa önce resmi VisDrone val zip'ini indirip `src/visdrone_convert.py` ile dönüştürmeniz gerekir (bkz. `colab_train.ipynb`'deki indirme/dönüştürme hücreleri — script kendi başına otomatik indirme yapmaz).

### Eğitim (Colab)

`colab_train.ipynb`'i [Colab'da açın](https://colab.research.google.com/github/loopBreakerr/Traffic-YOLO/blob/main/colab_train.ipynb), Runtime → GPU seçin, hücreleri sırayla çalıştırın. Önce 5 epoch'luk bir duman testi otomatik çalışır; geçtikten sonra son hücredeki `RUN_FULL_TRAINING = True` yapıp asıl eğitimi başlatabilirsiniz. Checkpoint'ler Google Drive'a yazılır, bağlantı kopmalarına karşı korumalıdır.

## Teknik Yolculuk ve Öğrenilenler

Bu proje ilk halinde tek dosyalık, hardcoded bir script'ti (~%30 mAP50, eğitim reproducible değildi). Yeniden ele alırken karşılaşılan ve çözülen sorunlar:

- **Sessiz sınıf-sırası hatası:** Modelin gerçek eğitildiği sınıf sırası (alfabetik) ile VisDrone'un resmi sınıf sırası farklıydı. Fark edilmeseydi değerlendirme metrikleri hatasız çalışıp sessizce yanlış sayılar üretecekti — kod hiç hata vermezdi. `visdrone_val.yaml`'da açık bir remap ile düzeltildi.
- **Warm-restart etkisi:** Var olan bir checkpoint'ten `resume=True` olmadan eğitime devam etmek, learning rate'i sıfırdan başlatıyor ve modeli geçici olarak zaten bulduğu iyi bir noktadan uzaklaştırıyor. İlk ~10 epoch'ta loss'un artıp mAP'in düşmesi bununla açıklandı; ~epoch 15'te toparlanma gözlemlendi.
- **Azalan getiri (diminishing returns):** Eğitim ilerledikçe epoch başına kazanç küçüldü (epoch 3→15 arası +0.0043/epoch, epoch 39→43 arası +0.002/epoch) — tipik bir eğitim eğrisi şekli, `patience` mekanizmasının var olma sebebi.
- **Temizlenmemiş checkpoint:** Colab bağlantısı 111. epoch'ta koptuğu için eğitim normal bitmedi, checkpoint otomatik "strip" edilmedi (18.3MB, optimizer+EMA dahil). Ultralytics'in `strip_optimizer` yardımcısıyla düzeltildi (6.0MB, sadece inference ağırlıkları).

## Bilinen Sınırlamalar

- Orijinal eğitimin notebook'u/tam hiperparametreleri kayıp; `train_config.yaml`'daki `batch`/`imgsz` değerleri makul varsayımlardır, orijinaliyle doğrulanmamıştır.
- `src/train.py` şu an gerçek `--resume` desteklemiyor — devam eden eğitimler yukarıda açıklanan warm-restart etkisini yaşıyor.
- v2 eğitimi 150 epoch'luk plana göre 111'de (Colab oturum kopması nedeniyle) durdu — algoritmanın kendi kararıyla değil, dışarıdan bir kesintiyle.
- `bicycle` ve `awning-tricycle` hâlâ modelin en zayıf sınıfları.

## Gelecek Çalışmalar

- [ ] `train.py`'ye gerçek `--resume` desteği eklemek
- [ ] Zayıf sınıflar için oversampling / hedefli augmentation denemek
- [ ] Daha büyük bir model (YOLOv8s) ile karşılaştırmak
- [ ] Nesne takibi (ByteTrack/DeepSORT) ile araç sayımı
- [ ] Basit bir Gradio/Streamlit demo
- [ ] Unit test + CI

## Veri Seti

[VisDrone2019-DET](https://github.com/VisDrone/VisDrone-Dataset) (Tianjin Üniversitesi) — 10 sınıf: pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor. Veri seti repoya dahil değildir; `colab_train.ipynb` çalıştırıldığında otomatik indirilir (yerel `src/evaluate.py` çalıştırması için verinin önceden `src/visdrone_convert.py` ile hazırlanmış olması gerekir).
