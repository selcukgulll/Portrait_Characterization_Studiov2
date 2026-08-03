# Football Portrait Studio v1.3 — Son İşlem V1

## Eklenenler

- Modeli tekrar çalıştırmadan sonuç görselini düzenleyen `Son İşlem` sekmesi
- Son üretilen sonucu otomatik editöre aktarma
- Haricî PNG/JPG/WebP açma
- Dört arka plan kaldırma yöntemi:
  - AI — ISNet
  - AI — U2Net
  - Kenar rengine göre
  - Beyaz zemine göre
- Maske ayarları:
  - tolerans
  - feather
  - büyüt/küçült
  - dehalo
  - en büyük bileşeni koru
  - delikleri doldur
- Parlaklık, kontrast, doygunluk, keskinlik, ayrıntı, hue ve gamma
- Siyah dış kontur: kalınlık, yoğunluk ve yumuşaklık
- İsteğe bağlı iç çizgi güçlendirme
- Şeffaf checkerboard önizleme
- Hazır presetler
- Tekli kayıt ve klasöre toplu uygulama

## Önerilen akış

1. `Toplu Üretim` sekmesinde **Üretimde arka plan kaldır (eski yöntem)** kapalı olsun.
2. Portreyi üret.
3. `Son İşlem` sekmesine geç.
4. Arka plan için önce:
   - düz beyazsa `Beyaz zemini temizle`
   - düz siyah/renkliyse `Kenar rengini temizle`
   - karmaşıksa `AI — ISNet`
5. `Game Sticker` presetini uygula.
6. Kontur ve renk değerlerini küçük adımlarla düzelt.
7. `Postprocessed PNG Kaydet`.

## Not

AI arka plan modelleri ilk kullanımda rembg model dosyası indirebilir. İnternet yoksa
Beyaz/Kenar rengi modları çalışmaya devam eder.
