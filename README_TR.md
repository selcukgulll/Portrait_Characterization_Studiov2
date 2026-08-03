# Football Portrait Studio

Transfermarkt benzeri doğrudan fotoğraf URL'leri bulunan bir CSV'den oyuncu portrelerini indirir ve kaynak pozu mümkün olduğunca koruyarak karikatürize eder.

## Bu sürüm ne yapıyor?

- `player_id`, `name`, `image_url` sütunlarını varsayılan olarak tanır.
- Her oyuncunun görselini URL'den indirir.
- Stable Diffusion 1.5 image-to-image kullanır.
- IP-Adapter ile kaynak yüz ve görünüm rehberliği uygular.
- İsteğe bağlı özel stil LoRA yükler.
- Orijinal vesikalık kompozisyonunu korumak için kare padding uygular.
- Arka planı kaldırıp şeffaf PNG üretir.
- Dosyaları `player_id.png` olarak kaydeder.
- Hataları `report.csv` içine yazar.
- Var olan dosyaları atlayarak kaldığı yerden devam edebilir.

## Önemli gerçek

Ekteki stile gerçekten yaklaşmak için o stile ait bir **SD 1.5 LoRA** gerekir. LoRA olmadan uygulama genel bir çizgi-film/oyun-portresi görünümü üretir; yüklediğin örneğin çizgi dili birebir oluşmaz.

## Kurulum

### 1. Python

Windows için Python **3.11 x64** kur. Kurulumda `Add Python to PATH` seçeneğini aç.

### 2. Otomatik kurulum

Klasörde:

```bat
install_gpu.bat
```

çalıştır. NVIDIA GPU varsa CUDA destekli PyTorch kurulmaya çalışılır. Kurulum birkaç GB indirebilir.

### 3. Uygulamayı aç

```bat
run_app.bat
```

İlk üretimde Stable Diffusion ve IP-Adapter modelleri Hugging Face'ten indirilir. Daha sonraki çalıştırmalarda önbellekten açılır.

## İlk test ayarı

Önce yalnızca 3 oyuncu:

- Yüz benzerliği: `0.82`
- Karikatür agresyonu: `0.45–0.55`
- LoRA stil gücü: `0.75–0.95`
- Guidance: `6.5–7.5`
- Steps: `20–24`
- Boyut: `384` veya `512`
- Renk koruma: `0.55–0.70`
- Keskinlik: `1.10–1.30`

## Kontrollerin anlamı

### Yüz benzerliği
IP-Adapter ağırlığını artırır ve image-to-image dönüşümünü biraz daha kısıtlar. Çok yükselirse sonuç fotoğrafa fazla yaklaşabilir.

### Karikatür agresyonu
Diffusion `strength` değerinin temelidir. Yükseldikçe yüz, saç ve forma daha fazla yeniden çizilir. Pozun bozulmaya başladığı yerde azalt.

### LoRA stil gücü
Özel stil dosyasının etkisini ayarlar. `1.0` civarı çoğu LoRA için iyi başlangıçtır.

### Renk koruma
Kaynak görselin düşük frekanslı renklerini sonuca kısmen geri karıştırır. Forma renklerinin değişmesini azaltır.

### Seed
Aynı ayarlar ve aynı seed yaklaşık aynı sonucu verir. Her oyuncuya farklı seed seçeneği çeşitlilik sağlar.

## Özel LoRA

LoRA dosyanı örneğin:

```text
models\football_sticker_style.safetensors
```

konumuna koy ve uygulamada `Stil LoRA` alanından seç.

LoRA'nın SD 1.5 tabanlı olması gerekir; SDXL LoRA bu varsayılan modelle uyumlu değildir.

## EXE oluşturma

```bat
build_exe.bat
```

Sonuç:

```text
dist\FootballPortraitStudio\FootballPortraitStudio.exe
```

Bu `onedir` tipidir. Tüm `FootballPortraitStudio` klasörünü birlikte taşımalısın. Tek dosyalı EXE, PyTorch ve Qt nedeniyle aşırı büyük ve açılışta çok yavaş olur.

Model ağırlıkları EXE içine gömülmez; Hugging Face önbelleğinde tutulur.

## Çıktılar

```text
output/
├─ completed/
│  ├─ 10.png
│  └─ 26.png
├─ failed/
├─ sources/
└─ report.csv
```

## 4 GB VRAM notu

- 384 veya 512 çözünürlük kullan.
- Steps değerini 18–24 tut.
- Aynı anda başka GPU uygulamalarını kapat.
- Model CPU offload ile çalıştığı için üretim yavaş olabilir.
- `CUDA out of memory` alınırsa 384 px dene.

## Lisans ve kullanım

Model, LoRA ve kaynak fotoğrafların lisanslarını ticarî kullanım öncesi ayrı ayrı kontrol et.
