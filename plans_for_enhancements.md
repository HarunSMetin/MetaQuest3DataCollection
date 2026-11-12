## Önce teşhis: "darboğaz" tek bir şey değil

Bildiklerimiz:
- Model tutuşta ve yerleştirmede ~1–2 cm hata yapıyor. Demolarda bu 4–5 mm.
- Tuğla konumlarını doğrudan verdiğimiz oracle bile %22'de kaldı.
- tower_v3'te model, **kendi eğitim karelerinde** bile 3–15 mm yanlış aksiyon üretiyordu. Bu önemli bir ipucu.

Hata üç ayrı yerden geliyor olabilir. Çözümleri de farklı:

| Hipotez | Ne demek | Nasıl test edilir | Çözüm |
|---|---|---|---|
| **H1. Model veriyi yeterince ince öğrenemiyor** | Doğru durumda bile mm hassas aksiyon veremiyor | Eğitim ve doğrulama karelerinde tahmin edilen hedefi TCP'ye çevirip mm hatasını ölç. ~30 dk, GPU'yu az kullanır | Aksiyon temsili, EMA |
| **H2. Hata birikimi** | Demolarda hiç görmediği durumlara (kaçık tutuş, havada bekleme) düşüyor | Rollout'larda sapmanın nerede başladığına bak. Kısmen gördük: kaçık tutuştan sonra çöküyor | DAgger, düzeltme verisi |
| **H3. Algı** | Tuğlayı veya hizayı mm seviyesinde göremiyor | Oracle %22: algı tek sebep değil, ama payı olabilir | Bilek kamerası çözünürlüğü, 3D |

Benim tahminim: **H1 + H2 birlikte.** H1'e özellikle dikkat çekmek istiyorum. Mutlak eklem açısı tahmin ediyoruz. Bütün veri aralığı [-1, 1]'e normalize ediliyor ve 1 mm TCP hareketi kabaca 0.001 rad'a denk geliyor. Normalize uzayda bu çok küçük bir fark. Model "kabaca doğru" yere gidiyor, ama son milimetreleri çözünürlük olarak ifade etmekte zorlanıyor. Kendi eğitim karesinde 3–15 mm hata bunun belirtisi.

## Plan A: şimdi, düşük maliyetli

1. **H1 ölçümü.** 60k modeli için eğitim ve doğrulama karelerinde mm hatası. Bu, B'deki aksiyon temsili değişikliğinin gerekip gerekmediğini söyler.
2. **Checkpoint ortalaması (EMA'nın bedava testi).** 60k ve 80k ağırlıklarının ortalamasını alıp değerlendiririz. Eğitim gerekmez, ~25 dk sürer. Fark görürsek EMA'yı kesin ekleriz.
3. **DAgger** (sen kaydediyorsun). Kaçık tutuşta ve havada beklerken yaptığın düzeltmeler.
4. Sonra **v4 + DAgger** verisiyle, EMA açık bir eğitim.

**Beklenti:** Seviye 1 %14 → %30–50 arası olursa doğru yoldayız.

## Plan B: A yetmezse

- **Göreli aksiyonlar.** Mutlak eklem hedefi yerine "şu anki pozdan şu kadar git". UMI ve π0 bu yolu kullanıyor. Normalizasyon küçük hareketlere ayrılır, mm çözünürlüğü artar. H1 ölçümü hatayı büyük gösterirse bu ilk değişiklik olur. Kod değişikliği `FeatureSpec` ile `PolicyRunner` içinde, 1 gün kadar.
- **İnce faz için ayrı görüş.** Son yaklaşımda bilek kamerasından yüksek çözünürlüklü bir kırpım. 240×320'de tuğla birkaç düzine piksel. Daha keskin görüntü hizalamaya yardım eder.
- **2 gözlem karesi** (orijinal Diffusion Policy ayarı). Hareketin yönünü daha iyi görür, ama batch yarıya iner.

## Plan C: güçlü yedek, simülasyonun avantajını kullanmak

Elimizde kimsenin kullanmadığı bir şey var. `scripts/make_media.py`, gerçek IK ve gripper ile kuleyi baştan sona kuruyor ve başarıyor. Yani **tuğla konumlarını bilen, script'li bir uzman** hazır.

- **Otomatik DAgger ("teacher-student").** Bu script'i, her durumdan "şimdi ne yapmalı" diyebilen reaktif bir uzmana çeviririz. Policy simde koşar, uzman her karede doğru aksiyonu etiketler, policy bununla yeniden eğitilir, döngü tekrar eder. **İnsan gerekmez, sınırsız düzeltme verisi** üretilir. Literatürde sim tabanlı hassas işlerin standart çözümü bu (ör. "Learning by Cheating").
  - Artı: H2'yi tamamen hedefliyor, en hızlı ölçeklenen yol.
  - Eksi: Etiketler insan stilinden script stiline kayar. Önceden insan demosunu tercih etmiştin. İnsan demolarıyla karıştırarak dengelenebilir.
  - İş: ~2–3 gün.
- **Residual RL (Ankile 2024).** Diffusion policy donmuş kalır. Üstüne küçük bir ağ eklenir, PPO ile simde "seviye tıkladı mı" ödülüyle son milimetreleri düzeltmeyi öğrenir. Hassas montajda literatürdeki en büyük sıçrama bundan geldi.
  - Eksi: RL ayarı zahmetli. 6 GB GPU'da render ederek paralel sim yavaş olur, residual ağı state ile eğitip sonra damıtmak gerekebilir.

## Plan D: o da olmazsa

- **Hibrit kontrol.** Policy tuğlayı ~1 cm yakına getirir, son adım klasik bir oturtma primitifi olur: kuvvet kontrollü bastırma ve küçük spiral arama. Stud'lardaki pah (chamfer) birkaç mm hatayı zaten merkezliyor. Endüstride insertion'ın çoğu böyle çözülür. Saf öğrenme değil, ama güvenilir.
- **Görevi küçültüp hattı doğrulamak.** "Tek tuğlayı al, belirlenen noktaya koy" hassasiyeti bile tutmuyorsa sorun görevde değil, öğrenme hattının kendisindedir. Ona göre geriye gideriz.

## Karar ağacı

```
H1 ölçümü ─┬─ eğitim karesinde hata > 5 mm ─→ Plan B (göreli aksiyon) + EMA
           └─ küçük ──────────────────────→ sorun hata birikimi: DAgger
DAgger + EMA sonrası ─┬─ ≥ %50 ─→ daha fazla DAgger, sonra 3–4 tuğlaya geç
                      └─ < %30 ─→ Plan C (otomatik DAgger)
Plan C de takılırsa ─→ residual RL ya da Plan D (hibrit)
```

## EMA tam olarak ne yapıyor?

Eğitim sırasında ağırlıklar her adımda biraz oynar. Her batch farklı olduğu için iyi bir bölgenin etrafında "titrer". EMA bu ağırlıkların hareketli ortalamasını ayrı bir kopyada tutar:

```
θ_ema ← 0.9999 · θ_ema + 0.0001 · θ      (her eğitim adımında)
```

Değerlendirmede bu ortalama kopya kullanılır. Etkisi, son ~10 bin adımın ağırlıklarının ortalamasını almak gibidir.

**Neden önemli:** Titreşim yüzünden her checkpoint biraz farklı hatalar yapar. Bizdeki 60k → 80k oynaklığı da muhtemelen bundan (taban %84 → %68). Ortalama, iyi bölgenin merkezine daha yakındır. Diffusion'da etkisi daha büyük, çünkü gürültü giderme 10 adım tekrarlanıyor ve her adımdaki küçük hata birikiyor. Görüntü diffusion modellerinde EMA standart. Orijinal Diffusion Policy de kullanıyor. LeRobot'ta ise yok.

**Bize katkısı:** Dürüst beklentim birkaç puanlık iyileşme ve checkpoint'ler arasında daha tutarlı sonuçlar. Tutuşları 2 cm'den 5 mm'ye indirmez, darboğazın çözümü değil. Maliyeti çok düşük olduğu için yine de açık olmalı. Tek sorun ağırlıkların bir kopyası daha (~1 GB); 6 GB GPU'da sığmazsa CPU'da tutulabilir.

---

İstersen DAgger'a geçmeden önce Plan A'nın 1. ve 2. maddelerini (H1 ölçümü ve checkpoint ortalaması) başlatayım. Toplam ~1 saat sürer, sen DAgger kaydına başlamadan sonuç çıkar.