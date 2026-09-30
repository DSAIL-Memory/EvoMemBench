# Tez hikâyesi — "Küçük modellerin RAG başarımını, çelişen belleği ucuza temizleyerek yükseltmek"

> ## ⚠️ ÇERÇEVE AÇISINDAN AŞILMIŞTIR — 2026-09-17
>
> Tezin bağlayıcı çerçevesi artık **`THESIS_CLAIMS.md`**'dir (kullanıcı kararları,
> 2026-09-17). Bu dosya, `hnav_idonly` etrafında kurulmuş tarihsel tez hikâyesidir
> ve kayıt olarak korunur; aşağıdaki sayılar geçerlidir, ancak **tezin yöntemi
> H-Nav-GEO**'dur (ayrıştırıcısız, çıkarım anında LLM'siz geometrik ekran + NLI) ve
> tez karşılaştırması **yalnız native taban ile H-Nav-GEO** arasındadır. Beş model ×
> üç bağlam boyutunda tam bir deneme koşusu yapılmış ve H-Nav-GEO 15 hücrenin
> 15'inde tabanı geçmiştir; tezin manşeti budur. Çerçeve konusunda bu dosya ile
> `THESIS_CLAIMS.md` çelişirse `THESIS_CLAIMS.md` geçerlidir.


Bu belge, **hangi iddianın kurulabileceğini ve hangi kanıtla** kurulacağını
gösterir. Her sayı, depodaki taahhüt edilmiş (committed) bir artefakttan
gelir ve kaynağı yanında yazılıdır. Sonunda **iddia edilemeyecek** şeyler de
listelenmiştir — savunmada asıl korunması gereken kısım odur.

---

## 1. Tek cümlelik tez

> Gelişen (evolving) bir vektör belleğinde **çelişen olgular**, küçük bir
> dil modelinin RAG başarımını asıl sınırlayan etkendir; bu çelişkiler
> **çıkarım anında ek LLM çağrısı gerektirmeyen, hatta istemi kısaltan**
> ucuz bir geometrik+sembolik denetçiyle, **ölçülmüş sıfır bilgi kaybıyla**
> tespit edilip temizlenebilir ve bu, **2B–9B aralığındaki beş ayrı
> cevaplayıcı modelin** doğruluğunu benchmark'a göre **+5 ile +65 puan**
> arasında yükseltir (15 model-kol hücresinin 15'i pozitif).

Ölçek notu: alt sınır (+5) gemma-3-4b'nin sh_64k'sidir ve **anlamlı
değildir**; üst sınır (+65) Qwen3-4B'nin sh_6k'sidir. Kazanç aralığı geniş,
çünkü kazanç `varılan doğruluk − native`'dir ve native modelden modele
kendi nedenleriyle değişir. Tezde **kazanç aralığı değil, varılan doğruluk**
öne çıkarılmalıdır (§4).

Üç bileşen: (a) **problem gerçek**, (b) **çözüm ucuz**, (c) **çözüm güvenli**.
Aşağıda üçü de ayrı ayrı kanıtlanıyor.

---

## 2. Problem gerçek: küçük model doğru kaydı bulur, yanlış *sürümü* söyler

Bu, hikâyenin can damarı. Hata tipi rastgele değil, sistematik:

| kanıt | sayı | kaynak |
| --- | --- | --- |
| Çelişkili soru hatalarının **doğru anahtarın eski değerini** üretme oranı | **572 / 575** (8 taahhütlü koşu) | `stage0_results/question_strata.json`, `hnav/labeling/question_strata.py` |
| sh_6k'de çelişkili katmanda yerel (native) doğruluk | **2 / 74** | `stage0_results/stage1/detector_gap_retrieval_sh6k.json` |
| sh_64k'de çelişkili katmanda yerel doğruluk | **17 / 66** | `stage0_results/abtt/abtt_arm_A1_raw_sh64k.json` |
| Tekil (unique) katmanda yerel doğruluk — kontrol | 26/26, 35/35, 28/34 | aynı artefaktlar |

Yorum: model, bağlamda **hiç çelişki yokken** neredeyse kusursuz (26/26);
aynı bağlama aynı anahtarın eski bir sürümü eklendiğinde çöküyor (2/74).
Yani başarısızlık bilgi eksikliği değil, **sürüm ayrımı yapamama**. Küçük
modelin zayıflığı tam olarak budur ve tezin hedeflediği boşluk budur.

---

## 3. Çözüm ucuz — üç ayrı maliyet ekseninde

### 3.1 Çıkarım anında ek üretim (generative) çağrısı: **tam olarak sıfır**
Denetçi sayfayı düzenler ve **yerel RAG ile aynı tek sohbet çağrısını** yapar
— sadece istem daha kısadır. Kanıt: çevrimiçi katmanda (`hnav/core/`,
`hnav/adapters/`) hiçbir sohbet/completions çağrı yolu yoktur; oradaki tek
ağ çağrısı gömme uç noktasıdır (`hnav/core/embedding.py:342-353`).

**Bir kez, çevrimdışı ödenen bedel:**

| kalem | miktar | not |
| --- | --- | --- |
| Olgu + sorgu gömmeleri | sh_64k için **4,680** vektör, çalışma anında **0 cache kaçağı** | `ABTT_REPORT.md:56`; gerçek dağıtımda bunu vektör deposu zaten yazma anında öder |
| NLI ön-geçişi (çapraz kodlayıcı) | **410 / 4,868 / 8,956** ileri geçiş (sh_6k / sh_32k / sh_64k), 435M parametreli DeBERTa | 100 soru için; **tek bir sorunun cevaplayıcı-model ön-dolum maliyetinin ≈ %1.4'ü** |
| Eşik kalibrasyonu | yalnızca sh_6k+sh_32k, **LLM'siz, altın cevapsız** | amaç fonksiyonu yalnız tespit kalitesi |

Kritik olan: bu üç kalem de **cevaplayıcı modelden bağımsızdır**; yeni bir
model denemek ön-geçişi yeniden kurmaz (`pipelines/README.md`).

### 3.2 İstem uzunluğu: **negatif maliyet** (kısalıyor)

| koşu | istem karakter değişimi | kaynak |
| --- | ---: | --- |
| hnav_raw sh_64k | **−0.31 %** (−48,625 karakter) | `abtt_arm_A1_raw_sh64k.json` → `tokens.detector_suppress` |
| hnav_geo sh_64k | −0.22 % | `pipelines/hnav_geo/results/.../detector_gap_sh_64k.json` |
| hnav_geo sh_6k | **−2.87 %** | aynı klasör |

Denetçi olgu siliyor, eklemiyor: doğruluk artarken **token maliyeti düşüyor**.
Bu, "ucuz" iddiasının en güçlü tek cümlesi.

### 3.3 Denetçinin kendi hesabı: sayfa başına, sabit ve küçük
Soru başına ≤ 50 olgudan oluşan havuzda ikili kosinüs + QR artığı + tablo
araması. Gömme vektörleri zaten RAG için hesaplanmış durumda; ek maliyet
2560-boyutlu birkaç bin iç çarpım. GPU gerekmiyor (bu oturumdaki tüm
çözümleme dizüstü CPU'da koştu).

---

## 4. Çözüm işe yarıyor — **beş cevaplayıcı modelde, 15 kolun 15'inde**

Çoklu-model kampanyası tamamlandı (2026-08-30/31, ~19.500 tamamlama).
Değişen tek şey **cevaplayıcı model**; bellek, erişim, ön-geçişler,
baskılama planları, istemler, üretim ayarları ve puanlama dondurulmuş.
Her hücre **tek atış**, soru-eşlemeli McNemar, `page_source=benchmark`.

Tablo elle yazılmadı: `pipelines/MULTIMODEL_SUMMARY.md`
artefaktlardan üretilir (`hnav/geometry_filter/multimodel_summary.py`).

**Genel doğruluk /100 — `hnav_idonly` (en iyi kol):**

| model | sh_6k | sh_32k | sh_64k (tutulan) |
| --- | --- | --- | --- |
| google/gemma-3-4b-it | 45 → **89** (+44) | 38 → **52** (+14) | 33 → **38** (+5, a.d.) |
| google/gemma-4-E2B-it | 40 → **83** (+43) | 44 → **63** (+19) | 37 → **45** (+8) |
| microsoft/Phi-4-mini-instruct | 40 → **89** (+49) | 50 → **72** (+22) | 46 → **57** (+11) |
| Qwen/Qwen3-4B-Instruct-2507 | 30 → **95** (+65) | 53 → **85** (+32) | 45 → **66** (+21) |
| Qwen/Qwen3.5-9B | 39 → **99** (+60) | 61 → **92** (+31) | 51 → **69** (+18) |

15 hücrenin **15'i pozitif**; sh_64k'de gemma-3 dışında hepsi anlamlı.
`hnav_raw` her modelde `hnav_idonly`'nin 0–2 puan altında; `hnav_geo` her
modelde ikisinin de altında **ve** koşul 4'ten **VOID** (§5).

### İki yanlış iddianın düzeltilmesi

Bu kampanya sırasında iki kez fazla iddia ettim; ikisi de veriyle çürüdü.

1. **"Kazanç küçük/zayıf modelde en büyüktür."** Yanlış. sh_6k'de üç
   modelin native'i ~40 iken kazanç model gücüyle *artıyordu*, ama beş
   modelle bakınca ilişki yok: native 30 olan Qwen3-4B **+65**, native 45
   olan gemma-3 **+44** alıyor.
2. **"Kazanç model gücüyle birlikte artar."** Bu da genel olarak yanlış.
   sh_6k'de kazanç native ile **ters** ilişkili — çünkü tavan sıkışık
   (83–99), kazanç büyük ölçüde `tavan − native`.

**Veriyle savunulabilen ifade, kazanç değil *varılan doğruluk*:**

| bağlam | H-Nav sonrası doğruluk (küçükten büyüğe) | yayılım |
| --- | --- | --- |
| sh_6k | 83, 89, 89, 95, 99 | 16 puan |
| sh_32k | 52, 63, 72, 85, 92 | 40 puan |
| sh_64k | 38, 45, 57, 66, 69 | 31 puan |

Varılan doğruluk **her üç bağlam uzunluğunda da model yeteneğiyle
monoton** ve yayılım bağlam büyüdükçe açılıyor. Mekanizma bunu açıklıyor:
H-Nav bayat olguları sayfadan çıkarır, ama modelin **temizlenmiş sayfayı
okuyup yanıtlaması** hâlâ gerekir. Yani denetim, modelin *zaten sahip
olduğu* yeteneği doğruluğa çevirir — çevrilecek yetenek olmalıdır.

Tez çerçevesi bu yüzden "zayıf modeli kurtarır" değil, **"gizil yeteneği
ucuza doğruluğa çevirir"** olmalıdır.

**Neden bağlam büyüdükçe kazanç azalıyor?** Çünkü mekanizma doyuyor ve
darboğaz **erişime (retrieval)** kayıyor — §6.

### Ölçüm aracı da bir sonuçtur: gemma-3 / fp8 vakası

gemma-3-4b ilk kez, kampanyanın referans modelden **devraldığı**
`--kv-cache-dtype fp8` ile koşturuldu ve **13/100 (sh_6k)** aldı; çıktısı
`"United States of United States of United States of United"` idi. Aynı
ağırlıklar, aynı istemler, aynı baskılama planı, sadece önbellek tipi
BF16 olunca **89/100**. 76 puanlık fark tamamen sunum ayarından geliyordu.

Bu koşu silinmedi; `*_VOID_fp8_kv/` altında `VOID.md` ile birlikte duruyor.
Metodolojik ders: **yapılandırma kökeni deneyin parçasıdır**; "çalıştı ve
akıcı metin üretti" bir aracın doğru çalıştığının kanıtı değildir.
O koşu, o zamanki tüm ön-uçuş kontrollerinden geçmişti — çünkü hiçbiri
yanıtların **doğru** olup olmadığına bakmıyordu.

**Oracle'a yakınlık:** sh_6k'de kusursuz baskılama oracle'ı 0.91, denetçi
0.90 (oran 0.984); sh_32k'de 0.88 vs 0.86 (0.957)
(`stage0_results/stage1/stale_suppression_probe_*.json`, `hnav/BUILD_NOTES.md`).
Yani denetçi, mekanizmanın verebileceğinin **%96–98'ini** veriyor.

---

## 5. Çözüm güvenli — ve bu, yapısal olarak kanıtlı

- **Sıfır zarar kuralı**: hiçbir baskılanan olgu, anahtarının **güncel**
  değerini taşıyamaz. Tüm taahhütlü kollarda kalibrasyonda 0 zararlı
  (`classify_drops`, `detector_gap.py:407-444`).
- **Yapısal kanıt (E2E-4)**: her doğrulanan çift `same_key` şartını
  sağlıyorsa zarar **inşaat gereği** sıfırdır — `same_key` (relation,
  subject) üzerinde bir denklik bağıntısıdır, grup iki anahtara yayılamaz;
  `suppress` her grubun en yüksek serisini korur; dolayısıyla anahtarın en
  yeni değeri asla düşmez. Kalibrasyon taraması bunu ızgaranın **her
  hücresinde** doğruladı (`IDONLY_PREREG.md`).
- **Ve bu, tezin en keskin bulgusu**: geometrik ekranda böyle bir garanti
  **yoktur**. `hnav_geo` kolunda kalibrasyonda zarar 0 iken sh_64k'de
  **8 zararlı baskılama** oluştu ve koşu kendi ön-kayıtlı 4. koşuluyla
  **geçersiz (void)** sayıldı: geometrik grup iki farklı anahtarı
  birleştirdi ve baskılama bir anahtarın **tüm üyelerini** sildi
  (`Italy · resmî dil`, `The Game · müzik türü`, `outfielder · spor`).
  Dahası, geometrinin parser'dan **fazladan** yaptığı 5 baskılamanın
  **tamamı** bu zararlı olanlardır — 5'te 5. Yani geometrinin sembolik
  kimliğe kattığı tek şey bilgi kaybıdır. Aynı kusur, taahhütlü
  `hnav_abtt_noparser` kolunda da vardır (5 zararlı, koşu geçersiz).
  **Sembolik kimlik burada bir tercih değil, güvenlik koşuludur.**
- **Korunan katman**: tekil (unique) katmanda kayıp yok (26/26, 34/35,
  27/34 — tek kayıp q77, düzenlenmiş sayfada modelin reddi).
- **A/A tabanı 0**, sızıntı denetimi, sayfa-düzenleme bütünlüğü, pozitif
  kontrol: her koşuda temiz.

---

## 6. Dürüst sınır: tavan nerede ve neden

sh_64k'de parser kolunun 29 çelişkili hatasının ayrıştırması
(`E2E4_COMPLEMENTARITY.md` §6):

| neden | adet | düzeltilebilir mi |
| --- | ---: | --- |
| **erişim kaybı** — altın olgu sayfaya hiç gelmiyor | **22** | hayır (sayfa düzenlemesiyle imkânsız) |
| **parametrik bilgi** — sayfa zaten doğru, model kendi ağırlıklarından yanıtlıyor | 5 | hayır |
| **tespit kaybı** | **2** | evet |

Dolayısıyla **herhangi bir baskılama tabanlı denetçinin mutlak tavanı
72/100**'dür ve parser kolu 64 ile bu tavanın 2 tespit hatası uzağındadır.
Bu, "neden 64'ten sonrası zor" sorusunun kanıtlı yanıtıdır ve tezde
zayıflık değil, **olgunluk** göstergesidir.

---

## 7. Ne iddia edilebilir / ne edilemez

**Edilebilir**
1. Çelişen bellek, küçük modelin RAG başarımında birinci dereceden bir
   engeldir (572/575 hata sistematiği).
2. Bu engel, çıkarımda ek LLM maliyeti olmadan, istemi kısaltarak,
   ölçülmüş sıfır zararla büyük ölçüde kaldırılabilir (+19 … +66).
3. Denetçi, mekanizmanın oracle tavanının %96–98'ini yakalar.
4. Kalan hata, **denetçinin değil erişimin** hatasıdır (22/29).
5. Kimlik (identity) kararı için sembolik ayrıştırıcı ile gömme geometrisi
   **birbirini tamamlamaz**; geometri sembolik kümenin %99'unun alt
   kümesidir (E2E-3, E2E-4).

6. **Etki tek modele özgü değildir** — beş cevaplayıcı model (2B-sınıfı
   gemma-4-E2B'den 9B Qwen3.5'e; dört mimari, iki vLLM sürümü), 15 kolun
   15'inde pozitif. Bu, §8'de "sıradaki hamle" diye yazılan kampanyanın
   sonucudur; artık yapılmıştır.
7. **Baskılama planları cevaplayıcı modelden bağımsızdır** — `hnav_geo`
   sh_64k'de beş modelin **beşinde de** `n_suppressed_harmful = 8` ve
   `n_suppressed_superseded = 524`, bayt bayt aynı. Bu bir çıkarım değil,
   beş kez tekrarlanmış bir ölçüm.

**Edilemez (savunmada bunları kendin söyle)**
- ~~"Her modelde geçerlidir"~~ → artık **beş modelde** ölçüldü (yukarıda
  madde 6). Ama hâlâ **beş modeldir**, "her model" değil; hepsi 2–9B
  yoğun/hibrit İngilizce yönerge modelleridir.
- "Kazanç modelin zayıflığıyla ölçülür" — **çürütüldü** (§4). Ne zayıf
  model lehine ne de güçlü model lehine basit bir eğri vardır; monoton
  olan şey *varılan doğruluk*tur, kazanç değil.
- "Her gömücüde geçerlidir" — eşikler Qwen3-Embedding-4B uzayının
  koordinatlarıdır; G1 bunların taşınmadığını ölçtü.
- "Anlamsal doğrulama gereksizdir" — yalnızca **tek değerli bağıntılı**
  bir depo için gösterildi; çok değerli bağıntılarda NLI kapısı gerçek iş
  yapar.
- "sh_262k'ye genellenir" — dışlandı (bağlam penceresi karşılaştırmayı
  bozar).
- "Geometri parser'ın yerini alabilir" — alamaz: 64 → 56 (p = 0.021).

---

## 8. Çoklu-model kampanyası — **yapıldı**, iki sorunun da yanıtı var

Bu bölüm bir plandı; 2026-08-30/31'de yürütüldü. Beş cevaplayıcı model,
üç kol, üç alt küme, hepsi tek atış. Sorulan iki soru:

**1. Yönetişim kazancı modeller arası tekrarlanıyor mu?** **Evet** — 15
kolun 15'i pozitif (§4). Bu artık §7'de "edilebilir" tarafında.

**2. Parser'ın üstünlüğü aynı sorularda mı yoğunlaşıyor — yapısal mı,
model etkileşimi mi?** **Yanıt: ikisi de, ve ayrıştırılabiliyorlar.**

`hnav_idonly` ile `hnav_raw`'ın istemlerinin *farklılaştığı* sh_64k soru
kümesi, iki tamamen farklı modelde **bayt bayt aynı** çıktı:

```
[5, 8, 15, 20, 23, 38, 49, 50, 52, 55, 58, 59, 73, 82, 85, 88, 98]   — 17 soru
```

Bu **yapısal**: baskılama planında LLM yoktur, dolayısıyla fırsat kümesi
modelden bağımsızdır ve artefaktlardan model çalıştırmadan kanıtlanabilir.

Bu 17'nin *kaçının doğru cevaba dönüştüğü* ise **model etkileşimidir**:

| model | `idonly`'nin kazandığı | kaybettiği | net |
| --- | --- | --- | --- |
| Qwen3-4B | {23, 98} | — | +2 |
| Phi-4-mini | {98} | {55} | 0 |
| gemma-4-E2B | {23, 59} | — | +2 |
| Qwen3.5-9B | {82, 98} | — | +2 |

Adı geçen her soru 17'nin içinde; hiçbir model bu kümenin dışından bir
soru kazanmadı ya da kaybetmedi. Dört modelde **264 model-soru fırsatında
tek bir bozulma** (Phi-4-mini/q55).

**Dürüst istatistik uyarısı:** bu dört sonucu tek bir p-değerinde
havuzlamayın. Dört model **aynı 100 soruyu**, **aynı planlarla**
yanıtlıyor; bağımsız örnek değiller ve naif havuzlama gücü olduğundan
büyük gösterir. Savunulabilir ifade yön ve tutarlılıktır: 5/5 modelde
`idonly ≥ raw`, ve mekanizma yapısal olarak kanıtlı.

Bu ayrıştırma iddiayı da **sınırlar**: `hnav_idonly`, sh_64k'de hangi
model olursa olsun `hnav_raw`'ı **17 sorudan fazla** geçemez, ve gözlenen
dönüşüm bunun küçük bir kesridir (0–2, %6–12).

### Sıradaki hamle (yeni)

Artık açık olan boşluk **gömücü (embedder)**: tüm eşikler
Qwen3-Embedding-4B uzayının koordinatlarıdır ve G1 bunların taşınmadığını
ölçtü. İkinci bir gömücü, yeni bir kalibrasyon kampanyası demektir —
bir koşturucu bayrağı değil.
