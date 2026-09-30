# TEZ BULGULARI — Kanıt Defteri

> Amaç: bugün **savunulabilir** olan her bulgunun tek yerde, kanıt dosyasıyla,
> sınırlılığıyla ve durumuyla listesi. Dış denetime (danışman / hakem / harici
> model) verilecek hâli budur: her sayının ham kaynağı gösterilir.
> Son güncelleme: 2026-08-15. Kod durumu: 315 test yeşil.
>
> **Durum etiketleri:** `KESİN` = ölçüldü, bağımsız doğrulandı, düzeltme
> beklemiyor · `SAĞLAM` = ölçüldü ve denetlendi · `GEÇİCİ` = 512-token kesme
> düzeltmesi sonrası **yeniden türetilecek** (chunk düzeyi gömme türevleri) ·
> `AÇIK` = ölçüm sırada.

---

## A. Ana bulgu — arenanın doğruluğu çakışmasız sorulardan geliyor

**İddia.** MemoryAgentBench `Conflict_Resolution` arenasında manşet doğruluk,
çakışma **içermeyen** sorulardan geliyor; çakışan sorularda model, istem açıkça
"büyük seri numarası daha yeni" dese de **bayat (superseded) değeri** üretiyor.

| ölçüm | sh_6k | sh_32k | sh_64k | sh_262k |
|---|---|---|---|---|
| çakışmayan soru (n) | 26 | 35 | 34 | 22 |
| çakışan soru (n) | 74 | 65 | 66 | 76 (+2 eşleşmeyen) |
| **çakışmayan doğruluk** | **26/26 — 8 koşunun 8'inde** | — | — | — |
| **çakışan doğruluk** | **0–5 / 74** | — | — | — |
| manşet doğruluk (m3) | 0.330 | 0.470 | 0.440 | 0.200 |
| **çakışan-only doğruluk** | **0.095** `estimate` | **[0.185, 0.723]** `bound` | **[0.152, 0.667]** `bound` | **[0.000, 0.263]** `bound` |

> **Neden yalnız sh_6k nokta tahmin:** "çakışmayan soru hep doğru" öncülü
> **yalnız sh_6k'da ölçüldü** (26/26, 8 koşu). Diğerlerinde öncül ölçülmedi,
> sh_262k'da ise **çürütüldü** (`assumption_refuted: true`; öncül doğru olsaydı
> ima edilen değer negatif çıkardı — yani öncülün yanlışlığının kanıtı).
> Varsayımsız savunulabilir ifade **aralıktır**; sh_262k için çakışmayan
> doğruluk üst sınırı 20/22 = 0.909. Negatif bir olasılık yayımlamak tüm
> ekstrapolasyonu haklı olarak şüpheye açardı — bu yüzden `question_strata.json`
> `kind: estimate|bound` alanını **yapısal** olarak taşır, negatif değer
> üretilemez. Ayrıca ima/sınır satırları m3 harness istemiyle ölçülmüştür
> (benchmark'ın şablonlu sorgusu değil); doğrudan ölçüm yalnız sh_6k'dadır.

**Hata taksonomisi (8 koşu, 575 çakışan-soru hatası):** 572 `stale_value`,
3 `off_list`, **0 boş**. Model bağlamı okuyor; kuralı uygulamıyor.

- **Kanıt:** `stage0_results/question_strata.json`, `hnav/labeling/question_strata.py`,
  `stage0_results/t4_s2_evidence/sh_6k_{off,offA,offB,shadow,detA..detD}_results.json`.
  Yeniden hesaplanan notlar koşuların kendi `substring_exact_match` alanıyla **800/800** uyuşuyor.
- **Doğrulama:** iki kez bağımsız (orkestratör betiği → üretim modülü); modülde
  negatif kontroller (yanlış etiketli fixture, "aynı değer iki kez ≠ çakışma",
  "başka anahtarın değeri ≠ stale") ve `gold_rule.py` sırasıyla çapraz oracle.
- **Sınırlılık:** doğrudan ölçüm yalnız sh_6k'da (diğer üç subset için
  çakışmayan-soru-hep-doğru varsayımıyla ima edilir; sh_262k'da varsayım **ihlal
  ediliyor** ve dosyada işaretli). Tek model (Qwen3-4B-Instruct-2507), tek arena.
- **Durum: KESİN.**

**Neden önemli:** (1) bu arenayı kullanan her çalışmanın manşet sayısı büyük
ölçüde çakışmasız soruları ölçüyor; (2) açık talimatla verilen supersession
kuralı ~%95 işlemiyor — bellek yönetişimi için doğrudan motivasyon; (3) tavan
gerçek ve büyük (sh_6k'da 100 sorunun 71'i yanlış, **tek** hata kipi).

**Açık ayrım (probe ölçecek):** bayat değeri vermesinin nedeni *konum/varlık*
mı (bağlamdaki bayat kayda tutunma) yoksa *parametrik öncelik* mi (dünya
bilgisinin bağlamı ezmesi — bayat değerler çoğunlukla dünya-doğrusu). Bu ayrım,
herhangi bir okuma-yolu müdahalesinin işe yarayıp yaramayacağını belirler.
`hnav/stage1/stale_suppression_probe.py` (yazıldı, 938 çağrı, kutu bekliyor).

---

## A2. TAVAN ÖLÇÜLDÜ — bayat kaydı bastırmak çakışan soru doğruluğunu 5–10× artırıyor

**İddia.** A'daki hata kipi **düzeltilebilirdir**: bağlamdan bayat kayıt
çıkarıldığında model doğru cevabı veriyor. Yani başarısızlık "modelin dünya
bilgisini bağlama tercih etmesi" (parametrik öncelik) DEĞİL, **bağlamdaki bayat
kaydın varlığına/konumuna tutunmadır**.

Oracle probe, gerçek model + benchmark'ın kendi istemi + dondurulmuş `:8003`
substratı, yalnız kalibrasyon split'i (938 çağrı):

| kol | sh_6k genel | sh_6k çakışan | sh_32k genel | sh_32k çakışan | McNemar |
|---|---|---|---|---|---|
| native | 0.290 | 4/74 (%5.4) | 0.420 | 7/65 (%10.8) | — |
| A/A tabanı (native_repeat) | 0.290 | 4/74 | 0.420 | 7/65 | **0/0 uyuşmazlık, iki subset'te de** |
| **oracle_suppress** (bayatı sil) | **0.910** | **66/74 (%89.2)** | **0.880** | **53/65 (%81.5)** | +62 (p=4e-19) · +46 (p=3e-14) |
| **oracle_recency** (LATEST'i sona al) | 0.460 | 20/74 (%27.0) | 0.680 | 33/65 (%50.8) | +17 (p=8e-05) · +26 (p=9e-07) |
| anti (LATEST'i başa al) | 0.260 | 1/74 (%1.4) | 0.380 | 4/65 (%6.2) | −3 · −4, anlamsız |

- **Kanıt:** `stage0_results/stage1/stale_suppression_probe_{sh6k,sh32k}.json`,
  `hnav/stage1/stale_suppression_probe.py` (34 test).
- **Koruyucu koşul sağlanıyor:** çakışmayan katman iki yardım kolunda da
  **bozulmuyor** (25/26 ve 35/35 korunuyor) → bu katmanda zarar sıfır.
- **`oracle_recency` token-nötrdür** — hiçbir bilgi silinmez, yalnız konum
  değişir; yine de çakışan doğruluğu 5× (sh_6k) ve 4.7× (sh_32k) artırır.
- **Mekanizma:** çapa **geç konumdur**. LATEST'i sona almak yardım ediyor, başa
  almak (anti) zarar veriyor. Bu, T11'de chunk düzeyi **yukarı** rerank'in neden
  sistematik zararlı olduğunu da açıklar (bkz. `STAGE1_NULL_ANALIZI.md`):
  superseder'ı yardım eden konumdan **uzaklaştırıyordu**, üstelik ~250 olgu
  taşıyan bir chunk granülerliğinde.
- **Sınırlılık (kritik):** bu kollar **oracle**'dır — sorunun anahtarını gold ile
  belirler. Sevk edilebilir politika yalnız dedektör çıktısını kullanabilir;
  **oracle→dedektör boşluğu** ayrıca ölçülmelidir (T13, koşuyor). Tavan budur,
  elde edilen değil.
- **Durum: KESİN (tavan olarak).** Kalibrasyon split'inde iki kez replike;
  sh_64k/sh_262k'ya dokunulmadı.

---

## A3. TAVANIN ~%96–98'i DEDEKTÖRLE, GOLD OLMADAN YAKALANDI

**İddia.** A2'deki tavan oracle'dı (anahtar gold ile belirleniyordu). Aynı
müdahale, **yalnız dedektör çıktısıyla** (gold yok, cevap yok, gelecek olgu yok)
sürüldüğünde tavanın neredeyse tamamı korunuyor — üstelik **daha az token**
harcayarak.

| subset | native | **detector_suppress** | çakışan katman | McNemar | token | dedektör/oracle |
|---|---|---|---|---|---|---|
| sh_6k | 0.290 | **0.900** | 4/74 → **66/74 (%89.2)** | +61, p=1.4e-17 | **−%3.48** | net 61/62 = **0.984**, çakışan kazanç **1.000** |
| sh_32k | 0.420 | **0.860** | 7/65 → **51/65 (%78.5)** | +44, p=1.3e-12 | **−%0.63** | net 44/46 = **0.957** |

- **A/A tabanı yine 0/0 uyuşmazlık** (iki subset), native kolu bağımsız probe
  koşusuyla ve 8 tarihsel koşuyla tutarlı.
- **Koruyucu koşul (tam ve dürüst hâli):** sh_32k'da çakışmayan katman
  **35/35 korunuyor, 0/0 uyuşmazlık**. sh_6k'da **25/26 → 24/26**: tek kayıp,
  gold olgusu silindiği için değil (sh_6k gold-cut = 0), modelin doğru varlığı
  bozuk üretmesi yüzündendir ("Shinzō Abe" → "Sinzō Abe"). Değerlendirici bunu
  yine de kayıp sayar; bu yüzden "zarar sıfır" değil, **"zarar 1/26 ve nedeni
  müdahale değil substrat üretimi"** denmelidir.
- **Yanlışlama kontrolü tutarsız:** `anti` kolu sh_6k'da beklendiği gibi zarar
  verdi (−4) ama sh_32k'da **yardım etti** (+6, p=0.21). Konum hikâyesini
  zayıflatan bu tutarsızlık raporda açıkça yazılmalıdır.
- **Gold'u dedektörle değiştirmenin toplam bedeli 1.000 çağrıda iki çevirmedir.**
- **Dedektör kalitesi (işletim noktası: cos_pair 0.90 · r_min 0.44 ·
  ambiguity none · nli 0.90 · pair_filter True; LLM/gold/doğruluk görmeden
  dondurulmuştur):** çift precision **1.0000** (2.673 doğrulanmış çift, 0 yanlış),
  çakışan-soru recall **133/139 = 0.957**, **bir anahtarın güncel değerini taşıyan
  0 olgu silindi**. `pair_filter=False` yarısında medyan precision 0.137 ve
  medyan hücre 769 güncel-değer olgusunu silerdi — eleme tercih değil,
  zorunluluktur.
- **Beyan edilen sapma:** `ambiguity_mode="none"`, dondurulmuş Stage-0
  `nmargin`/`H_z` ekranını devre dışı bırakır; gerekçe, bunların **512-token
  kesme kusurundan etkilenen tek kapı girdisi** olması ve recall'u 0.957 → 0.403
  → 0.144 boğmasıdır. Politika bu durumda **her soruda** ateşler; kalibrasyonda
  precision 1.00 ile güvenli, ama held-out'ta ayrıca gerekçelendirilmelidir.
- **Bilinen, sayılabilir hata kipi:** gold en yüksek seri değilse dedektör
  gold'lu olguyu siler (kalibrasyonda 2/200; `gold_rule`'a göre sh_262k'da
  73/77 gold-LATEST olduğundan oran orada daha yüksek). Ön-kayda **sayıyla**
  girecek.
- **Kanıt:** `stage0_results/stage1/detector_gap_{sh6k,sh32k}.json`,
  `stage0_results/stage1_operating_point.json`, `hnav/BUILD_NOTES.md` §11,
  409 test.
- **Durum: SAĞLAM (kalibrasyon split'i).** Held-out (sh_64k) tek-atışlık
  doğrulama ön-kayıtla ve denetim sonrası yapılacak.

**Bu ne demek:** kalibrasyon split'inde H-Nav, gold'suz bir dedektörle,
doğruluğu **+61 ve +44 puan** artırıyor, **token harcamasını düşürüyor** ve
korunması gereken katmanda zararı ≤1/26 (nedeni müdahale değil) tutuyor —
tezin üç başarı ölçütü (doğruluk ↑, token verimliliği ↑, harm ≈ 0) aynı anda
sağlanmış durumda. Eksik olan tek şey held-out doğrulamadır.

> **KAPSAM UYARISI (denetçi notu 2 — raporda öne çıkarılacak).** Bu deney
> **bütün-bağlam** koşumudur: istem `Memory 1: <tüm bağlam>` biçimindedir,
> retrieval boru hattının top-10 sayfası değil. Bu, oracle/dedektör oranını
> anlamlı kılan doğru tasarımdır; ama **sevk yolu (`apply_read_decision` ile
> alınan sayfanın düzenlenmesi) doğruluk açısından henüz ölçülmemiştir** —
> yalnız doğruluğu (correctness) test edilmiştir. Dış denetçinin soracağı ilk
> soru budur; keşifsel bir retrieval-yolu kolu ayrıca koşulup **ayrı** olarak
> raporlanacaktır.
>
> **Ayrıca:** işletim noktasında `nmargin`/`H_z` ekranı kapalı olduğu için
> politika **her soruda** değerlendirme yapar; dolayısıyla sevk edilen mekanizma
> "H-Nav'ın dondurulmuş Stage-0 kapısı" DEĞİL, "olgu-düzeyi çakışma dedektörü
> (çift kosinüsü + span artığı + ayrıştırılmış özne elemesi + çift yönlü NLI),
> koşulsuz uygulanmış" hâlidir. Yükü taşıyan şey precision 1.00'dır, kapı
> değildir; ön-koşul katmanı bu konfigürasyonda **doğrulanmamıştır**.

---

## A4. ★ DOĞRULAMA SONUCU — held-out sh_64k, tek atış, ön-kayıtlı: **BİRİNCİL ÖLÇÜT SAĞLANDI**

**Tasarım:** `stage0_results/stage1_preregistration_v2.md` (+ Değişiklik 1–4),
commit `b0ed608` **19:52:24**, atış **22:00:02** — ön-kayıt veriden önce, sıra
commit zaman damgalarıyla kanıtlı. Tek atış, opsiyonel durdurma yok, analiz kodu
önceden donduruldu. Sevk edilen ayar (retrieval yolu, benchmark'ın kendi
sayfası), dondurulmuş `:8003`, 500 çağrı.

| kol | genel | çakışmayan | **çakışan** | b/c | net | p | token |
|---|---|---|---|---|---|---|---|
| native | 0.450 | 28/34 | 17/66 (%25.8) | — | — | — | 0 |
| A/A tabanı | 0.450 | 28/34 | 17/66 | **0/0** | 0 | 1.0 | 0 |
| **detector_suppress** | **0.640** | 27/34 | **37/66 (%56.1)** | **0/20** | **+20** | **1.9e-06** | **−%0.31** |
| detector_demote_late | 0.480 | 28/34 | 20/66 | 2/5 | +3 | 0.45 | 0.00% |
| detector_anti | 0.430 | 28/34 | 15/66 | 3/1 | −2 | 0.63 | 0.00% |

### Ön-kayıtlı birincil ölçüt: **SAĞLANDI**
`net ≥ +10` (**+20**) **VE** `p < 0.01` (**1.9e-06**) **VE** token ≤ 0 (**−%0.31**).
Çakışan katmanda **b = 0** — tek bir çakışan soru bile zarar görmedi.

### Koruyucu iddia: **GEÇERSİZ (tek soruyla)**
Çakışmayan katmanda 1 kayıp: q77, **`refusal_after_edit`** — native `"John Milton"`
→ müdahaleli `"...does not contain any information about"`, üstelik **gold olgu
sayfada duruyorken**. Net −1, §5a'yı sağlar; §5b sınıfı `malformed_generation`
olmadığı için koruyucu iddiayı geçersiz kılar. Değişiklik 4 uyarınca bu
**yalnız koruyucu iddiayı** geçersiz kılar; koşum geçerlidir ve atış harcanmıştır.

**Ön-kayıtta yazılı sonuç, harfiyen:**
> **Etkili ama henüz güvenli değil.** Dedektörle doğrulanmış bayat belleğin
> olgu düzeyinde bastırılması, çakışan katman doğruluğunu 17/66 → 37/66'ya
> çıkarır (+20 net, p=1.9e-06); bedeli, düzenleme sonrası ret yoluyla kaybedilen
> bir çakışmayan sorudur. Bu mekanizma anlaşılıp ortadan kaldırılana kadar,
> çakışmasız sorgu içeren trafikte **dağıtım için önerilmez**.

### Diğer ön-kayıtlı kalemler
- **Zarar sınıfları ayrı ayrı:** suppress 1 (`refusal_after_edit`), demote_late 2,
  anti 3. **`gold_cut` = her kolda 0.**
- **Kayıtlı tahmin 2 gold-cut idi; gözlenen: tespit düzeyinde 1, doğruluk
  çevirmesi 0. Tahmin bir eksik tutturdu — yeniden yorumlanmadan "ıskalandı"
  olarak raporlanır.** Doğru mekanizma (denetçi tarafından artefaktan
  doğrulanmıştır; ilk aktarımda iki soru yer değiştirmişti): maruz kalan iki
  sorudan **q18**'in anahtarına hiç dokunulmadı (gold seri 7 silinmedi) ve o
  soru zaten natively **yanlıştı**; **q20**'de gold olgu (seri 2374) **gerçekten
  silindi**, üstelik native **doğruydu** — ve model **silinmesine rağmen yine
  doğru cevapladı**. Yani 2 tahmin edilen gold-cut, 1 silme ve **0 doğruluk
  kaybı** üretti. Bu, tahminin arkasındaki örtük "gold_cut ⇒ zarar" varsayımını
  zayıflatır ve gizlenmek yerine açıkça yazılmalıdır. (Havuz sınırı atfı
  **geri çekildi** — artefaktan doğrulanamıyordu; `detector_gap` artık soru
  başına `n_pool`/`pool` kaydediyor ki bu tür iddialar denetlenebilsin.)
- **★ Ama q20'nin ayrıntısı bir UYARIDIR, güvence değil.** O soruda sayfadan
  "Europe" silindi ve geriye yalnız **"Asia"** kaldı — model yine de **"Europe"**
  cevabını verdi. Yani bağlamı değil, **dünya bilgisini** kullandı: T12'nin
  ayırmak için kurulduğu H-parametrik davranış. Dolayısıyla **"gold cut'lardan
  0 doğruluk kaybı" bulgusu, gold cut'ların güvenli olduğunun değil,
  DEĞERLENDİRİCİNİN bunu yakalayamadığının kanıtıdır**: sayfa daha yanlış hâle
  getirildi ve metrik yine de memnun kaldı. Tezde bu, olumlu bir sonuç olarak
  değil, ölçüm sınırlılığı olarak yazılmalıdır.
- **Void koşulları 1,2,3,4,6,7,8: HEPSİ GEÇTİ** (uyuşmazlık 0; A/A **0/0**;
  735 bastırmanın **0**'ı zararlı — denetçi ayrıştırıcı gerçeğine karşı
  bağımsız doğruladı, precision **1.000000**; sayfa kaynağı benchmark;
  containment 0, page_edit hatası 0, 100/100 ateşleme). **Koşum geçersiz
  DEĞİL.** *Marjlar açıkça yazılmalıdır:* VC2 dar geçti — native 0.450,
  önceden sabitlenmiş [0.30, 0.50] bandının **üst kenarında**; çakışmayan
  native 28/34 = 0.824, 0.80 tabanını **tek soruyla** aşıyor. İkisi de gerçek
  geçiştir ama hakem bunu kontrol edecektir; kendimiz yazalım.
- **Etki ölçekle küçüldü — tam da öngörülen nedenle:** +62/+66 (sh_6k),
  +44/+38 (sh_32k), **+20** (sh_64k). 50-olgu havuz sınırı + 17 chunk'ın
  10'unun alınması, dedektörün görebildiğini sınırlıyor (735 bastırma vs
  1416/1257). §8.2 bu yönü atıştan önce yazmıştı.
- **Yön kontrolü üç kez de doğru:** sevk edilen düzende `anti` −1, −6, −2 —
  bir kez bile ters işaret vermedi.
- **Kanıt:** `stage0_results/stage1/detector_gap_sh64k.json`,
  `stage0_results/stage1_preregistration_v2.md`, 455 test.
- **Durum: KESİN.** Held-out, tek atış, ön-kayıtlı, denetimden geçmiş.

---

## B. Metodolojik katkı — NLI tek başına bellek çakışmasını doğrulayamaz

**İddia.** Çift yönlü NLI çelişki skoru, bellek çakışması doğrulaması için
**tek başına yetersiz**: aynı şablon/farklı özne çiftlerini çelişki sayıyor.
Ayrıştırılmış **özne-kimliği elemesi** eklendiğinde yanlış-doğrulama sıfırlanıyor.

| yapılandırma | yanlış-doğrulanan çift oranı |
|---|---|
| yalnız çift yönlü NLI (cos 0.90, dondurulmuş r) | **0.933** (12.896 farklı-anahtar / 923 gerçek supersession) |
| yalnız çift yönlü NLI (cos 0.94) | 0.33–0.39 |
| **+ `same_key_pair` özne elemesi** | **0.000 — 162 hücrenin hepsinde, precision 1.00** |

Örnek (gerçek model, box'ta ölçüldü): *"Thomas Kyd was born in the city of
London."* vs *"Marlowe was born in the city of London."* → çelişki **0.99949 /
0.99983**, iki yönde de. Mantıken çelişmeyen iki olgu.

- **Kanıt:** T11 kalibrasyon çıktısı + `hnav/adapters/mab_adapter.py`
  (`same_key_pair`), `hnav/core/read_gate.py` (çift yönlü kapı), supervisor
  denetim kaydı (bağımsız box probe'u).
- **Sınırlılık:** eleme, ayrıştırılabilir özne gerektirir (bu arenada
  `conflict_analysis.parse`, %99.4+ kapsam); serbest metin bellekte karşılığı
  ayrıca tasarlanmalı.
- **Durum: SAĞLAM.** Diğer tüm sonuçlardan bağımsız; RAG bellek sistemlerine
  doğrudan aktarılabilir.

---

## C. Doğrulanmış tespit katmanı (Stage-0)

| bileşen | sonuç | durum |
|---|---|---|
| Geometri öncülü (M1) | çakışan çift medyan benzerlik **0.964** vs kontrol **0.60**; **AUC ≥ 0.9999** 4/4 subset | SAĞLAM (fact düzeyi — kesmeden etkilenmez) |
| Geometrik gruplama (M1b) | best-F1 **0.892** (sh_6k, τ=0.91) → **0.757** (sh_262k, τ=0.95); precision 0.83–0.90 | SAĞLAM (fact düzeyi) |
| Replika sadakati (M0) | **top-1 = top-k = Kendall τ = 1.0000**, maks skor hatası ≤ 4.5e-5, 400/400 çift | KESİN (benchmark'ın kendi vektörleriyle) |
| Sinyal dejenerasyonu (M2) | **NOT_DEGENERATE — 2/2 kalibrasyon, düzeltilmiş embedder ile YENİDEN KAZANILDI**; sh_64k/sh_262k hâlâ L512 dönemi | SAĞLAM (kalibrasyon), AÇIK (held-out) |
| Gölge nötrlüğü (T4/S2) | off↔shadow %2.42 < off↔off %3.04; TOST ±2.0 eşdeğerlik (p=0.0008/0.017) | SAĞLAM |
| Ayrıştırıcı kapsamı | %99.44–99.65 | KESİN |

---

## D. Substrat bulguları (metodoloji katkısı)

**D1 — bf16 gömme hassasiyeti retrieval sadakatini sessizce yok ediyor.**
Aynı embedder bf16 servis edildiğinde top-k sıra özdeşliği **0.24**'e düştü;
float32'de **1.0000**. Mekanizma: birim-norm sapması ±2e-3, dot-product
sıralamasının L2 ile özdeşliğini beraberlik ölçeğinde bozuyor.
Kanıt: `stage0_results/final/m0_replica_fidelity.GATE_20260814_bf16.json` (öncesi)
+ `m0_replica_fidelity.json` (sonrası). **Durum: KESİN.**

**D2 — vLLM temperature=0'da koşudan koşuya deterministik değil.**
Çıktı uyuşmazlığı ortalama **%3.0** (maks %9); özdeş iki baseline koşusu arasında
exact-match **26.0 vs 30.0** (4 puan). Ön-kayıtlı TOST protokolüyle ölçüldü
(10 off + 5 shadow koşu). **Yeni incelik:** gürültü **tamamen çakışan soru
katmanında** — 28 koşu-çiftinde çakışmayan soruda **sıfır** çevirme.
Kanıt: `stage0_results/t4_s2_trials_summary.json`, `t4_s2_protocol.md`,
`question_strata.json`. **Durum: KESİN.**
*Sonuç:* bu literatürdeki tek-koşu benchmark iddiaları bu bandın içinde.

**D2b — YENİDEN TÜRETME SONUCU (2026-08-16): kusur gerçekti, yayılımı sınırlıydı.**
Düzeltilmiş embedder ile kalibrasyon split'i yeniden ölçüldü
(`stage0_results/refit_L8192/`, `refit_threshold_diff.md`):

| büyüklük | sonuç |
|---|---|
| **`R_MIN_CAL`** | **DEĞİŞMEDİ** — 0.1923661662 → 0.1923663786 (bağıl 1.1e-06, fp32 gürültüsü). *Tüm triyajın dayandığı "olgu düzeyi sinyaller güvenli" öncülü DOĞRULANDI.* |
| M1 AUC, `gate_pass` | 1.000000, True — değişmedi |
| M1b τ/P/R/F1, dolayısıyla `COS_PAIR_CAL`=0.92 | **bit-özdeş** |
| m3 yazma tarafı, m4 girdileri | 4–6 ondalıkta özdeş |
| `nmargin` p25 (havuzlanmış) | **−%18.0** (0.0047644 → 0.0039061) |
| `H_z` p75 (havuzlanmış) | +%4.1 (1.9569 → 2.0363) |
| sh_32k retrieval belirsizliği | `margin` p50 **−%54**, `nmargin` p50 **−%52** — kesilmiş ölçüm arenayı olduğundan **kolay** göstermiş |

**Havuzlama kusurunun sert kanıtı:** havuzlanmış p75, **her iki dönemde de**
sh_32k'nın medyanına eşit (1.9569 vs 1.9573; 2.0363 vs 2.0377) — yani bu
istatistik hiçbir subset'i tarif etmiyor. Ayrıca `H_z` ekranı sh_6k'da
**yapısal olarak erişilemez**: `n_chunks=2` iken `H_z ≡ 0.3653338551` (tam),
dolayısıyla hiçbir kesin eşitsizlik onu seçemez — eski eşikle de, yeni eşikle
de, sh_6k'nın kendi p75'iyle de. Eşikler **subset başına** tüketilmelidir.

**Karar:** canlı sabitler (`NMARGIN_CAL`/`H_Z_CAL`) **değiştirilmedi**; L512
dönemi değerleri "analiz için aşıldı, artefakt yeniden üretilebilirliği için
korundu" diye işaretlendi ve düzeltilmiş **subset-başına** değerler yanına
kaydedildi. Gerekçe: (1) doğrulama koşumu dâhil commit'li artefaktlar bu
sabitlerle üretildi — sessiz değişim, ön-kayıtlı tek atışı kendi koduyla
yeniden üretilemez kılardı; (2) sevk edilen konfigürasyonda bu sabitler zaten
**atıldır** (`ambiguity_mode="none"`). *Not:* m3/m4'ün LLM-türevli etiketleri
L512 dönemidir (girdileri özdeş doğrulandı) — beyan edilir, yeniden koşulmaz.

**D3 — 512-token kesme tuzağı (kendi kusurumuz, düzeltildi).**
`build_embedder` dört konumsal argüman geçtiği için `max_length=512` hiç
ezilmiyordu; chunk'lar ~4096 token (ölçülen en büyük kalibrasyon chunk'ı
**4.333** tiktoken). Yani chunk düzeyi sinyaller metnin ilk ~%12'sinden
hesaplanmış. M0'ın 1.0000 sadakati bunu **kapsamıyordu** (benchmark'ın kendi
vektörlerini yeniden kullanıyor). Düzeltme `5240774`: `DEFAULT_MAX_LENGTH=8192`,
tüm argümanlar anahtar kelimeyle, **cache namespace'e `L{max_length}` eklendi**
(aksi hâlde 24k kesilmiş vektör geri okunur ve düzeltme "değişiklik yok" ölçerdi).
**Durum: KESİN (kusur ve düzeltme); türevleri GEÇİCİ.**
*Raporlanabilir ders:* gömme boru hattında sessiz kesme + içerik-adresli cache,
düzeltmeyi de görünmez kılar; cache anahtarı her parametreyi taşımalı.

---

## E. CrossEp-Know yazma tarafı yapısal olarak farklı

Byte-özdeş yinelenen yazma oranı (küme-ortalaması, **cluster-first**):
**0.117 kalibrasyon / 0.072 held-out**, 120 bağlamın 89'unda mevcut, en kötü
küme **0.706**; lexical Jaccard ≥0.9 oranı 0.164/0.112. Karşılaştırma:
MemoryAgentBench'te `duplicate_rate` **0.000** her yerde.
Mekanizma tanımlı: her örneğin yörüngesi bağlamın ortak System Context bloğunu
yeniden chunk'lıyor.

- **Kanıt:** `stage0_results/crossep/m5_crossep_write_headroom_qwen3_embedding_SMOKE.json`,
  `CROSSEP_HEADROOM_RAPORU.md`. MD5 ve Jaccard **embedder-bağımsız** → smoke
  koşusuna rağmen gerçek; kesme düzeltmesinden etkilenmez.
- **Doğrulama:** supervisor bağımsız yeniden saydı (en kötü küme 89/126 = 0.7063,
  ham JSONL'den gerçek chunker ile).
- **Anlamsal eksen ÖLÇÜLDÜ (2026-08-16, gerçek embedder, 120 bağlam / 7.879
  yazma olayı):**

| eksen | kalibrasyon | held-out | kontrol |
|---|---|---|---|
| byte-özdeş (MD5) | 0.117 | 0.072 | — |
| **çift-bazlı yakın-yineleme (cos ≥ 0.95)** | **%21.8 bağlam-içi** | — | **%0.04 bağlam-arası** (20.000 çiftin 8'i) — **~545× ayrışma** |
| en-yakın-komşu ≥ 0.95 (şişirilmiş istatistik) | 0.863 | 0.853 | — |
| **çift yönlü çelişki ≥ 0.90** | **0.0129** | — | — |
| `is_critical_delta` | **0.0000** | **0.0000** | — |

> **Hangi sayı yazılmalı:** **%21.8 vs %0.04** (çift-bazlı, kontrollü, ~545×
> ayrışma). 0.86 bir **en-yakın-komşu** istatistiğidir: bağımsızlık altında
> ölçülen p=0.218 ve gerçek mağaza boyutlarıyla **öngörülen oran 0.944** — yani
> gözlenen 0.863'ün **üstünde**. Dolayısıyla 0.86 **iddianın parçası değildir**;
> çift oranı ve mağaza boyutunun zaten öngördüğü şeydir, ek kanıt taşımaz.
> *(Düzeltme: önceki metin bağlam-arası oranı "%0.0 / hiçbiri" diye yuvarlamıştı;
> gerçek değer 20.000 çiftin **8'i**. `m5b` kontrol betiği bunu yakaladı —
> kontrolü kod olarak commit etmenin amacı tam da budur.)*

- **Mekanizma ayrımı (hipotezim yarı yanlıştı, kontrolle düzeltildi):**
  *exact* yinelemeler harness artefaktıdır (%11.6'sı System Context bloğunun
  birebir alt dizisi — MD5 oranıyla neredeyse aynı); ama **yakın-yineleme
  kütlesi değildir**: sistem içeriği neredeyse sıfır olan "organik" chunk'lar
  da `sim ≥ 0.95`'e %85.0/%82.9 oranında ulaşıyor ve yakın-yineleme
  komşularının %62–63'ü aynı bağlamın **farklı bir örneğinden** geliyor —
  gerçek çapraz-epizot birikimi.
- **Çelişki ekseni pratikte YOK** — çelişki 0.0129 ve `is_critical_delta`
  0.0000, fazlalık kütlesinin **iki kat büyüklük altında**. Birincil arenanın
  tam tersi vurgu. **Uyarı:** 0.0129 bir **alt sınırdır** — NLI girdilerinin
  **680/800'ü (oran 0.850)** DeBERTa'nın kendi 512-pozisyon sınırında kesildi
  (modelin özelliği, ayar değil); tam metinle daha yüksek çıkabilir ama iki
  kat büyüklük farkını kapatması olası değildir.
- **Sınırlılık:** split 48/72 **küme** (ICC 0.346, etkin N ≈ 276/884) — güç
  küme-öncelikli hesaplanmalı. Artefaktta `embedder_provenance` alanı **yoktu**
  (sonradan eklendi, geçmişe dönük **doldurulMAdı**); GQA düzeltmesinin
  uygulandığı yalnız **dolaylı** kanıtlıdır (ön-koşum probu True; 3/3 OOM veren
  koşum 18.7 GiB'de tamamlandı).
- **[GATE] KARARI (kabul edildi):** **fazlalık tavanı gerçek ve ölçülmüştür;
  doğruluğa dönüşümü kanıtlanmamış ve şu an ölçülemezdir.** Yazma-yolu
  NO_GO'sunun dürüst muadili. İşaret bile öngörülemez: boşalan retrieval
  yuvaları yararlı çeşitlilik ekleyebilir de, tek ilgili chunk'ı dışarı da
  atabilir (birincil arenada 64k'da yardım eden telafinin 262k'da net zarar
  vermesi emsali). Ayrıca "fazlalık var" ≠ "fazlalık zarar veriyor":
  `rank_self` top-1 0.972/0.979 ve düşük QR yeniliği, **tutarlı** bir mağaza
  ile de uyumludur.
- **Durum: SAĞLAM (her iki eksen de ölçüldü), AÇIK (doğruluğa dönüşüm).**

---

## F. Dürüst olumsuz sonuçlar (tezde yer alacak)

1. **Yazma-yolu müdahale tavanı ≈ 0 (MAB).** Veto sonrası müdahale oranı
   %0–1.6; doğrulama subset'inde (sh_64k) could-change-correctness **0.00**.
   → `write_policy` KALICI NO_GO. `KAPI_KARARI.md` §2.
2. **H2 marginal-diff testi ön-kayıtlı konjonksiyonu geçemedi** — düşen koşul
   LRT **p = 0.341** (yön pozitif: in-sample Δauc +0.0674, CV +0.1185).
   `stage0_results/final/m4_marginal_diff_test.json`.
3. **Chunk düzeyi YUKARI rerank net fayda vermedi** — 162 kapı işletim
   noktasında, hücre başına 68–115 sıra-değişen soruda. **Dar okuma zorunlu:**
   kalibrasyon split'inde retrieval zaten eksiksiz (n_chunks 2/9 ≤ top_k 10),
   dolayısıyla bu, "H-Nav çalışmıyor" değil "tek yönlü chunk permütasyonu bu
   koşulda kaldıraçsız" demektir. Güç analizi de zayıf (aşağıda).
4. **Kalibrasyon hedefinin gücü yetersizdi.** Ölçülen %3.3/soru gürültü
   tabanında, hedef saf gürültü altında ~%35 yanlış-pozitif veriyor ve mükemmel
   bir müdahaleye karşı bile ~%81'de doyuyor; 162 hücre ~1–3 bağımsız teste
   denk. → sh_64k ön-kaydı **geri çekildi** (`stage0_results/stage1_preregistration.md`,
   WITHDRAWN, gerekçeli).
5. **Kendi eşiğimizin kusuru: `H_Z_CAL` bir mağaza-boyutu dedektörü.**
   n_chunks=2'de entropi tavanı `ln 2 = 0.693`; dondurulmuş eşik **1.9569** →
   sh_6k'da **hiçbir zaman** ateşleyemez (m2 sh_6k `min=max=p50=0.36533`).
   Ayrıca pooled percentile ile fit edilmiş, deponun kendi "stratify, never
   pool" kuralına aykırı. Yeniden türetme sırada, **subset başına**.

6. **ABTT anizotropi düzeltmesi doğruluğu değiştirmedi (kesin sıfır).**
   Kosinüs ekranından **önce** ABTT uygulandı, eşikler düzeltilmiş uzayda
   sıfırdan yeniden fit edildi (`cos_pair` 0.90 → 0.30), sh_64k'da ön-kayıtlı
   olarak atıldı. Sonuç: **her iki kolda da çakışan katmanda 37/66 — tek bir
   soru bile farklı değil** (%95 GA [0, 0], McNemar p = 1), eşit zarar ve eşit
   token maliyetiyle.
   - **Geometri gerçekten düzeldi:** anizotropi 0.6024 → ~0.000; aday-çift
     tabanı 0.5815 → 0.083; eşit duyarlılıkta ekran kesinliği %5.3 → **%51.3**;
     kesinlik 1.000'de duyarlılık 0.0750 → **0.5125** (sh_6k) ve 0.0072 →
     **0.2910** (sh_32k).
   - **Sonuç güçsüzlükten değil:** ham kol, §A4'teki doğrulama kampanyasını
     **500/500 aynı sonuç ve sıfır farklı model çıktısıyla** yeniden üretti;
     A/A tabanı gerçek 0/0. Ölçüm gürültüsü yok — sınırlama 66 soruluk
     genellenebilirlik.
   - **Etkisiz de değil:** iki kol **100 sorunun 12'sinde** farklı bastırma
     planı üretti (16 fakt farkı), model çıktısı tam **1** soruda değişti,
     doğruluk **0** soruda değişti. İki geometrinin anlaşamadığı faktlar,
     hiçbir sorunun bağlı olmadığı faktlar.
   - **Yorum:** ABTT, darboğaz olmayan aşamayı iyileştiriyor. Bu hattın
     kesinliği regex `pair_filter` + NLI'dan geliyor, kosinüsten değil.
     Mekanizmayı **sınırlar, çürütmez**: ölçülen kazanımlar tam olarak
     kosinüsün kesinliği **tek başına** taşımak zorunda olduğu rejimde yaşıyor
     — yani ayrıştırılabilir şablonu olmayan her arenada. Belirleyici devam
     deneyi: regex ekranını kaldırıp iki geometriyi yeniden koşturmak.
   - **Ek ölçüm:** sorgu vektörünü de beyazlatmak **zararlı** — erişilebilir
     gerçek-supersession çiftleri sh_6k'da 1,443 → 1,048 (−%27) düşüyor, çünkü
     `select_pool` daha kötü bir havuz kuruyor. ABTT simetrik fakt–fakt
     karşılaştırmasına yardım eder, asimetrik soru–fakt erişimine zarar verir.
   - Kanıt: `stage0_results/abtt/` (ön-kayıt `dd4439b`, analiz kodu `132a532`,
     sonuçlar `5917fa1`), `ABTT_REPORT.md`.

---

## G. Bugün iddia EDİLEMEYENLER

- ✅ **İDDİA EDİLEBİLİR — denetçinin onayladığı tam cümle (tezde bu hâliyle):**
  > Held-out `factconsolidation_sh_64k` alt kümesinde, tek bir ön-kayıtlı
  > doğrulama koşumunda (100 soru × 5 kol, 500 çağrı, dondurulmuş substrat,
  > benchmark'ın aldığı top-10 sayfa), dedektörle doğrulanmış superseded
  > olguların **olgu düzeyinde bastırılması**, çakışan katman doğruluğunu
  > **17/66 → 37/66** çıkardı — **+20 net uyuşmaz çift, McNemar exact
  > p = 1.9×10⁻⁶, çakışan katmanda sıfır zarar** — tam **0/0** ölçülmüş bir A/A
  > gürültü tabanına karşı ve **token maliyeti olmadan (−%0.31)**. Bastırma
  > precision'ı **1.000** (735/735 silinen olgu bağımsız olarak superseded
  > doğrulandı; hiçbir anahtarın güncel değeri silinmedi). Kapı **gold'suzdu**:
  > işletim noktası, hiçbir kol notlanmadan önce yalnız tespit kalitesiyle
  > donduruldu. **Ön-kayıtlı koruyucu ölçüt sağlanMAdı**: bir çakışmayan soru
  > geriledi — ihtiyaç duyduğu olgu sayfada dururken model cevap vermeyi
  > reddetti (`refusal_after_edit`). Mekanizma bu nedenle **etkili ama henüz
  > güvenli değildir** ve bu zarar kipi ortadan kaldırılana kadar çakışmasız
  > sorgu içeren trafikte dağıtım için önerilmez.
- ⚠️ **"Etkili" mutlaka kapsamlandırılmalıdır:** *bu arenanın çakışan
  katmanında* etkili. Niteliksiz bırakılırsa genel etkililik gibi okunur.
- ❌ **sh_64k için oracle oranı yok.** %98.4 / %95.7 tavan-yakalama rakamları
  yalnız kalibrasyona aittir ve burada farklı harness'tandır; sh_64k'da oracle
  kolu yoktur (bütün-bağlam fiziksel olarak sığmıyor).
- ❌ **"Zararsız" / "regresyon yok"** denemez. Koruyucu iddia geçersizdir ve bu,
  iyileşmeyle **aynı nefeste** söylenmelidir, sonraki paragrafta değil.
- ❌ Kalibrasyondan tahmin: +62/+66 ve +44/+38 aktarılıyormuş gibi sunulamaz
  (§8.2 ve R3 ile yasaklı).
- ❌ **"H-Nav'ın Stage-0 kapısı doğruluğu artırır"** denemez — `nmargin`/`H_z`
  atıldır, ateşleme koşulsuzdur, Stage-0 ön-koşul katmanı doğrulanmamıştır.
- ❌ Genelleme: tek arena, tek alt küme, tek ölçek, tek substrat, tek atış.
  Gold-cut tahmini **ıskalandı** ve öyle raporlanmalıdır.
- ❌ **"Ölçekte güvenli"** denemez: 50-olgu havuz sınırı ve eksik retrieval
  (17 chunk'ın 10'u) hem faydayı hem maruziyeti yalnız burada ölçülen biçimde
  daraltıyor.
- ❌ "H-Nav'ın okuma-yolu müdahalesi işe yaramaz." Test edilen tek mekanizma
  (tek yönlü chunk rerank) kaldıracın olmadığı bir split'te, güçsüz bir hedefle
  denendi. Bunu H-Nav hakkında olumsuz sonuç diye yazmak yanlış-olumsuz olur.
- ❌ m3'ün sh_64k `+6/−1` enjeksiyon sonucu bir "sonuç" değil: tek tekrarsız
  koşu, benchmark'ın kendi istemi değil, ±3 çevirme gürültüsü, ve 7 chunker
  artefaktı enjeksiyonuyla kirli.
- ❌ Kalibrasyondan sh_64k/sh_262k'ya çıkarım. Dondurulmuş `H_z` eşiği bile
  subset'ler arası %0 / ~%50 / >%90 / %100 ateşliyor.

---

## H. Tavsiye edilen tez omurgası (bugünkü kanıtla)

**Ölçüm ve yönetişim katkısı** (A + B + C + D), **artı** açık bir müdahale
sorusu (probe sonucuna göre E veya A'nın devamı):

> H-Nav, bellek çakışması geometrisi için araçlandırılmış ve doğrulanmış bir
> tespit katmanıdır. Kanonik çakışma-çözümü arenasına uygulandığında, katmanın
> kendi ölçümleri arenanın manşet doğruluğunun çakışmasız sorulardan geldiğini
> ve modelin açık supersession talimatını ~%95 uygulamadığını gösterir. Bu,
> bellek yönetişiminin *nerede* gerekli olduğunu tanımlar. Yol boyunca: NLI'nin
> tek başına çakışma doğrulayıcısı olarak %33–93 yanlış-doğruladığı ve özne
> elemesiyle 0.000'a indiği; bf16 servisin retrieval sadakatini sessizce yok
> ettiği; ve değerlendirme substratının ±2–4 puan gürültü taşıdığı ölçülmüştür.

Bunu bir **iyileştirme** iddiasına çevirmek için tek gereken: probe'un
"bastırma/yerleşim tavanı" ölçümü pozitif çıkarsa, o mekanizmanın yeni bir
ön-kayıtla doğrulanması.
