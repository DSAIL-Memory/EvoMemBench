# DURUM RAPORU — H-Nav Stage-0 Kampanyası

> Tarih: 2026-08-14, öğleden önce (denetçi ajan raporu)
> Dal: `claude/evomembench-hnav-analysis-nfwl9z` · Box: `ozonderlab2` (`/mnt/nvmes/nvme1/egekutlu/EvoMemBench`)
> Bu raporun yazıldığı anda boru hattı box'ta **çalışıyor** (t4 aşamasında, nohup altında).

---

## 1. Bugüne kadar yapılanların özeti

### T0 — Ofansif ölçümlerin yeniden üretimi (lokal, stdlib)
- `conflict_analysis.py`: sh_262k'da 11.037 anahtar / 7.197 çakışmalı (%65,2) — doğrulandı.
- `gold_rule.py`: sh_262k sorularının %77'si çakışmalı; 73/77 soruda gold = LATEST — doğrulandı.
- `marginal_diff.py`: yalnız leksik vekil; hiçbir eşik buradan türetilmedi.

### T1 — M1 geometri kalibrasyonu (box, in-process float32 `HFEmbedder`, Qwen/Qwen3-Embedding-4B) — **PASS, S3 ateşlemedi**
| subset | n_facts | çakışma çifti | median whole_blob_sim | kontrol medyanı | AUC (çakışma vs kontrol) |
| --- | --- | --- | --- | --- | --- |
| sh_6k | 455 | 160 | **0.9636** | 0.5977 | 1.0000 |
| sh_32k | 2.310 | 835 | **0.9638** | 0.6051 | 1.0000 |
| sh_64k | 4.580 | 1.687 | **0.9638** | 0.6021 | 1.0000 |
| sh_262k | 18.332 | 7.197 | **0.9641** | 0.6109 | 0.9999 |

S3 eşiği 0.70'ti; medyan ~0.964 → geometri öncülü bu benchmark'ta sağlam. Çakışan çiftler kontrol çiftlerinden neredeyse mükemmel ayrışıyor.

### T2 — M1b gruplama ablasyonu (geometri vs regex-oracle) — **PASS**
| subset | en iyi F1 | precision | recall | eşik (τ) |
| --- | --- | --- | --- | --- |
| sh_6k | **0.892** | 0.860 | 0.925 | 0.91 |
| sh_32k | **0.839** | 0.847 | 0.831 | 0.93 |
| sh_64k | **0.821** | 0.896 | 0.758 | 0.94 |
| sh_262k | **0.757** | 0.829 | 0.696 | 0.95 |

Yorum (brief'teki şablon): geometri, ayrıştırma olmadan regex gruplamasını büyük ölçüde geri kazanıyor; F1 mağaza büyüklüğüyle düşüyor (recall kaybı). CrossEp-Know'a taşınabilirlik iddiası bu tabloya dayanacak; sh_262k'daki 0.757 raporda açıkça beyan edilmeli.

### Dünkü (13-14 Ağustos gecesi) blokerler ve düzeltmeleri
- **m2 FAIL** — nltk `punkt` indirilemedi (downloader izin reddi). Düzeltme: `setup_ozonderlab2.sh` punkt'ı `$NVME/nltk_data`'ya elle indiriyor; `NLTK_DATA` `_activate.sh`'de ihraç ediliyor. Bugün `nltk.sent_tokenize` box'ta doğrulandı.
- **m0 / t4 SKIP** — benchmark bağımlılıkları (langchain, faiss-cpu, …) eksikti. Düzeltme: minimal bağımlılık seti kuruldu; bugün her iki deps-probe da geçiyor.
- Bu düzeltmeler `50cb978` commit'inde; box bu sabah `git pull` ile güncellendi.

### Bugünkü SSH otomasyonu
- `ssh -o BatchMode=yes egekutlu@ozonderlab2...` anahtarla, şifresiz çalışıyor (dün akşam kurulan key + `chmod g-w,o-w ~` düzeltmesi).
- Orkestrasyon lokalden: her adım ssh ile tetikleniyor, uzun işler yalnız nohup altında, izleme status/log dosyalarından. PLAN_YARIN.md mimarisine uygun.

---

## 2. GATE S1 teşhisi (bugün 10:30) — sınıf (a): HARNESS HATASI, düzeltildi

### Ne oldu
Sabah 10:30'da devam ettirilen boru hattı, m0 (canlı-indeks replika sadakati) aşamasında ~16 saniyede **S1 kapısını ateşledi** ve kendini durdurdu:

```
sh_6k:   top1=0.9400 topk=0.9400 tau=0.8800 maxΔ=5.26e-01
sh_32k:  top1=0.9000 topk=0.2400 tau=0.9333 maxΔ=5.57e-01
S1 FIRED — worst top-k agreement 0.2400 < 0.999.
```

### Kanıt zinciri (hepsi dosyalanmış durumda)
1. **`hnav/_out/m0_replica_fidelity.GATE_20260814_bf16.json`** (box'ta arşivlendi): her iki subsette `unit_normalized: false`, banka vektör normları **0.9981–1.0022**. fp32-normalize edilmiş vektörlerde norm 1±1e-7 olurdu; ±2e-3 sapma tam olarak bf16 kuantizasyon ölçeğidir.
2. **`hnav/_out/pipeline/embed_server.log`** satır 11: `dtype=torch.bfloat16` — `run_stage0.sh` vLLM sunucusunu `--dtype` bayrağı OLMADAN başlatıyordu; vLLM checkpoint'in bf16'sına düştü. (`pooler_config: pooling_type='LAST', normalize=True` — pooling ve normalizasyon doğruydu; sorun yalnız dtype.)
3. **Mekanizma:** `FaissFlatReplica` sıralamayı iç çarpımla yapar (`matrix @ qv`); bu, native FAISS'in L2 sıralamasıyla **yalnız birim vektörlerde** özdeştir (`d² = 2 − 2·cos`). Norm ±2e-3 kayınca iki belge arasındaki skor farkı ~4e-3'ün altındaysa sıra dönebilir. sh_32k'da 9 chunk aynı bağlamın 4096-token'lık parçaları — orta sıralardaki cos farkları bu bandın içinde. `topk_agreement` k=n=9 üzerinden **tam sıra özdeşliği** istediği için 0.24'e çöktü; top1 (0.90) ve Kendall τ (0.93) yüksek kaldı. Bu imza, "yanlış sıralama kuralı" değil, "beraberlik ölçeğinde monotonluk ihlali" imzasıdır.
4. **Doğrulayıcı deney:** sunucu `--dtype float32` ile yeniden başlatıldığında prob `max |norm−1| = 1.09e-07` ölçtü (bf16'da ~2e-3 idi) ve **aynı replika kodu, aynı m0 çağrısı** ile:

```
sh_6k:   top1=1.0000 topk=1.0000 tau=1.0000 maxΔ=4.08e-05
sh_32k:  top1=1.0000 topk=1.0000 tau=1.0000 maxΔ=1.96e-05
```

### Sınıflandırma gerekçesi
- Kampanyanın dtype'ı **float32 olarak sabitlenmişti** (`hnav/config.py` varsayılanı, box `.env`'inde `HNAV_EMBED_DTYPE=float32`, T1 kalibrasyonu in-process float32 `HFEmbedder` ile). CLAUDE.md: "Dtype is pinned once chosen; drift changes cosines and moves every threshold."
- :8001 sunucusunun bf16 servis etmesi bu sabitlemeye aykırı bir **harness kayması**dır; replika mantığında değil, ölçüm enstrümanında hata → **sınıf (a)**. (Sınıf (b) olsaydı: fp32 vektörlerle de uyuşmazlık sürerdi — sürmedi, 400/400 çiftte tam özdeşlik.)

### Yapılan işlem (minimal düzeltme, commit `4d6feb6`)
- `hnav/deploy/run_stage0.sh`: sunucu başlatmaya `--dtype "$HNAV_EMBED_DTYPE"` (varsayılan float32) ve `--max-model-len 16384` eklendi (fp32 ağırlıklar 15,0 GiB; native 40960'ta profil OOM riski — vLLM'in kendi uyarısı; hiçbir MAB girdisi 16384 token'a yaklaşmıyor, kesme davranışı değişmez).
- Artık ":8001'de zaten cevap veren sunucuyu yeniden kullan" dalı, kullanmadan önce vektör normlarını probluyor (fp32 hassasiyetinde birim-norm değilse sunucuyu öldürüp temiz başlatıyor) — bırakılmış bir bf16 sunucu S1 hatasını sessizce geri getiremez.
- `hnav/deploy/serve_embeddings.sh` (manuel servis yolu) aynı bayraklarla hizalandı.
- `hnav/core/` ve `hnav/stage0/`'a **dokunulmadı**; replika kodu değişmedi.

### m0 yeniden koşum sonucu — **S1 GEÇTİ**
Tam koşum (4 subset × 100 soru = 400 çift, fp32 sunucu):

| subset | n_docs | top1 | topk | τ | maxΔ | unit_normalized |
| --- | --- | --- | --- | --- | --- | --- |
| sh_6k | 2 | 1.0000 | 1.0000 | 1.0000 | 4.08e-05 | true |
| sh_32k | 9 | 1.0000 | 1.0000 | 1.0000 | 2.44e-05 | true |
| sh_64k | 17 | 1.0000 | 1.0000 | 1.0000 | 3.08e-05 | true |
| sh_262k | 67 | 1.0000 | 1.0000 | 1.0000 | 4.53e-05 | true |

`m0.status: PASS 2026-08-14 10:49:14`. Not: 400 çift, protokolün 1.000 hedefinin altında; ancak birincil arenada toplam 400 soru var — bu, arenanın **eksiksiz** kapsamıdır (aşağıda açık soru).

---

## 2b. İkinci bulgu (aynı gün): t4 FAIL — LLM istemcisi embedding sunucusuna bağlanıyordu (yine harness, düzeltildi)

m0 PASS sonrası t4 ilk kez gerçekten koştu (önceki gecelerde hep deps-SKIP idi) ve ilk sorguda çöktü:

```
File ".../methods/embedding_retriever.py", line 349, in answer_query
    "answer": response.choices[0].message.content,
TypeError: 'NoneType' object is not subscriptable
```

Bu bir **kapı değildi** (S2 = exit 42, diff aşaması; buradaki exit 1 sıradan stage FAIL). PLAN_YARIN başarısızlık kitabı: "stage FAIL ederse → kökü düzelt, push, pull, redo."

**Kök neden:** `MemoryAgentBench/main.py:31` → `dotenv.load_dotenv(override=True)`. Bu çağrı `MAB/.env`'i **kabuk ortamının üzerine** yazar. İki-uç ayrımı tasarımında (`mab.env.template`) `.env`'deki `OPENAI_BASE_URL=:8001` yalnız embeddings içindir (dosyadan `dotenv_values()` ile okunur); LLM istemcisi (`OpenAI()` çıplak ctor) kabuğun `:8000`'ini kullanmalıydı. `override=True` kabuğun `:8000`'ini `:8001` ile ezdi → chat completion **embedding sunucusuna** gitti → `choices=None` → TypeError. `mab.env.template`'teki "LLM: os.environ ONLY" iddiası `main.py:31` yüzünden yanlıştı; t4 daha önce hiç koşmadığı için tuzak hiç tetiklenmemişti.

**Düzeltme (commit `19e8a47`, repo'nun korumalı-benchmark-düzenlemesi geleneğiyle):** `main.py:31` artık `HNAV_DOTENV_NO_OVERRIDE=1` ise override yapmıyor; bayrak yoksa davranış upstream ile bire bir aynı. `stage_t4` bayrağı **her iki kolda özdeş** set ediyor (S2 kolları birbiriyle karşılaştırır; nötrlük etkilenmez). Box'ta doğrulandı: bayrakla LLM `:8000` görüyor, `DEEPSEK_*` değişkenleri yine yükleniyor; bayraksız eski `:8001` ezmesi aynen yeniden üretiliyor. hnav test paketi: 153 passed.

---

## 2c. Üçüncü bulgu: `:8000` portunda İKİ vLLM süreci — biri motor-ölü, bağlantıların ~yarısını yutuyor (KULLANICI AKSİYONU GEREKLİ)

t4'ün dotenv düzeltmesi sonrası koşumu, ilk LLM çağrısında bu kez `:8000`'e doğru bağlandı (doğrulandı: sürecin fd=7'si `127.0.0.1:8000`'e ESTABLISHED) — ama cevap gelmedi. İnceleme şunu ortaya çıkardı:

```
LISTEN 127.0.0.1:8000  users:(("vllm",pid=50319,...))   ← 5g19s ayakta
LISTEN 127.0.0.1:8000  users:(("vllm",pid=52259,...))   ← 5g10s ayakta (SİZİN sunucunuz: 52520'nin ebeveyni)
```

İki özdeş `vllm serve ... --port 8000` süreci aynı portu `SO_REUSEPORT` ile paylaşıyor; çekirdek gelen bağlantıları ikisi arasında paylaştırıyor. **pid 50319'un motoru ölü**: çocuk işçisi defunct (`50575 [python3.11] <defunct>`), `nvidia-smi`'da hiç GPU belleği yok, metrikleri 14,1M token'da donmuş. Sağlıklı olan (52259 → 52519/52520, GPU0'da 15,5 GiB) sizin aktif sunucunuz. Kanıt: 6 küçük chat probu → 4'ü hızlı cevap, 2'si 20 sn'de zaman aşımı; `/metrics` kazımaları iki farklı sayaç seti arasında gidip geliyor (103,5M ↔ 14,1M).

**Sonuç:** `:8000`'e açılan her YENİ bağlantı ~%50 ihtimalle ölü dinleyiciye düşüyor ve sonsuza dek asılı kalıyor. Bugünkü t4 askısı ve m3'ün önündeki risk budur. (Gecenin 01:53 preflight'ının geçmesi bağlantı pıyangosudur.)

**Yapılan/yapılamayan:** Ölü süreci (`kill 50319` — yalnız o; 52259 ailesine dokunmadan) kaldırmayı denedim; **izin sistemi engelledi** ve engel saygıyla kabul edildi — sizin süreçlerinize dokunma kararı size ait. OpenAI istemcisi 600 sn'de zaman aşımına uğrayıp yeni bağlantıyla yeniden dener (3 deneme); bu yüzden boru hattı topallayarak ilerleyebilir ya da t4 `APITimeoutError` ile FAIL edip durabilir — her iki durumda da status dosyaları korunur ve koşum kaldığı yerden devam ettirilebilir.

**ÇÖZÜLDÜ (öğleden sonra):** Kullanıcı onayıyla `kill 50319` orkestratör tarafından çalıştırıldı; :8000 artık 6/6 probda anında cevap veriyor; sağlıklı sunucu (52259/52520) dokunulmadı. hnav ortamından manuel chat doğrulaması: `'T4-PROBE-OK'` anında döndü.

---

## 2d. Dördüncü bulgu: t4'ün üçüncü başarısızlığı — sonuç dosyası glob'u + zehirli devam-etme (resume) dosyası (düzeltildi, commit `5f8e44a`)

Zombi öldürüldükten sonra 12:26'daki koşumda t4 saniyeler içinde yine FAIL etti ("result JSONs not found"). Teşhis, ilk bakışta görünenin tam tersini ortaya çıkardı:

- **Her iki kol da aslında BAŞARIYLA TAMAMLANMIŞTI** — 100'er gerçek cevap, özdeş metrikler (substring_exact_match 26.0). Log'daki "boş Answer:" satırları sorgu şablonunun kendi kuyruk istemidir (model çıktısı değil); "Context 0 already processed, skipping build vectorstore" sorgu başına normal bellek-içi mesajdır; "anında akan sorgular" vLLM prefix-cache isabetleridir (shadow ort. `query_time_len` 0.082 sn). Auth/base_url/guard sorunu YOKTUR (§2b düzeltmesi çalışıyor; kanıt: kol süreci fd'siyle :8000'e ESTABLISHED + yukarıdaki probe).
- **Asıl stage hatası:** `stage_t4` sonuç JSON'unu `$off_dir/*.json`'da arıyordu; `main.py` ise `<output_dir>/<DatasetName>/*_results.json`'a — bir seviye derine — yazar. Dosyalar vardı, glob bulamıyordu.
- **Daha sinsisi (gizli S2 tuzağı):** `main.py` sonuçları artımlı kaydeder ve `load_existing_results()` yarım kalmış `results.json`'dan DEVAM eder. Off kolunun dosyasında zombi-devrine ait 4 giriş (farklı sunucu koşullarında hesaplanmış; `initialization.py:90`'daki resume yolu gold cevabı list→str bozar) + 96 taze giriş vardı; shadow ise %100 tazeydi. Bu çiftle `diff_neutrality` **SAHTE bir S2** ateşleyecekti — enstrümantasyon suçlanacaktı, oysa fark iki devrin karışımından geliyordu.

**Düzeltme (`5f8e44a`, iki kolda özdeş):** (1) glob iki seviyeyi de arıyor; (2) `stage_t4` her subset koşumundan önce iki kolun çıktı dizinini `rm -rf` ile temizliyor — her S2 karşılaştırması artık iki temiz, aynı-devir koşum arasında. Box'taki zehirli `outputs/hnav_t4_*` dizinleri elle de temizlendi (`hnav/_cache/emb` dokunulmadı; `outputs/rag_retrieved` salt-yazılır telemetri, okunmuyor).

---

## 2e. GATE S2 ATEŞLEDİ (12:36) — ve A/A testi farkın kaynağını H-Nav'dan değil, sunucu alt-katmanından çıkardı

12:34 koşumunda t4, iki TEMİZ kolla (§2d düzeltmeleri sonrası) koştu ve **S2 ateşledi**. Kurallar gereği **hiçbir düzeltme yapılmadı** — boru hattı kendini durdurdu, kapı sonucu bir ölçümdür. Yapılan tek şey teşhis analizi:

**S2 çifti (off vs shadow, sh_6k):** 100 sorgunun **2'sinde** çıktı farklı (idx 70: 'Canadair' / 'Westinghouse Electric'; idx 90: 'association football' / 'basketball'). `input_len` 100/100 özdeş — iki kol modele aynı istemleri verdi; yalnız üretilen metin değişti.

**A/A kontrol deneyi (off vs off — H-Nav kodu HİÇ yok, iki kez aynı koşum):** 100 sorgunun **5'inde** çıktı farklı (idx 36, 42, 66, 80, 81), **4'ünde doğruluk değişti** (exact_match 26.0 → 30.0). Yine `input_len` 100/100 özdeş.

**Sonuç:** off↔shadow farkı (2/100), alt-katmanın kendi A/A gürültü tabanının (5/100) İÇİNDE. `:8000` vLLM sunucusu (0.9.1, continuous batching + prefix caching) **temperature=0'da koşumdan koşuma bit-özdeş DEĞİL** — brief'in "t=0'da meşru varyasyon kaynağı yoktur" öncülü bu servis yığınında ampirik olarak yanlış. Shadow enstrümantasyonunun sunucu gürültüsünün ötesinde hiçbir etkisi gözlenmedi; ama bayt-özdeşlik kriteri bu alt-katmanda **karşılanamaz**.

**Kanıt dosyaları** (commit'li): `stage0_results/t4_s2_evidence/sh_6k_{off,shadow,offA,offB}_results.json`. Karar insana aittir (aşağıda açık soru); kapı statüsü `t4: GATE S2` olarak duruyor, hiçbir downstream aşama koşmadı.

---

## 2f. Kullanıcının S2 protokolünün uygulanması (öğleden sonra): m2 TAMAM, gürültü tabanı ölçüldü, deterministik hakem koşumu

Kullanıcı kararı (orkestratör aracılığıyla): (1) tekrarlı denemelerle gürültü tabanı; (2) analiz planı veri toplamadan ÖNCE ön-kayıt; (3) kesin S2 hükmü deterministik bir sunucuda. Uygulama:

**Ön-kayıt:** `stage0_results/t4_s2_protocol.md` + sürücü + analiz betiği, commit `7cb8323` — box'a çekilme saati ~13:05, sürücü başlangıcı 13:06:05. Ön-kayıt veri toplamadan önce gerçekleşti (git geçmişi kanıt).

**m2 (paralel program, GPU1):** İlk deneme OOM (fp32 embedder + batch 32 × 512-token chunk aktivasyonları 23,5 GiB'ı aştı — m2, chunk embed'leyen İLK aşama; T1 yalnız kısa fact'ler embed'lemişti). Kampanyanın kendi konfigürasyon yüzeyi kullanıldı: `HNAV_EMBED_BATCH=8` (box `.env`; kod değişikliği yok; padding-maskeli pooling batch'ten bağımsız). İkinci koşum: **exit 0, `fallback_chunker=false` 4/4 subset**. M2 verdikti: **ham-skor entropisi cosine×100'de DEJENERE DEĞİL** (4/4 subset `NOT_DEGENERATE`) — önceki BFCL bulgusunun bu arenada reddi; brief'e göre yayınlanabilir. Margin p50: 1.235 (sh_6k) → 0.318 (sh_262k); etkin komşuluk 1.44 → 36.41.

**Gürültü tabanı denemeleri (:8000, sh_6k, m3'ten ÖNCE, sunucu boşta):** 1 atılan ısınma + 10 off + 5 shadow, serpiştirilmiş ön-kayıtlı sıra, hepsi exit 0. Ön-kayıtlı analiz (`t4_s2_trials_summary.json`, commit `8204cb5`):
- Off-içi ikili çıktı-uyumsuzluğu: **ort. %3,04, maks. %9,0** (45 çift); shadow-içi %2,2 (10 çift); **off↔shadow arası %2,42 (50 çift) — off-içi ortalamanın ALTINDA.**
- Koşum-düzeyi substring_exact_match: off 27,1 ± 1,52 (26–31!), shadow 26,6 ± 0,89.
- TOST (±2,0 puan marj): **eşdeğerlik kuruldu** (p_alt 0,0008; p_üst 0,017). Permütasyon (10k, seed 20260814): Δ = −0,0047, p = 0,475.
- Ön-kayıtlı karar kuralı: **alt-katman gürültüsü atfını DESTEKLİYOR; nötrlük aleyhine kanıt yok.** Kural gereği bu YALNIZ destekleyici kanıttır, PASS değildir.

**Program sapmaları (gerekçeli):** m2 ile denemeler eşzamanlı koşamadı (denemelerin retrieval'ı GPU1'deki :8001 embed sunucusunu, m2 aynı GPU'daki in-process embedder'ı ister); m3, Faz-3'ten SONRA başlatılıyor (m3 `build_embedder` ile GPU1'de ~16 GiB fp32 embedder'ı koşum boyunca tutar — `m3_headroom.py:482` + `gpu_guard 17` — :8002/:8001 deterministik çifti GPU1'e sığmazdı).

**Deterministik hakem (:8002, GPU1):** vLLM 0.9.1, yerel Qwen3-4B-Instruct-2507 ağırlıkları, `--enforce-eager --max-num-seqs 1 --no-enable-prefix-caching --max-model-len 12288 --gpu-memory-utilization 0.60`; embeddings :8001 **bf16** (yalnız bu test için belgelenmiş sapma — fp32 embed + chat GPU1'e birlikte sığmıyor; iki kol özdeş retrieval görür, determinizm kanıtı tüm yolu kapsar).

## 2h. m3'ün bağlam-sınırı yinelemeleri ve nihai sunucu boyutlandırması (sonradan eklendi)

§2g'deki 53.248 penceresi de yetmedi: sh_64k 55.835, sh_262k 66.435 token istedi. Ölçüm (Qwen tokenizer, offline): top-10 chunk toplamı sh_64k 49.590 / sh_262k 52.242; gözlenen 66.307'lik istem, okuma-tarafı `injected` (getirilen chunk'lardaki bayat fact'lerin en yeni sürümleri) listesinin ~14k token katkısını gösterdi. Yapısal tavan: injected ⊆ chunk-içi fact'lerin yeni sürümleri → ≤ 2×top10 + soru ≈ **104,6k**. Nihai boyutlandırma (BİR kez): `--max-model-len 106496 --kv-cache-dtype fp8 --enforce-eager` (bf16 KV 106k'da GPU1'e sığmaz; fp8-KV koordinatör/kullanıcı onaylı beyan edilen sapmadır ve 4 subsetin dördü de bu tek konfigürasyonda koşulmuştur). Eşikler dördüncü denemede de donmuş değerlerle bire bir aynı fit edildi (`nmargin<0.0048 H_z>1.9569 r_min<0.1924`, `unfit_for_analysis: false`). Gece ağ kesintisi (~21:00-01:50) koşumu etkilemedi (nohup); m3 20:13'te kendi başına tamamlandı.

**SONUÇ — ön-kayıtlı hüküm: BAYT-ÖZDEŞLİK ÜZERİNDEN KARAR VERİLEMEZ (CANNOT ADJUDICATE).**
- V1 motoru A/A (detA vs detB, ikisi de saf off): **1/100 çıktı farklı** (idx 62: 'Bernard Arnault' / 'Jack Dorsey'; input_len özdeş).
- Ön-kayıtlı geri-düşüş `VLLM_USE_V1=0` (V0 motoru) A/A (detC vs detD): **7/100 çıktı farklı** (idx 0,42,62,66,80,84,90; sem 31,0 vs 27,0) — daha kötü.
- **Mikro-prob:** her iki sunucu da TEK TEK özdeş isteklere bit-özdeş cevap veriyor (chat 3×: özdeş; embed 3×: maks fark 0.0). Yani akış-içi durum (KV blok yerleşimine bağlı fp indirgeme sırası; idx-0 gibi erken bir near-tie flip'in downstream tahsis geçmişini değiştirip kaskatlaması) uçtan uca koşumları t=0'da bile bit-özdeş olmaktan çıkarıyor.
- Ön-kayıtlı protokol Part 2 gereği: kanıtlanmış deterministik alt-katman OLMADAN off/shadow hakem çifti koşmak hiçbir şeyi karara bağlamaz — koşulmadı. "Bayt-özdeş değil → gerçek S2 ihlali" çıkarımı bu koşullarda YAPILAMAZ (protokolün önlemeye yazıldığı yanlış atıf tam olarak budur).
- Kanıt: `stage0_results/t4_s2_evidence/sh_6k_det{A,B,C,D}_results.json`.

**Toplam kanıt durumu (insan kararı için):** (i) off↔shadow farkı A/A gürültü tabanının içinde/altında (2/100 < 3,04/100; denemelerde çapraz %2,42 < off-içi %3,04); (ii) TOST eşdeğerliği ±2,0 marjda kuruldu; (iii) permütasyon p=0,475; (iv) bayt-özdeşlik kriteri bu yığında off-vs-off için bile karşılanamaz (V1 1/100, V0 7/100). Shadow enstrümantasyonuna atfedilebilir HİÇBİR fark gözlenmedi; ama brief'in bayt-özdeşlik kriteri sağlanamadığı için S2 PASS İLAN EDİLMEDİ — `t4.status` GATE olarak bırakıldı, karar T8'de insanın.

---

## 2g. S2 kapanışı (kullanıcı kararı) ve m3'ün bağlam-sınırı arızası (sınıf 2: alt-katman kapasitesi, düzeltme :8003)

**S2 kapanışı:** Kullanıcı, ön-kayıtlı istatistiksel eşdeğerlik çerçevesini uygulayarak S2'yi **PASS** ilan etti (commit `89d4f02`; `t4.status: PASS ... s2-closed-by-statistical-equivalence`). Karar insana aitti ve insan tarafından verildi; ajan yalnız kanıt üretti.

**m3 arızası (15:56):** İlk m3 koşumu sh_6k'nın TAMAMINI bitirdi (374 answer çağrısı; eşikler kalibrasyon splitinde donduruldu: `nmargin<0.0048  H_z>1.9569  r_min<0.1924`) ve sh_32k'nın LLM fazının başında öldü: `BadRequestError — 39,931 token istendi (mesajlarda 39,803), sunucu sınırı 32,000`.

**Teşhis — sınıf 2 (protokole sadık; bağlayıcı kısıt alt-katman):** m3'ün okuma-tarafı sayacı, benchmark'ın KENDİ cevap yolunu birebir aynalar: top-10 **chunk** çek, birleştir, cevapla (`counterfactual.py::retrieve_texts` + `m3_headroom.py:261` chunk_store). sh_32k = 9 chunk × ~4,4k Qwen token ≈ 39,8k — yani **benchmark'ın kendisi de (t4/sh_32k) :8000'in 32k sınırında aynı 400'ü alırdı**; t4 oraya hiç ulaşamadığı için görünmemişti. m3 fazla bağlam GÖNDERMİYOR; sınır, kullanıcının sunucu konfigürasyonu (`--max-model-len 32000`).

**Düzeltme (beyan edilen sapma):** Kullanıcının :8000'ine dokunulmadı. GPU1'de kendi sohbet sunucumuz `:8003` — aynı yerel checkpoint (Qwen3-4B-Instruct-2507), aynı `--dtype auto --max-num-seqs 32`, tek fark `--max-model-len 53248` (en uzun m3 istemi: top-10 chunk ≈ ~45k token + 128 tamamlama; 53248 marjlı) ve boş GPU1'de `--gpu-memory-utilization 0.90`. **Substrat tutarlılığı için m3 4 subsetin DÖRDÜNDE de :8003'te yeniden koşuluyor** (sh_6k'yı :8000 sonucuyla karışık bırakmak yerine) — gerekçe: tabakalı raporda tek substrat beyanı; sh_6k'nın yeniden hesabı ucuz. Eşikler aynı önbellekli embeddinglerden deterministik olarak yeniden üretilecek — YENİDEN AYAR DEĞİL; log'daki değerlerin donmuş değerlerle (0.0048/1.9569/0.1924) bire bir eşleştiği koşum sonrası doğrulanacak.

**GPU1 çakışması çözümü:** m3'ün in-process fp32 embedder'ı (16 GiB) :8003 (22,7 GiB) ile GPU1'e sığmaz. m3 yeniden koşumu `CUDA_VISIBLE_DEVICES=""` ile başlatıldı → `HFEmbedder` CPU'ya yüklenir (embedding.py:161 fallback); ilk koşum tüm metinleri embed'lediği için pass'ler %100 disk-önbellek isabeti — CPU modeli hiç forward koşmaz. Kod değişikliği yok.

---

## 3. Boru hattının şu anki durumu ve kalan aşamalar

Güncel koşum (tüm düzeltmeler sonrası, 12:34, box'ta nohup, `console9.log`):

```
cd /mnt/nvmes/nvme1/egekutlu/EvoMemBench && \
nohup bash hnav/deploy/run_stage0.sh > hnav/_out/pipeline/console9.log 2>&1 &
```

(`--redo` gerekmedi: t4 ve m2 status'ları FAIL olduğundan otomatik yeniden koşuyorlar; m0 PASS korunuyor. Önceki denemeler: console7 = zombi devri APITimeoutError; console8 = §2d glob/resume hatası.)

| stage | durum | not |
| --- | --- | --- |
| preflight, t1_smoke, t1, t2 | PASS (atlandı) | gece koşumundan |
| **m0** | **PASS 10:49** | S1 geçti, yukarıdaki tablo |
| **t4** | **PASS 16:05 — KULLANICI KARARI** | s2-closed-by-statistical-equivalence (commit `89d4f02`); kanıt zinciri §2e-2f |
| **m2** | **PASS 13:05** (manuel koşum, `HNAV_EMBED_BATCH=8`) | `fallback_chunker=false` 4/4; ham-entropi 4/4 NOT_DEGENERATE |
| **m3** | **PASS 15/08 01:52** (4. deneme, :8003 fp8-KV/106496) | §2g-2h: üç bağlam-sınırı yinelemesi; 4 subset TEK substratta; eşikler her denemede donmuş değerlerle özdeş |
| **m4** | **PASS 15/08 01:52** | **H2 GEÇMEDİ**: delta_auc +0.1185 ama key-clustered CI [−0.046, +0.188] sıfırı içeriyor; subset-clustered CI [+0.042, +0.087] yalnız 2 cluster (güvenilmez, m4 kendi uyarısı) |
| **report** | **PASS 15/08 01:52** | `report --strict: COMPLETE — every measurement present.` `STAGE0_REPORT.md` repo kökünde ve `stage0_results/final/` içinde |

**BORU HATTI TAMAM (T8 HARD STOP).** GO/NO_GO değerlendirmesi insana aittir; `KAPI_KARARI.md` §6'yı orkestratör dolduracak.

### m3 nihai headroom özeti (4/4 subset, tek substrat)

| subset | split | write n | conflict | would-intervene → veto sonrası | could-change-correctness | read: native failure (CONFLICT) | repair helped/harmed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| sh_6k | calib | 455 | 0.352 | 0.002 → 0.000 | — (hedef yok) | 0.67 | 0 / 0 |
| sh_32k | calib | 2.310 | 0.361 | 0.104 → 0.016 | **0.20** | 0.53 | 2 / 1 |
| sh_64k | conf | 4.580 | 0.368 | 0.063 → 0.013 | 0.00 | 0.56 | 6 / 1 |
| sh_262k | conf | 18.333 | 0.392 | 0.023 → 0.004 | — | 0.80 | **3 / 5** |

Okuma tarafı: READ_CONFLICT/STALE/RELEVANT_BELOW_K oranı her yerde 1.00 (yapısal: her soru çakışmalı anahtara gider); READ_MISSING sh_64k 0.28 / sh_262k 0.61, native failure 0.82-0.93. Kritik gözlem: **repair sh_262k'da net ZARARLI (3 helped / 5 harmed)** — okuma-tarafı müdahale büyük mağazada tersine tepiyor; yazma-tarafı veto-sonrası müdahale oranları ~%0-1.6. Bu tablolar Stage-1 tavanının kendisidir; yorum ve GO/NO_GO insanın.

- Boru hattı S2 ile kendini durdurdu (tasarım gereği); GPU1 boş, embed sunucu kapalı, :8000 sağlıklı (tek dinleyici).
- GPU0/:8000 (kullanıcının LLM'i, PID 52520) hiç dokunulmadı.
- S2 kararı sonrası ilerleme: t4 status'u GATE olduğundan relaunch t4'ü yeniden dener ve büyük olasılıkla yine S2 ateşler (A/A gürültüsü yapısal). m2/m3/m4 t4'e bağımlı DEĞİL (kendi çevrimdışı replikasını kullanırlar) — insan isterse t4'ü `SKIP <gerekçe>` olarak işaretleyip (`hnav/_out/pipeline/t4.status` elle) kalan aşamaları koşturabilir; bu işaretleme kapı kaydını SİLMEZ, rapor t4'ü NOT RUN/GATE olarak gösterir. Bu adım İNSAN kararıdır, ben atmadım.

---

## 4. Stage-1 tasarımcısının ihtiyaç duyacağı girdiler

1. **`STAGE0_REPORT.md` + `hnav/_out/m{0,1,1b,2,3,4}_*.json`** (box'ta; `hnav/_out` gitignore'da → `stage0_results/` altına bilinçli commit gerekir, PLAN_YARIN Safha 2).
2. **M3 headroom tabloları** — could-change-correctness oranları Stage-1'de kazanılabilecek maksimumu tanımlar; tezin başarı ölçütleri tablosuna bağlanır.
3. **M1b F1 tablosu (yukarıda)** — kazanımın geometriye mi metadata'ya mı atfedileceğinin kanıtı; sh_262k'da F1 0.757 → büyük mağazada recall kaybı Stage-1 eşik seçiminde dikkate alınmalı. Eşikler YALNIZ sh_6k+sh_32k'dan (kalibrasyon spliti) türetilebilir.
4. **M0 sonucu**: replika fp32 servis altında bire bir sadık → `rank_self`, `margin`, `dH_self`, `dH_neighbor`, `churn` sinyalleri GEÇERLİ. Ek bulgu: bf16 servis altında `FaissFlatReplica`'nın iç-çarpım sıralaması bozulur (topk 0.24 @ k=9). Stage-1 tasarımı, hedef sistem fp32-normalize embedding servis etmiyorsa sıralamayı tam L2 üzerinden yapmalı ya da dtype'ı sabitlemelidir. Bu, tez için raporlanabilir bir gürbüzlük notudur.
5. **M2 ham-entropi verdikti** (`H_raw` degenerasyonu cosine×100'de) — m2 çıktısında; iki cevap da yayınlanabilir.
6. **Değişmezler devam ediyor**: `HNAV_MODE=off`; `write_policy.py`/`read_policy.py` yok (T8 insan kararına kilitli); sh_64k/sh_262k'da eşik ayarı yasak; `hnav/_cache/emb/` silinmez/kopyalanmaz (model|dtype anahtarlı).
7. Arşiv kanıtı: `hnav/_out/m0_replica_fidelity.GATE_20260814_bf16.json` (bf16 GATE ölçümü) — `stage0_results/`'a kopyalanmalı.
8. **S2 / alt-katman nondeterminizmi bulgusu (§2e)**: vLLM continuous batching + prefix caching altında t=0 bit-özdeşliği yok (A/A: 5/100 çıktı, 4/100 doğruluk oynuyor). Stage-1 için iki içerim: (i) tek-koşum benchmark skorları ±~2-4 puanlık sunucu gürültüsü taşır — Stage-1 etki ölçümleri bu tabanın ÜzERİNDE olmalı ya da çok-koşum ortalaması kullanılmalı; (ii) "bayt-özdeşlik" türü nötrlük kriterleri bu yığında istatistiksel eşdeğerlikle değiştirilmeli. Kanıt: `stage0_results/t4_s2_evidence/`.

---

## 5. KULLANICIYA AÇIK SORULAR

Öğleden sonra kullanıcı onayı alınanlar (orkestratör aracılığıyla): **fp32 servis beyan edilen sapma olarak ONAYLANDI**; **bf16 S1 olayı teze gürbüzlük bulgusu olarak GİRECEK**; **M0 400/400 eksiksiz sayım olarak KABUL EDİLDİ**; **`kill 50319` kullanıcı onayıyla yapıldı** (§2c çözüldü). Kalan açık sorular:

1. **S2 kararı — ÇÖZÜLDÜ (kullanıcı kararı, 2026-08-14 ~16:00).** Protokol uygulandı; bayt-özdeşlik kriteri bu yığında off-vs-off için bile sağlanamıyor (V1 1/100, V0 7/100 — §2f), off↔shadow farkı her ölçümde gürültü tabanının içinde ve TOST eşdeğerliği ±2,0 marjda kurulu. Kullanıcı S2'yi **ön-kayıtlı istatistiksel eşdeğerlik çerçevesiyle KAPATMAYI** seçti: t4 = PASS-by-statistical-equivalence, tezde uyarlanmış kriter olarak tam beyanla yazılacak. `t4.status` bu karar uyarınca orkestratör tarafından PASS'a çekildi (karar referansı statü satırında).
2. **T8 kapısı bu akşam sizde.** Boru hattı raporu ürettiğinde GO/NO_GO değerlendirmesi yalnız insan tarafından, `EVOMEMBENCH_HNAV_STAGE0_PROTOCOL.md` §4'e göre yapılacak. Ajanlar karar vermeyecek; `KAPI_KARARI.md` yalnız karar destek dosyası olacak.
3. **`--max-model-len 16384` (embed sunucusu).** fp32 profil OOM riskine karşı eklendi; hiçbir MAB girdisi bu uzunluğa yaklaşmadığı için davranışsal etkisi yok. Sunucu bayrağı sapması olarak beyan ediyoruz — itirazınız var mı?
4. **t4/sh_32k bağlam sınırı riski (S2 sh_6k'da ateşlediği için henüz test edilmedi).** sh_32k'da 9 chunk × ~3,3k token + sorgu ≈ ~30k token istem, :8000 sunucunuzun `--max-model-len 32000` sınırına çok yakın (tokenizer farkıyla aşabilir). Aşarsa t4/sh_32k `BadRequestError` ile düşer; çözüm sizin sunucunuzun sınırını yükseltmek ya da bu subset için sapma beyanı olur — karar sizin.
5. **`main.py` resume davranışı (§2d).** `load_existing_results` gold cevabı list→str bozuyor ve devirler-arası karışıma açık; t4 için `rm -rf` ile etkisizleştirdik. Upstream'e karşı bir düzeltme (ör. resume'u onarmak) istenirse ayrı iş olarak ele alınmalı — Stage-0 için gerekmiyor.
