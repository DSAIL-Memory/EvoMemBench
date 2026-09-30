# STAGE-1 PLANI — Okuma-Yolu Seçici Onarım Kampanyası

> **REVİZYON R2 — 2026-08-15 (bağlayıcı; aşağıdaki R1 metni provenance için korunur).**
>
> **Bulgu (iki kez bağımsız doğrulandı, `stage0_results/question_strata.json`):**
> sh_6k'nın 100 sorusu çakışan/çakışmayan anahtara göre ayrıştırıldığında —
> çakışmayan anahtar (n=26): **8 koşunun 8'inde de 26/26 doğru**; çakışan anahtar
> (n=74): **0–5 doğru**. 575 çakışan-soru hatasının **572'si anahtarın BAYAT
> değerini** üretiyor (3 off-list, 0 boş). Yani: (a) arenanın manşet doğruluğu
> neredeyse tamamen çakışma İÇERMEYEN sorulardan geliyor (çakışan-only doğruluk
> %4–19), (b) model, istem açıkça "büyük seri numarası daha yeni" dese de
> supersession kuralını ~%95 uygulamıyor, (c) **tavan gerçek ve büyük** (sh_6k'da
> 100 sorunun 71'i yanlış, tek bir hata kipi).
>
> **Sonuç:** R1'in tek mekanizması (chunk düzeyinde YUKARI rerank) bu kaldıraca
> ulaşamıyor — 162 hücrelik kalibrasyon net>0 vermedi ve kalibrasyon split'inde
> retrieval zaten eksiksiz (n_chunks 2/9 ≤ top_k 10).
>
> **Kullanıcı kararları (2026-08-15, R2):**
> 1. **Mekanizma kümesi GENİŞLETİLDİ** — rerank'e ek olarak **bayat kaydı bastırma
>    (suppress)** ve **yerleşim (LATEST'i sona / recency)** yetkilendirildi.
>    Önce oracle probe tavanı ölçer; **yalnız probe'un kanıtladığı mekanizma**
>    uygulanır ve ön-kayda girer.
> 2. **512-token kesme kusuru DÜZELTİLECEK** — `HFEmbedder` 512'de kesiyor, oysa
>    chunk'lar 4096 token; M2/M3 sinyalleri ve dondurulmuş eşikler chunk'ların ilk
>    ~%12'sinden hesaplanmış, canlı yol ise benchmark'ın tam-uzunluk sıralamasını
>    kullanıyor (m0'ın 1.0000 sadakati bunu KAPSAMIYOR — benchmark'ın kendi
>    vektörlerini yeniden kullanıyordu). Düzeltilip **eşikler yalnız kalibrasyon
>    split'inde yeniden türetilecek** ve düzeltme beyan edilecek.
> 3. **sh_64k ön-kaydı GERİ ÇEKİLDİ** — gerekçesiyle kayda geçirilerek. Yeni
>    ön-kayıt probe sonrası yazılır; zarar tavanları ve güç hesabı ölçülen
>    **%3,3/soru gürültü tabanının üstünde** olacak.
>
> **R2 sırası:** oracle probe (kalibrasyon split'i) → mekanizma seçimi →
> uygulama → YENİ ön-kayıt → kampanya. Değişmeyenler: `write_policy.py` kalıcı
> yasak, sh_64k kampanyaya kadar dokunulmaz, eşikler yalnız sh_6k+sh_32k'da,
> H_raw karar beslemez, BFCL yasak.

---

> Dayanak (R1): `KAPI_KARARI.md` (T8 ayrışık verdikt) + kullanıcı kararları (2026-08-15,
> 8 soruluk plan oturumu). Kapsam: **yalnız okuma yolu, yalnız rerank, yalnız sh_64k
> doğrulama**. write_policy KALICI YASAK (NO_GO). Takvim: bu hafta sonu.

---

## 0. Kullanıcı kararlarının kaydı (bağlayıcı)

| Karar | Seçim |
|---|---|
| Mekanizma | **Yalnız rerank** (LATEST'i öne al). Filter/inject/merge YOK. |
| Kapı ön-koşulu | Rerank kararı iki aşamalı doğrulamadan geçmek ZORUNDA: **(1) geometrik filtre** (benzerlik + R metriği) → **(2) çift-yönlü NLI** (A⇒B ve B⇒A çelişki/içerme). Bu boru hattı **Agent B başlamadan ÖNCE inşa edilir**; bitince Agent B **otomatik** başlar (kullanıcı yetkisi). |
| Forensik | Yapılmayacak — teoriden tasarım (kullanıcı kararı; sınırlılık olarak beyan edilir). |
| Kapı ayarı | **Coverage-balanced** — kalibrasyon split'inde (sh_6k+sh_32k) beklenen net-yardımı maksimize edecek şekilde ayarlanır; sh_64k'ya DOKUNULMAZ. |
| Değerlendirme kapsamı | **Yalnız sh_64k**, ön-kayıtlı doğrulama kampanyası. |
| NLI motoru | **Adanmış cross-encoder** (DeBERTa-v3 MNLI sınıfı, ~400MB), çift yönlü; cevap LLM'inden bağımsız. |
| Protokol | **N=7 koşu/kol** (off vs live), eşleştirilmiş analiz. Başarı: **Δdoğruluk ≥ +3 puan** (eşleştirilmiş, gürültü tabanı sd≈1.5 üstünde) **VE** zarar kriteri: koşu-başına harmed≤helped VE havuzlanmış zarar oranı ≤%2 **VE** token nötr-veya-daha-iyi. |
| Takvim | Hafta sonu: 15-16 Ağu inşa+kalibrasyon, 17 Ağu doğrulama kampanyası, 18 Ağu sonuç. |
| Tez yapısı | **ALL-IN okuma-yolu başarısı** (kullanıcının bilinçli tercihi; hedge yok). |

## 1. Mimari

```
soru → retriever (dokunulmadı) → aday sıralama
         │
         ▼
   [READ GATE — inşa ön-koşulu]
   1. Geometrik filtre: aday çiftler (chunk_i, chunk_j) için
      benzerlik (cos) + R metriği (span-residual, r_min) →
      "çakışma adayı grubu" tespiti. Dondurulmuş Stage-0
      eşikleri taban; coverage-balanced ayar kalibrasyonda.
   2. Çift-yönlü NLI: cross-encoder ile (eski⇒yeni, yeni⇒eski)
      çelişki/içerme skorları. İki yön de eşiği geçerse
      "doğrulanmış çakışma" (novelty/supersession onaylı).
         │
         ▼ (yalnız doğrulanmış çakışmalarda)
   RERANK: LATEST-serili chunk'ı bayat rakiplerinin üstüne al.
   İçerik değişmez, chunk eklenmez/çıkarılmaz, token nötr.
         │
         ▼
   cevap LLM'i (aynı istem şablonu, yeniden sıralanmış bağlam)
```

## 2. İş sırası

### Faz A — READ GATE inşası (Agent A / builder; Agent B ÖNCESİ zorunlu)
1. `hnav/core/read_gate.py` — benchmark-agnostik: geometrik filtre (mevcut
   `geometry`/`retrieval_signals` üzerine) + NLI istemci arayüzü. Kapalı-form
   testler (test felsefesine uygun: bilinen çakışma çiftinde iki yön de
   contradiction; parafraz çiftinde entailment; ilgisiz çiftte nötr).
2. NLI servisi: DeBERTa-v3 MNLI sınıfı cross-encoder box'a indirilir
   (`$NVME/models`), lazy-load, GPU1 (küçük model, embedder ile sığar) veya CPU.
   `check_env`'e NLI denetimi eklenir.
3. **Protokol geçişi (bilinçli, commit mesajında beyanlı):**
   `read_policy.py` yasağı T8 sonrası KALKTI (yalnız read; write yasağı sürer).
   `test_no_raw_entropy_in_policy.py` buna göre güncellenir: write_policy
   yasağı test edilmeye devam eder; read_policy artık var olabilir ama
   H_raw kullanamaz (mevcut H_z kuralı aynen).
4. Tüm suite yeşil + box'ta doğrulama → **Agent B otomatik başlar.**

### Faz B — Stage-1 tasarım + ön-kayıt (Agent B)
1. `hnav/core/read_policy.py`: read_gate çıktısı → rerank kararı.
   Coverage-balanced eşik ayarı YALNIZ sh_6k+sh_32k üzerinde.
2. Canlı kablolama: `HNAV_MODE=live` yalnız okuma-yolu rerank'i etkinleştirir;
   `config.require_not_live()` Stage-0 korumasının kaldırılması bilinçli commit.
   Off-mode bayt-nötrlüğü korunur (S2 çerçevesi).
3. Stage-1 sunucusu: tek konfigürasyon dondurulur (:8003 mirası,
   `--max-model-len` sh_64k tavanına göre, fp8 KV gerekmiyorsa sade). A/A
   gürültü ölçümü BU konfigürasyonda tekrarlanır (5 koşu) — taban güncellenir.
4. **Ön-kayıt dosyası** (`stage0_results/stage1_preregistration.md`):
   N=7×2, eşleştirilmiş test, Δ≥3, zarar kriteri, token kriteri, analiz kodu —
   kampanyadan ÖNCE commit. S2 disiplini aynen.

### Faz C — Doğrulama kampanyası + analiz (Agent B → Agent C)
1. sh_64k: 7×off + 7×live, aynı sunucu, sıra randomize/interleaved.
2. Analiz ön-kayıtlı koda göre; sonuç ne çıkarsa o raporlanır.
3. Agent C karşılaştırma raporu (Türkçe): doğruluk + token + zarar, katmanlı;
   tez bölümü taslağı.

## 3. Değişmeyen kurallar
- sh_64k üzerinde HİÇBİR ayar yapılmaz; kampanya tek atıştır (7+7 koşu = tek
  ön-kayıtlı kampanya).
- write_policy.py yasak (NO_GO verdikt). H_raw karar beslemez. `_cache/emb` korunur.
- Kullanıcının :8000 süreci bize ait değil (hâlâ açık; Stage-1 kendi sunucusunu kullanır).
- Her anlamlı adım commit+push; kapı/kriter değişiklikleri yalnız kullanıcıyla.

## 4. Beyan edilecek sınırlılıklar (şimdiden bilinen)
- 262k zararı forensik yapılmadan çözülmemiş kabul edilir; sh_64k-sınırlı iddia.
- Coverage-balanced kapı, precision-first'e göre daha yüksek zarar riski taşır —
  zarar kriteri bu yüzden ön-kayıtta sert (havuzlanmış ≤%2).
- ALL-IN tez yapısı: null/negatif sonuçta tezin yeniden çerçevelenmesi gerekir
  (kullanıcının bilinçli tercihi).
