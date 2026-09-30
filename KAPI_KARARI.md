# KAPI KARARI — H-Nav Stage-0 → Stage-1 Geçiş Değerlendirmesi

> ## ⚠️ BU BELGE KISMEN AŞILMIŞTIR — 2026-08-16
>
> Bu dosya **2026-08-15 01:52 itibarıyla** doğruydu ve tarihsel karar kaydı
> olarak korunuyor. Sonraki ölçümler bazı bölümlerini geçersiz kıldı. Güncel
> durum için **`TEZ_BULGULARI.md`** (kanıt defteri) ve **`HNAV_FINAL_REPORT.md`**
> okunmalıdır.
>
> | Bu belgede yazan | Güncel durum |
> |---|---|
> | §1 satır 4: m2 **`NOT_DEGENERATE 4/4`**, margin p50 serisi | **GEÇİCİ İDİ, DÜZELTİLDİ.** Chunk gömmeleri 512 token'da kesiliyordu (~%12'si). Yeniden türetme sonrası: **2/2 kalibrasyon, yeniden kazanıldı**; sh_64k/sh_262k hâlâ L512 dönemi. Ayrıca kesme, sh_32k'yı olduğundan **kolay** göstermiş (margin p50 −%54). |
> | §1 dipnot: dondurulmuş eşikler `nmargin<0.0048` / `H_z>1.9569` | **SEVK EDİLEN MEKANİZMADA ATIL.** İşletim noktası `ambiguity_mode="none"` kullanır. Ayrıca `H_z` sh_6k'da **yapısal olarak erişilemez** (n_chunks=2 ⇒ H_z ≡ 0.36533 sabit) ve havuzlanmış percentile savunulamaz. Yalnız `r_min` (olgu tabanlı) etkilenmemiştir ve 1.1e-06 ile **sabittir**. |
> | §3: "read_policy **KOŞULLU**" | **KOŞUL SAĞLANDI ve mekanizma kuruldu, koşuldu, raporlandı.** Chunk düzeyi rerank reddedildi (sistematik zararlı); olgu düzeyi **bastırma** ön-kayıtlı held-out koşumda birincil ölçütü **GEÇTİ** (sh_64k: çakışan 17/66 → 37/66, net +20, p=1.9e-06, token −%0.31); koruyucu iddia tek soruda geçersiz. |
> | §4: Stage-1 çerçeve seçenekleri 1/2/3 | **AŞILDI.** Kullanıcı seçeneği 1'i seçti; sonuç yukarıdadır. |
> | §2 m3 okuma-yolu onarım sayıları (+6/−1 vb.) | **AŞILDI** — oracle probe ve dedektör ölçümleri (A2/A3) çok daha iyi ölçümlerdir; m3'ün LLM etiketleri L512 dönemidir. |
>
> **Değişmeyen ve hâlâ geçerli olan:** §1 satır 1–3, 5 (geometri öncülü,
> gruplama, replika sadakati, gölge nötrlüğü), §1 satır 7 (m4 / H2 GEÇMEDİ,
> LRT p=0.341), **write_policy NO_GO kalıcıdır**, ve §5'teki sapma listesi.

> **KARAR DURUMU: `TAMAMLANDI — AYRIŞIK VERDİKT (§6)`** — Stage-0'ın 10 aşaması
> da ölçüldü; `report.py --strict: COMPLETE` (2026-08-15 01:52). Bu dosya 10
> dakikada okunacak şekilde yazılmıştır. Ham veriler: `STAGE0_REPORT.md` +
> `stage0_results/final/` (7 ölçüm JSON'u, commit `e6c6d77`).

---

## 1. Bileşen bazında verdikt tablosu

| # | Bileşen | Ölçüm | Sonuç | Verdikt | Verdikt tipi |
|---|---|---|---|---|---|
| 1 | Geometri öncülü | T1/M1 | median sim ~0.964 vs kontrol ~0.60; AUC ≥ 0.9999, 4/4; S3 ateşlemedi | **GO** | detection |
| 2 | Geometri gruplama | T2/M1b | F1 0.892→0.757 (6k→262k), precision 0.83–0.90 | **GO (beyanlı)** | detection |
| 3 | Replika sadakati | m0 | top1=topk=τ=1.0000, 400/400 (fp32 sonrası) | **GO** | benchmark |
| 4 | Sinyal dejenerasyonu | m2 | **NOT_DEGENERATE 4/4** (BFCL bulgusunu çürütür); margin p50 1.235→0.318 | **GO** | detection |
| 5 | Shadow nötrlüğü | t4/S2 | off↔shadow %2.42 < off↔off %3.04; TOST ±2.0 eşdeğerlik; kullanıcı kararı | **GO (uyarlanmış kriter)** | benchmark |
| 6 | Müdahale tavanı | m3 | Aşağıda §2 — yazı-yolu tavanı ~%0–1,6; okuma-yolu karışık, 262k'da net ZARARLI | **AYRIŞIK — §6** | policy |
| 7 | Marginal-diff (H2) | m4 | delta_auc +0.0674 (in-sample; cross-validated +0.1185), yönü pozitif; ön-kayıtlı konjonksiyon **GEÇMEDİ** — düşen koşul **LRT p=0.341 ≥ 0.01** (key-clustered CI [−0.046,+0.188] 0'ı içermesi destekleyici kanıt; subset-clustered CI m4'ün kendi bayrağıyla güvenilmez, 2 küme) | **NO (ön-kayıtlı test)** | detection |
| 8 | Bütünleşik rapor | report --strict | COMPLETE, 10/10 aşama PASS | **TAMAM** | — |

**Dondurulmuş eşikler (yalnız sh_6k+sh_32k):** `nmargin<0.0048`, `H_z>1.9569`,
`r_min<0.1924`. Son koşuda bit-özdeş yeniden üretildi (`unfit_for_analysis: false`).

## 2. M3 headroom — kararın kalbi

| subset | split | write n | bayat-çakışan | müdahale (veto sonrası) | could-change-correctness | native başarısızlık (CONFLICT) | onarım yardım/zarar |
|---|---|---|---|---|---|---|---|
| sh_6k | kalib | 455 | 0.352 | 0.002 → 0.000 | — | 0.67 | 0/0 |
| sh_32k | kalib | 2.310 | 0.361 | 0.104 → **0.016** | **0.20** | 0.53 | 2/1 |
| sh_64k | doğrulama | 4.580 | 0.368 | 0.063 → **0.013** | **0.00** | 0.56 | **6/1** |
| sh_262k | doğrulama | 18.333 | 0.392 | 0.023 → **0.004** | — | 0.80 | **3/5** |

Okuma bayrakları: READ_CONFLICT/STALE/RELEVANT_BELOW_K = 1.00 her yerde
(yapısal doygunluk — ayırt edici değil); READ_MISSING 0.28→0.61; dup 0.000.

**Üç yük taşıyan gözlem:**
1. **Yazı-yolu tavanı veto sonrası %0–1,6** ve doğrulama subset'inde (sh_64k)
   could-change-correctness **0.00** — yazı müdahalesinin doğruluğa
   çevrilebileceği ölçülebilir alan YOK.
2. **Okuma-yolu onarımı ölçekle tersine dönüyor:** sh_64k'da 6 yardım / 1 zarar,
   sh_262k'da 3 yardım / **5 zarar** — koruyucu koşul harm≈0 en büyük mağazada
   İHLAL ediliyor (küçük örneklemle).
3. **Substrat gürültü tabanı ±2–4 puan** (A/A ölçümü): 100 soruluk tek koşuda
   bu tavanların hiçbiri gürültüden ayırt edilemez; her Stage-1 iddiası N≥5
   koşu + eşleştirilmiş analiz gerektirir.

## 3. KARAR (§6'da gerekçesiyle)

- **Enstrümanlar (bileşen 1–5): GO.** H-Nav'ın tespit katmanı bu arenada
  geçerli, kalibre ve nötr — tezin metodoloji bölümü sağlam.
- **Yazı-yolu canlı müdahale (write_policy): NO_GO.** Tavan yok (madde 2.1).
  Bu, tasarımın değil ARENANIN cevabı: tek-atlama fact-consolidation'da
  retrieval zaten LATEST'i buluyor; veto mekanizması doğru çalışıp geriye
  müdahale edilecek bir şey bırakmıyor. NO_GO yolu tezde savunulabilir bulgudur.
- **Okuma-yolu canlı müdahale (read_policy): KOŞULLU — kullanıcı kararı gerekli.**
  Pozitif sinyal var (native başarısızlık 0.53–0.80, sh_64k onarımı 6/1) ama
  262k'da net zarar ve tüm örneklemler gürültü tabanının altında. Stage-1'e
  ancak YENİDEN TASARLANMIŞ bir çerçeveyle girilebilir (aşağıda §4 önerisi).
  **Kullanıcı yetkilendirmesindeki "açık GO → otomatik devam" koşulu
  SAĞLANMADI; Stage-1 ajanı başlatılMAdı, kullanıcı bekleniyor.**

## 4. Önerilen Stage-1 çerçevesi (kullanıcı onayına sunulur)

Orijinal iddia ("H-Nav doğruluğu artırır") bu arenada ölçülebilir değil.
Kanıtın işaret ettiği savunulabilir üç alternatif, tercih sırasıyla:

1. **Okuma-yolu, sh_64k-sınıfı ölçek, çok-koşulu protokol:** onarım stratejisi
   262k zararını açıklayacak şekilde revize edilir (ör. yalnız yüksek-güven
   çakışmalarda müdahale); N≥5 koşu/kol, eşleştirilmiş analiz, ön-kayıt.
   Başarı: doğruluk Δ > gürültü tabanı VE harm≈0 VE token nötr.
2. **Tespit-katmanı tezi (müdahalesiz):** Stage-0'ın kendisi ana katkı olur —
   geçerli, kalibre, nötr bir bellek-gözetim katmanı + iki substrat bulgusu
   (bf16/S1, nondeterminizm). Ek koşu gerektirmez; en düşük risk.
3. **İkincil arena (CrossEp-Know) pilotu:** çapraz-epizot bellekte müdahale
   tavanı bu arenadakinden farklı olabilir; küçük ön-çalışma tavanı ölçer,
   sonra karar verilir. (ICC=0.346 kümeleme uyarısı geçerli.)

## 5. Beyan edilecek sapmalar (nihai liste)

1. Embed fp32 servis (native bf16) — kullanıcı onaylı.
2. S2 kriteri: bayt-özdeşlik → ön-kayıtlı istatistiksel eşdeğerlik (`7cb8323`); deterministik substrat kurulamadı (kanıtıyla).
3. m3 LLM'i :8003'te (`--max-model-len 106496 --kv-cache-dtype fp8 --enforce-eager`); yapısal istem tavanı ~104,6k token ÖLÇÜLDÜ (top-10 chunk 49,6–52,2k + ~14k injected fact).
4. M0 400/400 (arena eksiksiz) — kullanıcı onaylı.
5. Embed sunucu `--max-model-len 16384` (davranışsal etkisiz).
6. Chunker sınır düzeltmeleri (fact kaybı + sarkan seri), test edilmiş.
7. m2/m3 statüleri manuel-koşu yorumlu (runner GATE kilidi nedeniyle); ledger tutarlı.
8. Kullanıcının :8000 durdurma talimatı izin katmanı engeli nedeniyle uygulanamadı; sunucu HÂLÂ AÇIK (kullanıcı `kill 52259` komutunu kendisi koşmadı). Stage-0 sonuçlarına etkisi yok (m3 :8003 kullandı).

## 6. Gerekçeli sonuç

Stage-0 tam olarak yapması gerekeni yaptı: **enstrümanları doğruladı ve
müdahale tavanını ÖLÇTÜ.** Tavan, yazı yolunda yok; okuma yolunda var ama
ölçekle tersine dönüyor ve mevcut örneklem gürültünün altında. "GO/NO_GO"
ikiliği bu kanıtı taşımıyor; dürüst cevap ayrışıktır:

> **KARAR: Enstrümanlar GO · write_policy NO_GO · read_policy KOŞULLU.**
> Stage-1'in çerçevesi (§4'teki 1/2/3) kullanıcı seçimine bağlı — otomatik
> devam tetiklenmedi. NO_GO bileşeni dahil her sonuç tez bulgusudur.

*Orkestratör taslağı, 2026-08-15 02:0x. T8 nihai onayı insana aittir.*
