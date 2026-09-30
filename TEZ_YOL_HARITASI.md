# Tez Yol Haritası — H-Nav × EvoMemBench

> Yüksek lisans tezi çalışma dokümanı. Teknik ayrıntı için: `HNAV_AGENT_BRIEF.md`
> (spec), `RUNBOOK.md` (operasyon), `STAGE0_REPORT.md` (üretilecek).

## Tezin iddiası

**H-Nav:** öz-evrilen (self-evolving) vektör belleğe sahip LLM ajanlarında,
her bellek *yazımını* ve *okumasını* embedding geometrisi ve retrieval
sinyalleriyle denetleyen bir yönetişim katmanı. Tez, bu mekanizmanın
tasarımını, geçerliliğini ve etkisini savunur.

Hedef arena: EvoMemBench `Conflict_Resolution` — soruların **%65–77'si**
çakışan (güncellenen) bir fact üzerine kurulu. Yani "eski bilgi yeniyi
gölgeliyor" hata sınıfı burada marjinal değil, **baskın**. (Önceki BFCL
denemesi bu sınıf %3.5'ken null sonuç vermişti — arena seçimi tezin
gerekçesinin parçası.)

## İki aşamalı yapı — ve neden bu sırayla

```
Stage 0  "İyileştirme MÜMKÜN mü?"     ← şu an burdayız (otomatik pipeline koşuyor)
   ├─ T1  Geometri öncülü sağlam mı? (kill switch)
   ├─ M1b Kazanım geometriye mi atfedilir, metadata'ya mı? (hakem savunması)
   ├─ M0/T4  Enstrümantasyon sonucu değiştirmiyor mu? (byte-identical)
   ├─ M2  Sinyaller ayırt edici mi?
   ├─ M3  HEADROOM: müdahale kaç soruyu yanlıştan doğruya çevirebilirdi?
   │      → Stage 1'de kazanılabilecek maksimumun sayısal üst sınırı
   └─ M4  Diff-geometri bilgi katıyor mu? (ΔAUC, LRT)
        ↓
      T8 RAPORU → GO / NO_GO   ← insan kararı (ben + danışman)
        ↓ GO ise
Stage 1  "İyileştiriyor mu?"          ← asıl baseline karşılaştırması
   ├─ write_policy / read_policy implementasyonu
   ├─ HNAV_MODE=off (baseline)  vs  HNAV_MODE=live (H-Nav)
   ├─ Aynı subset, aynı evaluator, aynı tohum — T4 harness'i yeniden kullanılır
   └─ Kalibrasyon sh_6k+sh_32k'da donar, doğrulama sh_64k+sh_262k'da tek atış
```

Stage 0 olmadan Stage 1 koşmak, BFCL'deki hatayı tekrarlamak olurdu:
müdahale edilecek bir şey olduğunu kanıtlamadan müdahaleyi ölçmek.

## Başarı ölçütleri (Stage 1)

| Ölçüt | Metrik | Kaynak |
| --- | --- | --- |
| **1. Doğruluk artışı** | `substring_exact_match`: live − baseline farkı, subset bazında, bootstrap CI ile | benchmark çıktı dosyaları |
| **2. Token verimliliği** | ajan + bellek modülü toplam token: aynı doğrulukta ↓, veya doğruluk ↑ iken ≈ | `stats.tokens.*` alanları |
| Koruyucu koşul | harm rate ~0 — H-Nav'ın bozduğu soru sayısı ihmal edilebilir olmalı (Wilson üst sınırı, protokoldeki eşik) | M3 ile aynı yöntem |

İkisi birlikte raporlanır: doğruluğu token pahasına ya da tokenı doğruluk
pahasına almak "başarı" sayılmaz.

## Takvim (1–3 ay)

| Hafta | İş |
| --- | --- |
| 1 | Stage-0 pipeline'ın bitmesi + T8 raporu + kapı kararı (danışmanla) |
| 2–3 | GO ise: `write_policy`/`read_policy` — önce yalnızca **en yüksek headroom'lu tek bileşen** |
| 3–5 | Kalibrasyon split'inde eşik ayarı → dondur → doğrulama split'inde tek koşu |
| 5–6 | Ablation: M1b sonucuna göre "geometri vs metadata" karşılaştırma kolu |
| 6+ | (zaman kalırsa) CrossEp-Know ikincil arena; yazım |

NO_GO çıkarsa plan B hazır: üç verdikt (benchmark / detection / policy) hangi
katmanın çalışmadığını ayırır — bu da savunulabilir bir tez bulgusudur ve
yazım doğrudan başlar.

## Şu anki durum

- [x] T0 doğrulandı (veri sağlam, base rate %65–77)
- [x] Stage-0 kodu + 153 test + gözetimsiz pipeline (`run_stage0.sh`)
- [ ] Pipeline koşuyor → `hnav/_out/pipeline/SUMMARY.md`
- [ ] T8 raporu → kapı kararı
- [ ] Stage 1 (bu dokümanın "Başarı ölçütleri" bölümüne göre)

## Sabah kontrol listesi

```bash
cat hnav/_out/pipeline/SUMMARY.md      # her aşama PASS mı?
cat STAGE0_REPORT.md                   # varsa: M3 headroom tablosuna bak
```

M3'teki *could-change-correctness* oranı = Stage 1'de kazanılabilecek
maksimumun ön izlemesi. O sayı anlamlıysa Stage 1'e gireriz; sıfıra yakınsa
canlı deneye hiç girmeden nedenini raporlarız.
