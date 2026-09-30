# T11 KALİBRASYON "NULL"UNUN ÇÖZÜMÜ — bir hiçlik değil, yönlü bir bulgu

> Kaynak: `hnav/_out/stage1_calibration.json` (box, commit `9e0af72`, 2026-08-15
> 05:21 UTC, `smoke:false`, embed `float32`, cache_misses **0**, 469 LLM çağrısı,
> 162 hücre, 200 kalibrasyon sorusu). Analiz kutu erişimi döndükten sonra
> (2026-08-15 17:50) yapıldı. Kalibrasyon split'i dışına çıkılmadı.

---

## 1. Soru

T11 kalibrasyonu "net > 0 olan uygun işletim noktası yok — RAPORLA VE DUR"
verdi. Denetçi haklı olarak şunu işaret etti: **üç farklı dünya aynı çıktıyı
üretir** ve üçü apayrı sonuçlara götürür —

| desen | anlamı |
|---|---|
| tüm hücreler sıfır (helped=harmed=0) | müdahale grader'a hiç ulaşmadı → harness kusuru |
| dengeli çevirmeler (helped ≈ harmed) | gürültüyle sınırlı, bilgisiz null |
| sistematik negatif | müdahale **zararlı** → yönlü, raporlanabilir bulgu |

## 2. Cevap: üçüncüsü

```
162 hücre:   129 net negatif   ·   12 dengeli   ·   21 net pozitif   ·   0 feasible
net dağılımı: -5:15  -4:24  -3:18  -2:42  -1:30  0:12  +1:3  +2:3  +3:15
ortalama net -1.76 · medyan -2 · aralık -5..+3
toplam helped 582  ·  toplam harmed 867
sıra değişen soru sayısı: her hücrede 26–115 / 200 (hiçbiri sıfır değil)
```

**Harness kusuru elenmiştir:** hiçbir hücre "sıfır değişiklik" değil; her
hücrede 26–115 soru gerçekten yeniden sıralandı ve notlandı (469 farklı istem).

## 3. Asıl bulgu — özne elemesi AÇIKKEN rerank sistematik olarak ZARARLI

| `pair_filter` | hücre | net ort. | net medyan | net>0 hücre | toplam helped | toplam harmed | yanlış-doğrulama medyanı |
|---|---|---|---|---|---|---|---|
| **AÇIK** (özne elemesi) | 81 | **−2.63** | −3 | **0** | 228 | **441** | **0.000** |
| KAPALI | 81 | −0.89 | −1 | 21 | 354 | 426 | 0.863 |

Okunuşu:

1. **Doğru hedeflenen** (precision 1.00, yanlış-doğrulama 0.000) chunk rerank'i
   **yardım ettiğinin ~2 katı zarar veriyor** (228 vs 441) ve 81 hücrenin
   **hiçbirinde** net pozitif değil.
2. Eleme kapalıyken — yani müdahalelerin ~%86'sı **sahte** çakışmalar üzerine
   yapılırken — sonuç nötre yaklaşıyor ve 21 hücre net>0 çıkıyor. Bu "gelişme"
   çakışma çözümüne atfedilemez; rastgeleye yakın yeniden sıralamanın gürültü
   bandında salınmasıdır (en iyi hücre: helped 10 / harmed 7 / 115 değişen).
3. **Feasibility'yi öldüren kriter:** 21 net-pozitif hücrenin **hepsi**
   `false_verified_rate` eşiğinde (0.82–0.94 ≫ 0.05) elendi; çoğu ayrıca zarar
   tavanını aştı (harmed 5–7 > 4). Yani "uygun nokta yok" sonucu, ön-kayıtlı
   kalite kriterinin sahte-çakışma üzerine kurulu kazanımları reddetmesidir —
   kriter **doğru** çalışmıştır.

## 4. Mekanizma (kanıtla tutarlı açıklama)

Kalibrasyon split'inde **retrieval eksiksizdir** (n_chunks 2 ve 9 ≤ top_k 10):
tüm chunk'lar her istemde var, dolayısıyla rerank yalnız **sırayı** değiştirir,
içeriği değil. Buna rağmen her hücrede soruların %13–57'sinin cevabı değişiyor —
**model konum duyarlıdır**. Ama LATEST taşıyan chunk'ı öne almak net **zarar**
veriyor.

En tutarlı açıklama **granülerlik uyumsuzluğu**: bir chunk ~228–257 olgu taşır.
Bir anahtarın sırasını düzeltmek için chunk'ı öne almak, ilgisiz yüzlerce olgunun
göreli sırasını da bozar. Etkin sinyal/gürültü ≈ 1/250. Müdahale, çakışan
soruda kazandığından fazlasını **çakışmayan** sorularda (doğal doğruluk %100)
kaybediyor olabilir — bu, katman-kırılımlı ölçümle doğrudan sınanabilir ve
probe'un `unique` katmanı tam olarak bunu ölçüyor.

## 5. Tez için sonuç

**Yazılabilir olumsuz sonuç (dar ve doğru):**

> Kanonik çakışma-çözümü arenasında, doğrulanmış bir çakışma dedektörüyle
> (precision 1.00) yönlendirilen **chunk düzeyinde** yukarı yeniden sıralama,
> 162 kapı işletim noktasının hiçbirinde net fayda vermedi ve toplamda yardım
> ettiğinin iki katı zarar verdi (228 vs 441). Müdahale granülerliği,
> çakışmanın granülerliğiyle (olgu) eşleşmediğinde, yönetişim katmanının
> doğruluğu artırmak yerine düşürmesi beklenir.

**Yazılamayacak olan:** "H-Nav okuma yolu işe yaramaz." Test edilen tek şey
chunk düzeyinde tek yönlü permütasyondur; olgu düzeyi bastırma ve yerleşim
mekanizmaları ölçülmemiştir (probe koşuyor).

**Metodolojik yan bulgu:** eleme kapalıyken görünen "kazanımlar", eleme açıkken
kayboluyor. Yani **kötü bir dedektör, gürültü bandında sahte kazanım üretir.**
Müdahale değerlendirmelerinde dedektör precision'ı raporlanmadan bildirilen
kazanımlar güvenilmezdir — bu, RAG bellek literatürüne doğrudan aktarılabilir
bir uyarıdır.

## 6. Kayıt için: kriter tasarımı dersi

Zarar tavanı `harmed ≤ %2 × 200 = 4` idi; ölçülen substrat gürültüsü ise
soru başına ~%3.3, yani 200 soruda ~6.6 beklenen çevirme. **Tavan gürültü
tabanının altındaydı** — hiçbir müdahale, mükemmel olsa bile, bu koşulda
geçemezdi. Yeni ön-kayıtta zarar kriteri gürültü tabanının üstünde ve
katman-kırılımlı tanımlanacak (`unique` katmanında sıfır gürültü ölçüldüğü için
oradaki her düşüş sinyaldir).
