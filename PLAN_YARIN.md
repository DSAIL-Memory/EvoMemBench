# Yarının Planı — Lokal Orkestrasyon ile Stage-0'ın Bitirilmesi

> Sahibi: Claude (lokal makinede). Kullanıcı gün boyu ulaşılamaz durumda.
> Teslim sınırı: **Stage-0 tamam + kapı karar dosyası. Stage 1'e GİRİLMEZ** —
> T8 insan kapısı akşam kullanıcıda. Sunucular (:8000 LLM, GPU1) serbestçe
> kullanılabilir (kullanıcı onayı alındı, 2026-08-14).

## Mimari

```
┌─ LOKAL (Windows, Claude Code) ────────────────────────────────┐
│  Orkestrasyon: her adım ssh ile box'ta tetiklenir,            │
│  nohup altında koşar, status/log dosyalarından izlenir.       │
│  Uzun iş ASLA canlı ssh oturumuna bağlanmaz.                  │
│                                                               │
│  ssh -o BatchMode=yes egekutlu@ozonderlab2.bogazici.edu.tr    │
└──────────────────────┬────────────────────────────────────────┘
                       │ (key auth — BU AKŞAM kurulmalı, aşağıda)
┌─ BOX (ozonderlab2, 2×4090) ──────────────────────────────────┐
│  GPU0: vLLM :8000 (Qwen3-4B-Instruct) — LLM çağrıları        │
│  GPU1: sırayla → in-process embedder (m2,m3) /               │
│                  vLLM embed :8001 (m0,t4)                    │
│  Durum: T1 PASS ✓  T2 PASS ✓  (cache + _out box'ta duruyor) │
│  Yürütücü: run_stage0.sh — resume + --redo m0,t4             │
└──────────────────────────────────────────────────────────────┘
```

Neden box'ta koşuyor da lokalde değil: T1 kalibrasyonu box'ta **float32
HFEmbedder** ile yapıldı ve GEÇTİ. Sinyal ölçümleri (m2/m3) aynı embedder +
aynı dtype ile koşmak zorunda (brief kural 5 — embedder karışımı kalibrasyonu
geçersiz kılar). Lokal GPU (5060 Ti 16GB) float32'yi alamaz; lokalde koşmak
tüm kalibrasyonu float16 ile sıfırdan yapmak demek olurdu. Tünel/lokal erişim
orkestrasyon ve analiz içindir, ölçüm bilgisayarı box'tır.

## BU AKŞAM — kullanıcının yapması gerekenler (~5 dk)

Tek kritik önkoşul SSH key. Şifreli ssh ile otomasyon imkânsız (Bash aracı
etkileşimsiz).

```powershell
# 1. Key var mı? Yoksa üret (PowerShell veya Git Bash):
ls ~/.ssh/id_ed25519.pub 2>$null; if (-not $?) { ssh-keygen -t ed25519 -f $HOME/.ssh/id_ed25519 -N '""' }

# 2. Public key'i box'a ekle (şifreyi SON KEZ gireceksiniz):
type $HOME\.ssh\id_ed25519.pub | ssh egekutlu@ozonderlab2.bogazici.edu.tr "mkdir -p ~/.ssh && chmod g-w,o-w ~ && chmod 700 ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"

# 3. Test — ŞİFRE SORMADAN 'ozonderlab2' yazmalı:
ssh -o BatchMode=yes egekutlu@ozonderlab2.bogazici.edu.tr hostname
```

**`chmod g-w,o-w ~` satırı atlanamaz:** nltk'nin uyarısı home dizininin
group/world-writable olduğunu gösterdi; sshd StrictModes bu durumda
authorized_keys'i **sessizce reddeder**. Adım 3 şifre sormaya devam ederse
sebep budur.

Ayrıca (opsiyonel, sabaha da kalabilir): box'ta `git pull` — bu akşam push
edilen düzeltmeler (nltk manuel kurulum, minimal benchmark bağımlılıkları,
`--redo` liste desteği) yarın gerekli. Ben sabah ssh ile kendim de çekerim.

## YARIN — safha safha

### Safha 0 · Bağlantı ve durum denetimi (~10 dk)
1. `ssh ... hostname` — BatchMode testi. **Başarısızsa → Yedek Plan C'ye geç.**
2. Box'ta: `git pull`, `nvidia-smi` (GPU1 boş mu? :8000 ayakta mı?),
   `cat hnav/_out/pipeline/*.status` — beklenen: T1/T2 PASS, m2 FAIL, m0/t4 SKIP.
3. Güncellenmiş setup'ı koştur: `bash hnav/deploy/setup_ozonderlab2.sh`
   → nltk punkt'ı `$NVME/nltk_data`'ya elle indirir (downloader'ın izin
   reddini bypass eder), m0/t4 için minimal langchain/faiss-cpu setini kurar.
4. Doğrulama: `python -c "import nltk; nltk.sent_tokenize('A. B.')"` ssh ile.

### Safha 1 · Stage-0'ı bitir (~2-5 saat, çoğu m3)
Tek komut, box'ta nohup altında; ben lokalden status dosyalarını izlerim:

```bash
nohup bash hnav/deploy/run_stage0.sh --redo m0,t4 \
      > hnav/_out/pipeline/console5.log 2>&1 &
```

Resume mantığı: preflight/t1/t1_smoke/t2 PASS → atlanır. m2 FAIL → yeniden
koşar (punkt artık var; `fallback_chunker=false` tuzağı tekrar denetler).
m0/t4 `--redo` ile zorlanır (deps artık var). Sıra: m2 → embed server ↑ →
m0 (S1 kapısı) → t4 (S2 kapısı) → server ↓ → m3 (~2-3k LLM çağrısı, en uzun
kalem) → m4 → report.

İzleme kadansı: ilk 15 dk her 2-3 dk'da bir log; sonrasında stage geçişlerinde.
Bir stage FAIL ederse: log'u oku, kökü düzelt, push, box'ta pull, `--redo <stage>`
ile devam. **Kapı (S1/S2/S3) ateşlerse: düzeltme YOK — durdur, Safha 2'nin
kapsamını "neden ateşledi" analizine çevir.** Kapı ateşi hata değil, ölçümün
cevabıdır.

### Safha 2 · Kapı karar dosyası (~2 saat)
Girdi: `STAGE0_REPORT.md` + `hnav/_out/*.json` (box'tan çekilir).

`KAPI_KARARI.md` (repo kökü) — akşam 10 dakikada karar verdirecek şekilde:
- Bileşen bazında GO/NO_GO tablosu + her NO_GO'nun verdikti
  (benchmark / detection / policy — asla karıştırılmaz)
- **M3 headroom**: could-change-correctness oranları = Stage-1'de kazanılabilecek
  maksimum (tezin "Başarı ölçütleri" tablosuna bağlanır)
- M1b atfedilebilirlik: geometri F1 — kazanım geometriye mi metadata'ya mı yazılır
- M2 ham-entropi verdikti (iki cevap da yayınlanabilir)
- Önerim + gerekçe + Stage-1'in ilk adımı ne olurdu (tek bileşen, hangi eşik)
- Riskler ve raporda beyan edilecek sapmalar (ör. sınırlı örneklem, chunk kaybı)

Sonuç JSON'ları `stage0_results/` dizinine kopyalanıp **bilinçli commit**
edilir (hnav/_out gitignore'da — ham veri kaybolmasın).

### Safha 3 · Kalan zaman (rapor bitmişse)
Öncelik sırasıyla: (1) tez için görselleştirme — headroom tablosu, benzerlik
dağılımları, PR eğrisi (M1b); (2) `TEZ_YOL_HARITASI.md` durum güncellemesi;
(3) Stage-1 için AÇIK SORULAR listesi (tasarım değil — kapsam kararına saygı).

## Başarısızlık kitabı

| Durum | Tepki |
| --- | --- |
| S3/S1/S2 kapısı ateşler | Boru hattı zaten durur. Düzeltme girişimi YASAK. KAPI_KARARI.md o kapının analizi olur — NO_GO yolu da savunulabilir tez bulgusu. |
| SSH sabah çalışmıyor | **Plan C:** lokal-only gün → rapor/analiz araçları, karar dosyası şablonu, görselleştirme kodu (sonuç JSON'u bekleyen). Box işleri akşama yazılı talimat olarak hazırlanır. |
| SSH gün içinde kopar | Box işleri nohup'ta — ölmez. 5 dk aralıklı yeniden deneme, dönünce status'tan devam. |
| :8000 çöker (m3 ortasında) | m3 FAIL verir; status korunur. LLM'siz kalan işler öne alınır; m3 `--redo` ile sunucu dönünce. Sunucuyu ben yeniden BAŞLATMAM — kullanıcının süreci, PID 52520. |
| GPU1'de yabancı süreç | gpu_guard zaten reddeder. Beklerim + durum notu düşerim; HNAV_FORCE_GPU kullanılmaz. |
| m2'de yine fallback_chunker=true | Setup'ın nltk adımı logu incelenir; NLTK_DATA yolu ssh ile elle doğrulanır. Tuzak asla devre dışı bırakılmaz. |

## Değişmez kurallar (brief'ten — yarın da geçerli)
- `HNAV_MODE=live` asla. `write_policy.py`/`read_policy.py` yazılmaz.
- sh_64k/sh_262k üzerinde hiçbir eşik ayarlanmaz.
- Kapı sonuçları düzeltilecek "hata" değil, raporlanacak "ölçüm"dür.
- Her anlamlı adım commit + push (box'ta üretilen sonuçlar dahil).

## Akşam döndüğünüzde göreceğiniz
1. `KAPI_KARARI.md` — 10 dakikalık okuma, net öneri
2. `STAGE0_REPORT.md` — tam Stage-0 raporu (veya kapı ateşlediyse o noktaya kadar + analiz)
3. `stage0_results/` — ham ölçüm JSON'ları, commit'li
4. Bu dosyanın sonuna eklenmiş **GÜN SONU NOTU** — ne yapıldı, ne yapılamadı, neden
