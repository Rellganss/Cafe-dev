# Cafe Detection System

Deteksi: **orang masuk**, **transaksi (depan kasir)**, **durasi duduk**, **heatmap**, **peak hour**.
Stack: YOLO11 (COCO person) + ByteTrack + Supervision zones. ROI & garis **custom** via editor.

## Kenapa baseline YOLO+ROI (bukan train custom)

- Person detection: COCO udah solid, train custom buang waktu.
- "Duduk" / "transaksi" = posisi + waktu di ROI, bukan klasifikasi aksi → gak butuh pose model.
- Zone = polygon manual sekali setup per kamera. Murah, akurat, robust.

## File

| File | Fungsi |
|------|--------|
| `config.yaml` | Semua setting: RTSP, model, zone, garis, threshold |
| `setup_zones.py` | Editor interaktif gambar ROI + garis pakai mouse |
| `zones.py` | Logika line-count, dwell kasir/kursi, heatmap |
| `main.py` | Pipeline utama: detect → track → log CSV → video → heatmap |

## Pakai

1. Set RTSP di `config.yaml` (`source.rtsp`).
2. Gambar zone + garis:
   ```
   python setup_zones.py
   ```
   `l`=garis masuk, `p`=zona kasir, `s`=zona kursi, klik titik, ENTER simpan shape, `w` tulis ke config, `q` keluar.
3. Jalanin:
   ```
   python main.py                # RTSP live
   python main.py --source file  # test pakai file video
   python main.py --no-window    # headless/server
   ```
   `q` di window buat stop. Output ke `output/`.

## Output

- `output/annotated.mp4` — video overlay (box, ID, dwell detik, zone, counter).
- `output/events.csv` — `ts,event,zone,track_id,dwell_sec`. Event: `enter/exit` (garis), `transaction` (kasir), `seated` (kursi).
- `output/peak_hourly.csv` — entri per jam → peak hour.
- `output/heatmap.png` — heatmap posisi orang ditumpuk di frame.

## Tuning GTX 1650 (4GB)

- `model.weights: yolo11n.pt` default. Naik `yolo11s.pt` kalau fps cukup.
- Lag → `processing.frame_skip: 2` (proses tiap 2 frame).
- `processing.resize_width: 1280` → turunin ke 960 buat lebih kenceng.
- `tracker.frame_rate` samain sama fps stream biar dwell time akurat.

## Logic threshold (`config.yaml > logic`)

- `cashier_min_dwell_sec` — dwell di kasir ≥ ini = transaksi (buang yg lewat doang).
- `cashier_max_dwell_sec` — dwell > ini = staff/nongkrong, **bukan** transaksi (0=off). Filter staff yg nempel kasir.
- `seat_min_dwell_sec` — kaki **diam** ≥ ini = duduk beneran.
- `seat_move_eps_px` — gerak kaki antar-frame < ini dianggap diam. Naikin kalau kamera goyang/jauh.
- `zone_exit_grace_sec` — toleransi orang ilang sesaat sebelum dianggap keluar zona (anti dwell pecah).
- `min_box_area_px` — buang bbox < ini px² (deteksi jauh palsu). 0=off.
- `heatmap_gamma` — <1 angkat area sepi; `heatmap_decay` >0 bikin heatmap "lupa" pelan.

## Logika dwell (yg bikin akurat)

- **Duduk = kaki diam**, bukan sekadar ada di zona. Orang lewat/berdiri-geser gak kehitung duduk.
- **Debounce exit**: deteksi miss 1-2 frame atau jitter batas zona TIDAK reset dwell.
- **Filter staff**: transaksi cuma dihitung kalau dwell di rentang `[cashier_min, cashier_max]`.
- **1 event per kunjungan**, `dwell_sec` = total waktu di zona. Orang yg masih di zona pas stream stop tetap ke-log (flush).
- Anchor semua = **kaki** (bottom-center): garis & zona dihitung di posisi kaki, bukan kepala/torso.

## Limitasi (batas wajar)

- Occlusion berat → ID bisa switch (dwell reset). `lost_track_buffer` udah dinaikin (jembatani ~3s). Lebih bagus: kamera angle tinggi/top-down.
- Duduk vs berdiri-diam-lama: dibedain via gerak kaki, tapi orang berdiri mematung 30s di zona kursi masih bisa false. Rare.
- Garis masuk: arah in/out ikut urutan titik. **Panah kuning "IN"** di overlay/editor nunjukin sisi masuk. Kebalik → pencet `f` di editor, atau set `flip: true` di config garis itu.
