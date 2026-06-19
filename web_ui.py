"""Web dashboard buat dev — live feed + stats di browser (ganti cv2.imshow window).

Run: python web_ui.py                 (RTSP dari config)
     python web_ui.py --source webcam (webcam laptop, --cam 0 default)
     python web_ui.py --source file   (test pakai file video)
     python web_ui.py --port 8000

Buka: http://localhost:5000
  - Live video annotated (MJPEG, auto-stream).
  - Counter IN / OUT / now realtime.
  - Tabel event terakhir (enter/exit/transaction/seated).
  - Heatmap live + peak hour.

Pipeline jalan di background thread; Flask cuma nyajiin frame+stats.
Gak ganggu main.py/zones.py — ini wrapper terpisah.
"""
import os
import io
import csv
import time
import argparse
import threading
from collections import defaultdict, deque
from datetime import datetime

import cv2
import yaml
import numpy as np
import supervision as sv
from flask import Flask, Response, jsonify, render_template_string, request
from ultralytics import YOLO

from zones import ZoneManager
from main import load_cfg, open_stream, LatestFrame
from storage import Store

STORE = Store()              # SQLite persist event + metrik harian
# Katalog COCO relevan kafe (id -> nama). Dipakai label gambar + UI on/off class.
COCO_NAMES = {
    0: "orang", 41: "cup", 56: "kursi", 57: "sofa", 60: "meja",
}
FURN_NAMES = COCO_NAMES      # alias: nama buat gambar kotak furnitur/objek


def is_open(spec):
    """spec '' -> 24 jam. 'HH:MM-HH:MM' -> True kalau jam sekarang di rentang (support lewat tengah malam)."""
    if not spec:
        return True
    try:
        a, b = spec.split("-")
        sh, sm = (int(x) for x in a.split(":"))
        eh, em = (int(x) for x in b.split(":"))
        now = datetime.now()
        cur, start, end = now.hour * 60 + now.minute, sh * 60 + sm, eh * 60 + em
        return start <= cur < end if start <= end else (cur >= start or cur < end)
    except Exception:
        return True

# ---- shared state (pipeline thread -> Flask) ----
STATE = {
    "jpeg": None,            # bytes annotated frame terakhir
    "counts": {"entered": 0, "exited": 0, "now": 0},
    "events": deque(maxlen=80),
    "peak": defaultdict(int),
    "fps": 0.0,
    "wh": (0, 0),
    "source": "",
    "alive": False,          # pipeline thread hidup?
    "stale": False,          # frame basi (stream putus)?
    "occ": {"seats_total": 0, "seats_busy": 0, "seats_pct": 0, "cashier_now": 0},
    "alerts": [],            # peringatan live (antri kasir, dll)
}
LOCK = threading.Lock()
STOP = threading.Event()
RESTART = threading.Event()  # set dari /api/restart -> pipeline re-enter baca config baru (model/source/tracker)

# ---- editor zona (browser gambar -> config -> reload ZoneManager live) ----
ZM = {"zm": None}            # ZoneManager aktif (di-swap pas edit)
ZLOCK = threading.Lock()     # serialize edit config + rebuild zm
EDIT = {"cfg": None, "path": "config.yaml", "wh": (0, 0)}

app = Flask(__name__)


def _rebuild_zm():
    """Bikin ZoneManager baru dari EDIT['cfg'] + swap ke pipeline (reset count/track)."""
    W, H = EDIT["wh"]
    if W and H:
        ZM["zm"] = ZoneManager(EDIT["cfg"], (W, H))


# ---------- multi-kamera (switchable, zona per-kamera) ----------
def _ensure_cameras(cfg):
    """Jamin cfg punya list 'cameras' + 'active_camera'. Migrasi config lama (source+zona top-level) -> 1 kamera."""
    cams = cfg.get("cameras")
    if not cams:
        src = cfg.get("source", {}) or {}
        cfg["cameras"] = [{
            "id": "cam1", "name": "Kamera 1", "type": "rtsp",
            "address": src.get("rtsp", "") or "",
            "lines": cfg.get("lines", []) or [],
            "zones": cfg.get("zones", []) or [],
            "detect_roi": cfg.get("detect_roi", []) or [],
        }]
        cfg["active_camera"] = "cam1"
    elif not cfg.get("active_camera"):
        cfg["active_camera"] = cams[0]["id"]
    return cfg


def _active_cam(cfg):
    _ensure_cameras(cfg)
    aid = cfg.get("active_camera")
    for c in cfg["cameras"]:
        if c["id"] == aid:
            return c
    return cfg["cameras"][0]


def _load_active_zones(cfg):
    """Salin zona kamera aktif -> top-level (dipakai ZoneManager + editor zona)."""
    c = _active_cam(cfg)
    cfg["lines"] = c.get("lines", []) or []
    cfg["zones"] = c.get("zones", []) or []
    cfg["detect_roi"] = c.get("detect_roi", []) or []


def _sync_active_zones(cfg):
    """Salin top-level zona -> kamera aktif (dipanggil sblm simpan, biar edit zona nyimpen ke kamera benar)."""
    if not cfg.get("cameras"):
        return
    c = _active_cam(cfg)
    c["lines"] = cfg.get("lines", []) or []
    c["zones"] = cfg.get("zones", []) or []
    c["detect_roi"] = cfg.get("detect_roi", []) or []


def _new_cam_id(cfg):
    ids = {c["id"] for c in cfg.get("cameras", [])}
    n = 1
    while f"cam{n}" in ids:
        n += 1
    return f"cam{n}"


CONFIG_HEADER = """# ================= CONFIG CAFE DETECTION =================
# Auto-generate pas gambar zona di dashboard. Bagian lines/zones/detect_roi
# bakal ketimpa kalau gambar ulang di browser.
#
# FUNGSI TIAP BAGIAN:
#   detect_roi            -> AREA LANTAI. Cuma orang di dalam sini dideteksi.
#                            Gunanya: buang orang di POSTER/BANNER dinding.
#                            Kosong = deteksi seluruh layar.
#   lines (garis)         -> GARIS MASUK. Orang nyebrang = dihitung MASUK/KELUAR.
#   zones type: cashier   -> AREA KASIR. Diam di sini = dihitung TRANSAKSI.
#   zones type: seat      -> AREA KURSI. Duduk di sini = dihitung DURASI DUDUK.
#   zones type: staff     -> ZONA KERJA. Orang di sini = STAFF, TIDAK dihitung customer.
#
# SETELAN ANGKA (logic):
#   cashier_min_dwell_sec -> min detik di kasir biar dihitung transaksi
#   cashier_max_dwell_sec -> lebih dari ini = staff, BUKAN transaksi
#   seat_min_dwell_sec    -> min detik DIAM biar dihitung beneran duduk
#   seat_move_eps_px      -> gerak < ini px = dianggap diam
#   zone_exit_grace_sec   -> toleransi ilang sebentar sblm dianggap keluar zona
#   line_anchor           -> titik badan yg dicek nyebrang garis (center/bottom_center)
#   min_box_area_px       -> buang kotak deteksi < ini (0 = terima semua)
# ========================================================
"""


def _save_cfg():
    _sync_active_zones(EDIT["cfg"])     # zona top-level -> kamera aktif (sblm dump)
    with open(EDIT["path"], "w", encoding="utf-8") as f:
        f.write(CONFIG_HEADER)
        yaml.safe_dump(EDIT["cfg"], f, sort_keys=False, allow_unicode=True)


def pipeline(cfg, args):
    """Loop detect+track+zone, dorong annotated JPEG & stats ke STATE."""
    _ensure_cameras(cfg)
    _load_active_zones(cfg)            # zona kamera aktif -> top-level (dipakai ZoneManager + editor)
    cam0 = _active_cam(cfg)
    use_file = cam0.get("type") == "file"
    os.makedirs("output", exist_ok=True)

    def open_src():
        cam = _active_cam(cfg)         # baca tiap dipanggil (reconnect ikut kamera aktif terbaru)
        typ, addr = cam.get("type", "rtsp"), str(cam.get("address", "") or "")
        if typ == "webcam":
            idx = int(addr) if addr.strip().lstrip("-").isdigit() else 0
            c = cv2.VideoCapture(idx, cv2.CAP_DSHOW)   # DSHOW: buka webcam cepat di Windows
            return c, f"webcam:{idx}"
        c = cv2.VideoCapture(addr)
        if typ == "rtsp":
            c.set(cv2.CAP_PROP_BUFFERSIZE, 1)          # low latency RTSP
        return c, addr

    model = YOLO(cfg["model"]["weights"])
    tracker = sv.ByteTrack(
        track_activation_threshold=cfg["tracker"]["track_activation_threshold"],
        lost_track_buffer=cfg["tracker"]["lost_track_buffer"],
        minimum_matching_threshold=cfg["tracker"]["minimum_matching_threshold"],
        frame_rate=cfg["tracker"]["frame_rate"],
    )
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.4, text_thickness=1)

    cap, src = open_src()
    if not cap.isOpened():
        print(f"[ERR] Cannot open source: {src}")
        return
    ok, frame = cap.read()
    if not ok:
        print("[ERR] Cannot read first frame.")
        return

    resize_w = cfg["processing"]["resize_width"]

    def prep(f):
        if resize_w and f.shape[1] != resize_w:
            h = int(f.shape[0] * resize_w / f.shape[1])
            return cv2.resize(f, (resize_w, h))
        return f

    frame = prep(frame)
    H, W = frame.shape[:2]
    src_fps = cap.get(cv2.CAP_PROP_FPS) or cfg["tracker"]["frame_rate"]
    EDIT["cfg"], EDIT["path"], EDIT["wh"] = cfg, args.config, (W, H)
    ZM["zm"] = ZoneManager(cfg, (W, H))
    grabber = LatestFrame(cap, opener=lambda: open_src()[0]) if not use_file else None

    skip = cfg["processing"]["frame_skip"]
    conf, iou, imgsz, device, classes = (cfg["model"]["conf"], cfg["model"]["iou"],
                                         cfg["model"]["imgsz"], cfg["model"]["device"],
                                         cfg["model"]["classes"])
    min_area = float(cfg["logic"].get("min_box_area_px", 0) or 0)
    furn_classes = list(cfg["model"].get("furniture_classes", []) or [])   # kursi/meja (cuma digambar)
    call_classes = list(classes) + furn_classes                            # deteksi person + furnitur sekaligus
    auto_seat_on = bool(cfg["logic"].get("auto_seat_from_chair", False))   # chair -> zona auto, jangan digambar dobel
    queue_alert = int(cfg["logic"].get("queue_alert_count", 3))      # >= org di kasir = alert antri
    snap_on_ev = bool(cfg["logic"].get("snapshot_on_event", True))   # foto pas transaksi/masuk
    snap_events = set(cfg["logic"].get("snapshot_events", ["transaction", "enter"]))
    open_hours = cfg["logic"].get("open_hours", "") or ""            # jam operasi (kosong=24jam)
    max_cap = int(cfg["logic"].get("max_capacity", 0) or 0)          # alert kapasitas penuh
    furn_conf = float(cfg["model"].get("furniture_conf", conf))      # conf khusus furnitur
    disp = cfg.get("display", {}) or {}
    show_box = bool(disp.get("show_boxes", True))
    show_lbl = bool(disp.get("show_labels", True))
    show_zone = bool(disp.get("show_zones", True))
    jpeg_q = int(disp.get("jpeg_quality", 80))
    hm_refresh = float(disp.get("heatmap_refresh_sec", 5))
    os.makedirs("output/snapshots", exist_ok=True)

    with LOCK:
        STATE["wh"] = (W, H)
        STATE["source"] = src
        STATE["alive"] = True

    frame_idx = 0
    last_base = frame.copy()
    fps_t = time.time()
    fps_n = 0
    hm_save_t = time.time()

    print(f"[INFO] pipeline {W}x{H} src_fps~{src_fps:.1f} source={src}")

    while not STOP.is_set():
        if grabber is not None:
            ok, raw = grabber.read()
            if not ok:
                with LOCK:
                    STATE["stale"] = True
                time.sleep(0.01)
                continue
        else:
            ok, raw = cap.read()
            if not ok:
                break

        frame_idx += 1
        if frame_idx % skip != 0:
            continue

        if RESTART.is_set():           # /api/restart -> keluar loop, supervisor re-load model/source/tracker
            break
        # re-baca knob "panas" tiap frame -> bisa diubah live dari /api/settings tanpa restart
        with ZLOCK:
            lg, disp = cfg["logic"], (cfg.get("display") or {})
            min_area = float(lg.get("min_box_area_px", 0) or 0)
            queue_alert = int(lg.get("queue_alert_count", 3))
            snap_on_ev = bool(lg.get("snapshot_on_event", True))
            snap_events = set(lg.get("snapshot_events", ["transaction", "enter"]))
            open_hours = lg.get("open_hours", "") or ""
            max_cap = int(lg.get("max_capacity", 0) or 0)
            furn_conf = float(cfg["model"].get("furniture_conf", conf))
            show_box = bool(disp.get("show_boxes", True))
            show_lbl = bool(disp.get("show_labels", True))
            show_zone = bool(disp.get("show_zones", True))
            jpeg_q = int(disp.get("jpeg_quality", 80))
            hm_refresh = float(disp.get("heatmap_refresh_sec", 5))

        frame = prep(raw)
        last_base = frame.copy()

        if use_file:
            t_now = frame_idx / max(src_fps, 1e-6)
            iso_ts = f"frame_{frame_idx}"
        else:
            t_now = time.time()
            iso_ts = datetime.now().isoformat(timespec="seconds")
        open_now = is_open(open_hours)       # di luar jam buka -> gak dicatat

        zm = ZM["zm"]                       # ref terbaru (bisa di-swap saat edit zona)
        res = model(frame, conf=conf, iou=iou, imgsz=imgsz,
                    device=device, classes=call_classes, verbose=False)[0]
        det_all = sv.Detections.from_ultralytics(res)
        # PISAH: person -> counting/zona ; furnitur -> cuma digambar (gak ngotorin logika orang)
        cid = det_all.class_id if det_all.class_id is not None else np.array([])
        furn = det_all[np.isin(cid, furn_classes)] if len(furn_classes) and len(det_all) else det_all[np.zeros(len(det_all), bool)]
        if len(furn) and furn_conf > conf:   # filter furnitur pakai conf sendiri
            furn = furn[furn.confidence >= furn_conf]
        det = det_all[cid == 0] if len(det_all) else det_all
        if min_area > 0 and len(det):
            wh = det.xyxy[:, 2:4] - det.xyxy[:, 0:2]
            det = det[(wh[:, 0] * wh[:, 1]) >= min_area]
        det = zm.filter_roi(det)
        det = tracker.update_with_detections(det)

        # chair (56) -> auto-seat slot (kalau auto_seat_from_chair on)
        if len(furn):
            fcid = furn.class_id
            zm.sync_auto_seats(furn.xyxy[fcid == 56], t_now)
        else:
            zm.sync_auto_seats([], t_now)

        events = zm.update(det, t_now, iso_ts)
        if open_now:
            for e in events:
                STORE.log(e)                   # persist ke SQLite (cuma pas jam buka)
        else:
            events = []                        # tutup -> abaikan event (gak dihitung/snapshot)

        # annotate
        labels = []
        live = zm.live_dwell(t_now)
        ids = det.tracker_id if det.tracker_id is not None else []
        for tid in ids:
            tid = int(tid)
            if tid in live:
                _, ztype, secs, sitting = live[tid]
                tag = "DUDUK" if sitting else ("KASIR" if ztype == "cashier" else "")
                labels.append(f"#{tid} {int(secs)}s {tag}".strip())
            else:
                labels.append(f"#{tid}")
        annotated = frame.copy()
        if show_box:
            annotated = box_annotator.annotate(annotated, det)
        if show_lbl:
            annotated = label_annotator.annotate(annotated, det, labels)
        if show_zone:
            annotated = zm.render_overlay(annotated)

        # furnitur (kursi/meja): kotak cyan + hitung. TIDAK masuk counting orang.
        # Kalau auto-seat nyala, chair (56) udah jadi zona auto_N -> jangan gambar dobel.
        furn_count = {}
        for i in range(len(furn)):
            cls = int(furn.class_id[i])
            if zm.auto_seat and cls == 56:        # dinamis: ikut toggle live
                continue
            x1, y1, x2, y2 = [int(v) for v in furn.xyxy[i]]
            nm = FURN_NAMES.get(cls, str(cls))
            furn_count[nm] = furn_count.get(nm, 0) + 1
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 255, 0), 1)
            cv2.putText(annotated, nm, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        c = zm.counts
        ftxt = "  " + " ".join(f"{k}:{v}" for k, v in furn_count.items()) if furn_count else ""
        closed_txt = "" if open_now else "  [TUTUP]"
        cv2.putText(annotated, f"IN:{c['entered']} OUT:{c['exited']} now:{len(det)}{ftxt}{closed_txt}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # snapshot pas event terpilih (config snapshot_events)
        if snap_on_ev:
            for e in events:
                if e["event"] in snap_events:
                    fn = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                    cv2.imwrite(f"output/snapshots/{fn}_{e['event']}_{e['zone']}.jpg", annotated)

        # metrik live + alert
        occ = zm.occupancy(t_now)
        inside_live = max(0, c["entered"] - c["exited"])
        alerts = []
        if not open_now:
            alerts.append("TUTUP - di luar jam buka, gak dihitung")
        if max_cap > 0 and inside_live >= max_cap:
            alerts.append(f"Kapasitas penuh: {inside_live}/{max_cap}")
        if occ["cashier_now"] >= queue_alert:
            alerts.append(f"Antri kasir: {occ['cashier_now']} orang")
        if STATE["stale"]:
            alerts.append("Stream CCTV putus")

        ok_enc, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, jpeg_q])

        # fps
        fps_n += 1
        now = time.time()
        if now - fps_t >= 1.0:
            cur_fps = fps_n / (now - fps_t)
            fps_n, fps_t = 0, now
        else:
            cur_fps = STATE["fps"]

        with LOCK:
            if ok_enc:
                STATE["jpeg"] = buf.tobytes()
            STATE["counts"] = {"entered": c["entered"], "exited": c["exited"], "now": len(det)}
            STATE["fps"] = round(cur_fps, 1)
            STATE["stale"] = False
            STATE["occ"] = occ
            STATE["alerts"] = alerts
            for e in events:
                STATE["events"].appendleft(e)

        # heatmap refresh (interval dari config)
        if now - hm_save_t >= hm_refresh:
            try:
                zm.save_heatmap("output/heatmap.png", base_frame=last_base)
            except Exception as ex:
                print("[warn] heatmap:", ex)
            hm_save_t = now

    if grabber is not None:
        grabber.stop()
    cap.release()
    with LOCK:
        STATE["alive"] = False
    print("[INFO] pipeline stopped")


# ---------------- routes ----------------
PAGE = """<!doctype html><html lang=id><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Cafe Detection — Dev Dashboard</title>
<style>
  :root{--bg:#0d1117;--card:#161b22;--bd:#30363d;--tx:#c9d1d9;--mut:#8b949e;
        --in:#3fb950;--out:#f85149;--now:#58a6ff;--acc:#bc8cff}
  *{box-sizing:border-box}
  body{margin:0;font:14px/1.5 system-ui,Segoe UI,sans-serif;background:var(--bg);color:var(--tx)}
  header{padding:12px 20px;border-bottom:1px solid var(--bd);display:flex;
         align-items:center;gap:14px;background:var(--card)}
  header h1{font-size:16px;margin:0;font-weight:600}
  .dot{width:9px;height:9px;border-radius:50%;background:var(--out)}
  .dot.on{background:var(--in);box-shadow:0 0 8px var(--in)}
  .meta{color:var(--mut);font-size:12px;margin-left:auto}
  main{display:grid;grid-template-columns:1fr 360px;gap:16px;padding:16px;max-width:1500px;margin:0 auto}
  .vid{background:#000;border:1px solid var(--bd);border-radius:10px;overflow:hidden;position:relative}
  .vid img{display:block;width:100%}
  .badge{position:absolute;top:10px;right:10px;background:#000a;padding:4px 10px;
         border-radius:20px;font-size:12px;color:var(--mut)}
  aside{display:flex;flex-direction:column;gap:16px}
  .cards{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}
  .card{background:var(--card);border:1px solid var(--bd);border-radius:10px;padding:12px;text-align:center}
  .card .n{font-size:28px;font-weight:700;line-height:1.1}
  .card .l{font-size:11px;color:var(--mut);text-transform:uppercase;letter-spacing:.5px}
  .in .n{color:var(--in)} .out .n{color:var(--out)} .now .n{color:var(--now)}
  .panel{background:var(--card);border:1px solid var(--bd);border-radius:10px;overflow:hidden}
  .panel h2{font-size:12px;margin:0;padding:10px 12px;border-bottom:1px solid var(--bd);
            color:var(--mut);text-transform:uppercase;letter-spacing:.5px}
  table{width:100%;border-collapse:collapse;font-size:12px}
  td,th{padding:6px 10px;text-align:left;border-bottom:1px solid #21262d}
  th{color:var(--mut);font-weight:500}
  tbody tr:last-child td{border-bottom:0}
  .ev{font-weight:600}
  .ev.enter{color:var(--in)} .ev.exit{color:var(--out)}
  .ev.transaction{color:var(--acc)} .ev.seated{color:#d29922}
  .evlist{max-height:260px;overflow:auto}
  .hm img{display:block;width:100%}
  .peak{padding:10px 12px;font-size:13px}
  .peak b{color:var(--acc)}
  .empty{padding:14px;color:var(--mut);font-size:12px}
  .toolbar{display:flex;flex-wrap:wrap;gap:6px;padding:10px;background:var(--card);
           border:1px solid var(--bd);border-bottom:0;border-radius:10px 10px 0 0;align-items:center}
  .toolbar button{background:#21262d;color:var(--tx);border:1px solid var(--bd);border-radius:6px;
           padding:5px 10px;font-size:12px;cursor:pointer}
  .toolbar button:hover{border-color:#8b949e}
  .toolbar button.on{background:var(--now);color:#000;border-color:var(--now);font-weight:600}
  .toolbar .sep{width:1px;height:20px;background:var(--bd);margin:0 4px}
  .toolbar .hint{color:var(--mut);font-size:11px;margin-left:auto}
  .vid{border-radius:0 0 10px 10px}
  .vid canvas{position:absolute;top:0;left:0;width:100%;height:100%;cursor:crosshair}
  .alerts{display:flex;flex-direction:column;gap:6px;margin-bottom:10px}
  .alert{background:#3d1518;border:1px solid #f85149;color:#ffb4ab;padding:8px 12px;
         border-radius:8px;font-size:13px;font-weight:600}
  .met{display:grid;grid-template-columns:1fr 1fr;gap:8px;padding:10px}
  .met .m{background:#0d1117;border:1px solid var(--bd);border-radius:8px;padding:8px 10px}
  .met .m .v{font-size:20px;font-weight:700}
  .met .m .k{font-size:10px;color:var(--mut);text-transform:uppercase;letter-spacing:.4px}
  .met .inside .v{color:var(--now)} .met .conv .v{color:var(--acc)}
  .legend{display:flex;gap:14px;padding:2px 12px 8px;font-size:11px;color:var(--mut)}
  .legend i{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:4px;vertical-align:middle}
  .dlbar{display:flex;gap:8px;padding:0 12px 12px;flex-wrap:wrap}
  .dl{flex:1;text-align:center;background:#21262d;color:var(--tx);border:1px solid var(--bd);
      border-radius:6px;padding:7px 8px;font-size:12px;text-decoration:none;white-space:nowrap}
  .dl:hover{border-color:#8b949e}
  .modal{display:none;position:fixed;inset:0;background:#000b;z-index:50;
         align-items:center;justify-content:center;padding:16px}
  .modal.on{display:flex}
  .sheet{background:var(--card);border:1px solid var(--bd);border-radius:12px;
         width:100%;max-width:560px;max-height:88vh;display:flex;flex-direction:column}
  .shead{padding:12px 16px;border-bottom:1px solid var(--bd);display:flex;align-items:center}
  .shead b{font-size:15px}
  .xbtn{margin-left:auto;background:#21262d;color:var(--tx);border:1px solid var(--bd);
        border-radius:6px;padding:4px 10px;cursor:pointer}
  .sbody{padding:8px 16px;overflow:auto}
  .ssec{font-size:11px;font-weight:700;color:var(--acc);text-transform:uppercase;letter-spacing:.5px;
        margin:12px 0 2px;padding-top:8px;border-top:1px solid var(--bd)}
  .ssec:first-child{border-top:0;margin-top:2px}
  .srow{display:flex;align-items:center;gap:10px;padding:7px 0;border-bottom:1px solid #21262d}
  .srow:last-child{border-bottom:0}
  .srow label{flex:1;font-size:13px}
  .srow .kk{font-size:10px;color:var(--mut);display:block}
  .rbadge{font-size:9px;background:#3d2d00;color:#e3b341;border:1px solid #9e7b00;
          border-radius:4px;padding:1px 5px;margin-left:6px;vertical-align:middle}
  .srow input[type=number],.srow input[type=text],.srow select{width:130px;background:#0d1117;color:var(--tx);
        border:1px solid var(--bd);border-radius:6px;padding:5px 8px;font-size:13px}
  .srow input[type=checkbox]{width:18px;height:18px}
  .rbtn{background:#21262d;color:#e3b341;border:1px solid #9e7b00;border-radius:6px;
        padding:8px 14px;font-weight:600;cursor:pointer}
  #cam_sel{background:#21262d;color:var(--tx);border:1px solid var(--bd);border-radius:6px;
        padding:5px 8px;font-size:12px;max-width:160px}
  .camrow{display:flex;align-items:center;gap:6px;padding:6px 0;border-bottom:1px solid #21262d}
  .camrow input,.camrow select{background:#0d1117;color:var(--tx);border:1px solid var(--bd);
        border-radius:6px;padding:5px 7px;font-size:12px}
  .camrow .cn{width:96px} .camrow .ca{flex:1;min-width:90px}
  .camrow .tag{font-size:9px;color:var(--in);border:1px solid var(--in);border-radius:4px;padding:1px 5px}
  .camrow button{background:#21262d;color:var(--tx);border:1px solid var(--bd);border-radius:6px;
        padding:5px 8px;font-size:12px;cursor:pointer}
  .camrow button.sw{color:#58a6ff;border-color:#1f6feb}
  .camrow button.del{color:#f85149;border-color:#6e2723}
  .camadd{display:flex;gap:6px;flex-wrap:wrap;padding:8px 0 4px;border-bottom:1px solid var(--bd)}
  .camadd input{flex:1;min-width:80px;background:#0d1117;color:var(--tx);border:1px solid var(--bd);
        border-radius:6px;padding:6px 8px;font-size:12px}
  .camadd #ca_addr{flex-basis:100%}
  .camadd select{background:#0d1117;color:var(--tx);border:1px solid var(--bd);border-radius:6px;padding:6px}
  .clshint{font-size:11px;color:var(--mut);padding:2px 0 6px}
  #cls_grid{display:grid;grid-template-columns:repeat(3,1fr);gap:4px 10px}
  .clschip{display:flex;align-items:center;gap:6px;font-size:12px;padding:3px 0}
  .clschip input{width:16px;height:16px}
  .clschip .cid{font-size:9px;color:var(--mut)}
  .sfoot{padding:12px 16px;border-top:1px solid var(--bd);display:flex;align-items:center;gap:10px}
  .smut{font-size:11px;color:var(--mut);flex:1}
  .savebtn{background:var(--in);color:#000;border:0;border-radius:6px;padding:8px 16px;
           font-weight:600;cursor:pointer}
  /* ---- responsive HP ---- */
  @media(max-width:860px){
    main{grid-template-columns:1fr;padding:10px;gap:10px}
    header{padding:10px 14px;gap:10px}
    header h1{font-size:14px}
    .meta{font-size:10px}
    .toolbar{gap:5px;padding:8px}
    .toolbar button{padding:7px 9px;font-size:12px}
    .toolbar .hint{flex-basis:100%;margin-left:0;order:9}
    .card .n{font-size:24px}
    .evlist{max-height:200px}
  }
  @media(max-width:420px){
    .cards{grid-template-columns:1fr 1fr}
    .met{grid-template-columns:1fr 1fr}
    .toolbar button{font-size:11px;padding:6px 8px}
  }
</style></head><body>
<header>
  <span class=dot id=dot></span>
  <h1>Cafe Detection · Dev</h1>
  <span class=meta id=meta>connecting…</span>
</header>
<main>
  <div>
    <div class=alerts id=alerts></div>
    <div class=toolbar>
      <button data-mode=view class=on id=m_view>👁 Lihat</button>
      <span class=sep></span>
      <button data-mode=line id=m_line>↔ Garis masuk</button>
      <button data-mode=cashier id=m_cashier>🟠 Area Kasir</button>
      <button data-mode=seat id=m_seat>🟢 Area Kursi</button>
      <button data-mode=staff id=m_staff>🟣 Zona Kerja</button>
      <button data-mode=roi id=m_roi>🔵 Area Lantai</button>
      <span class=sep></span>
      <button id=b_close>✔ Tutup kotak</button>
      <button id=b_undopt>⤺ Undo titik</button>
      <button id=b_flip>⇄ Flip garis</button>
      <button id=b_undo>🗑 Hapus shape</button>
      <button id=b_clearroi>🧹 Hapus Lantai</button>
      <span class=sep></span>
      <button id=b_auto>🪑 Auto-kursi: ?</button>
      <span class=sep></span>
      <span style="font-size:12px;color:var(--mut)">📷</span>
      <select id=cam_sel title="kamera aktif"></select>
      <button id=b_settings>⚙ Setelan</button>
      <span class=hint id=ehint>mode Lihat</span>
    </div>
    <div class=vid style="position:relative">
      <img id=feed src="/video" alt="live feed">
      <canvas id=cv></canvas>
      <span class=badge id=fps>– fps</span>
    </div>
  </div>
  <aside>
    <div class=cards>
      <div class="card in"><div class=n id=c_in>0</div><div class=l>masuk</div></div>
      <div class="card out"><div class=n id=c_out>0</div><div class=l>keluar</div></div>
      <div class="card now"><div class=n id=c_now>0</div><div class=l>sekarang</div></div>
    </div>
    <div class=panel>
      <h2>Hari ini</h2>
      <div class=met>
        <div class="m inside"><div class=v id=m_inside>0</div><div class=k>di dalam</div></div>
        <div class=m><div class=v id=m_seats>0/0</div><div class=k>meja keisi</div></div>
        <div class=m><div class=v id=m_trans>0</div><div class=k>transaksi</div></div>
        <div class=m><div class=v id=m_avg>0s</div><div class=k>rata2 duduk</div></div>
        <div class="m conv"><div class=v id=m_conv>0%</div><div class=k>konversi beli</div></div>
        <div class=m><div class=v id=m_cash>0</div><div class=k>di kasir kini</div></div>
      </div>
    </div>
    <div class=panel>
      <h2>Peak hour</h2>
      <div class=peak id=peak>–</div>
    </div>
    <div class=panel>
      <h2>Tren per jam (hari ini)</h2>
      <canvas id=chart width=336 height=130 style="width:100%;display:block;padding:8px"></canvas>
      <div class=legend>
        <span><i style="background:#58a6ff"></i>masuk</span>
        <span><i style="background:#bc8cff"></i>transaksi</span>
        <span><i style="background:#d29922"></i>duduk</span>
      </div>
      <div class=dlbar>
        <a class=dl href="/report.csv?type=events" download>⬇ CSV hari ini</a>
        <a class=dl href="/report.csv?type=summary&days=30" download>⬇ Rekap 30 hari</a>
      </div>
    </div>
    <div class=panel>
      <h2>Event terakhir</h2>
      <div class=evlist>
        <table><thead><tr><th>waktu</th><th>event</th><th>zona</th><th>id</th><th>dwell</th></tr></thead>
        <tbody id=events><tr><td colspan=5 class=empty>belum ada event…</td></tr></tbody></table>
      </div>
    </div>
    <div class="panel hm">
      <h2>Heatmap (refresh 5s)</h2>
      <img id=hm src="/heatmap.png" alt="heatmap">
    </div>
  </aside>
</main>
<div class=modal id=settings_modal>
  <div class=sheet>
    <div class=shead><b>⚙ Setelan (live)</b>
      <button id=s_close class=xbtn>✕</button></div>
    <div class=sbody>
      <div class=ssec>Kamera</div>
      <div id=cam_mgr>memuat…</div>
      <div class=camadd>
        <input id=ca_name placeholder="nama (cth: Pintu Depan)">
        <select id=ca_type>
          <option value=rtsp>RTSP</option>
          <option value=webcam>Webcam</option>
          <option value=file>File</option>
        </select>
        <input id=ca_addr placeholder="rtsp://... / index webcam / path file">
        <button id=ca_add class=savebtn>+ Tambah</button>
      </div>
      <div class=ssec>Objek terdeteksi</div>
      <div class=clshint>Centang objek yg mau dideteksi. <b>orang</b> = penghitungan; lainnya = digambar saja. Perlu restart.</div>
      <div id=cls_grid>memuat…</div>
      <div style="text-align:right;padding:8px 0">
        <button id=cls_save class=rbtn>♻ Simpan objek & Restart</button>
      </div>
      <div id=s_fields>memuat…</div>
    </div>
    <div class=sfoot>
      <span class=smut>Disimpan ke config.yaml. Tag <b>restart</b> = perlu reload pipeline.</span>
      <button id=s_restart class=rbtn>♻ Simpan & Restart</button>
      <button id=s_save class=savebtn>💾 Simpan</button>
    </div>
  </div>
</div>
<script>
const $=s=>document.querySelector(s);
async function tick(){
  try{
    const r=await fetch('/api/stats');const d=await r.json();
    $('#dot').classList.toggle('on',d.alive&&!d.stale);
    $('#meta').textContent=`${d.source} · ${d.wh[0]}×${d.wh[1]}`+(d.stale?' · STREAM PUTUS':'');
    $('#fps').textContent=d.fps+' fps';
    $('#c_in').textContent=d.counts.entered;
    $('#c_out').textContent=d.counts.exited;
    $('#c_now').textContent=d.counts.now;
    $('#peak').innerHTML=d.peak?`<b>${d.peak.hour}</b> — ${d.peak.entries} masuk`:'–';
    // metrik hari ini
    const t=d.today||{},o=d.occ||{};
    $('#m_inside').textContent=o.inside??0;
    $('#m_seats').textContent=`${o.seats_busy??0}/${o.seats_total??0}`+(o.seats_total?` (${o.seats_pct}%)`:'');
    $('#m_trans').textContent=t.transactions??0;
    $('#m_avg').textContent=(t.avg_seat_sec?Math.round(t.avg_seat_sec):0)+'s';
    $('#m_conv').textContent=Math.round((t.conversion||0)*100)+'%';
    $('#m_cash').textContent=o.cashier_now??0;
    // alert banner
    const al=d.alerts||[];
    $('#alerts').innerHTML=al.map(a=>`<div class=alert>⚠ ${a}</div>`).join('');
    // toggle auto-kursi status
    const ab=$('#b_auto');
    ab.textContent='🪑 Auto-kursi: '+(d.auto_seat?'ON':'OFF');
    ab.classList.toggle('on',!!d.auto_seat);
    const tb=$('#events');
    if(!d.events.length){tb.innerHTML='<tr><td colspan=5 class=empty>belum ada event…</td></tr>';}
    else{tb.innerHTML=d.events.map(e=>`<tr><td>${e.ts}</td>`+
      `<td class="ev ${e.event}">${e.event}</td><td>${e.zone}</td>`+
      `<td>${e.track_id}</td><td>${e.dwell_sec||''}</td></tr>`).join('');}
  }catch(e){$('#dot').classList.remove('on');$('#meta').textContent='disconnected';}
}
setInterval(tick,1000);tick();
// heatmap cache-bust tiap 5s
setInterval(()=>{$('#hm').src='/heatmap.png?t='+Date.now();},5000);

// ---- grafik tren per jam ----
async function drawChart(){
  let data;
  try{data=await(await fetch('/api/hourly')).json();}catch(e){return;}
  const cvs=$('#chart'),x=cvs.getContext('2d'),W=cvs.width,H=cvs.height;
  x.clearRect(0,0,W,H);
  const padL=22,padB=14,padT=6;
  const series=[['entered','#58a6ff'],['transactions','#bc8cff'],['seated','#d29922']];
  const max=Math.max(1,...data.flatMap(d=>series.map(s=>d[s[0]])));
  const gw=(W-padL)/24, bw=Math.max(1,(gw-2)/3);
  x.strokeStyle='#30363d';x.fillStyle='#8b949e';x.font='9px system-ui';x.lineWidth=1;
  // sumbu Y (max + tengah)
  [max,Math.round(max/2),0].forEach(v=>{
    const yy=padT+(H-padT-padB)*(1-v/max);
    x.beginPath();x.moveTo(padL,yy);x.lineTo(W,yy);x.globalAlpha=.25;x.stroke();x.globalAlpha=1;
    x.fillText(v,0,yy+3);
  });
  data.forEach((d,h)=>{
    const x0=padL+h*gw;
    series.forEach((s,si)=>{
      const v=d[s[0]],bh=(H-padT-padB)*(v/max);
      x.fillStyle=s[1];
      x.fillRect(x0+1+si*bw,H-padB-bh,bw,bh);
    });
    if(h%3===0){x.fillStyle='#8b949e';x.fillText(h,x0+1,H-3);}
  });
}
drawChart();setInterval(drawChart,10000);

// ---------------- editor zona ----------------
const COL={line:'#ff00ff',cashier:'#ffa500',seat:'#00ff00',staff:'#c800c8',roi:'#1e90ff'};
const cv=$('#cv'),ctx=cv.getContext('2d');
let mode='view',pts=[],W=0,H=0,cfg={lines:[],zones:[],roi:[]},mouse=null;

async function loadCfg(){
  const d=await(await fetch('/api/config')).json();
  cfg={lines:d.lines,zones:d.zones,roi:d.roi};
  if(d.wh[0]){W=d.wh[0];H=d.wh[1];cv.width=W;cv.height=H;}
  draw();
}
function setMode(m){
  mode=m;pts=[];
  document.querySelectorAll('.toolbar button[data-mode]').forEach(b=>
    b.classList.toggle('on',b.dataset.mode===m));
  const h={view:'mode Lihat — klik tombol buat gambar zona',
    line:'GARIS MASUK: klik 2 titik di jalur orang lewat. Hitung orang MASUK/KELUAR. (auto-save)',
    cashier:'AREA KASIR: klik sudut area depan kasir, "Tutup kotak" (≥3). Hitung TRANSAKSI.',
    seat:'AREA KURSI: klik sudut kursi/meja, "Tutup kotak" (≥3). Hitung DURASI DUDUK.',
    staff:'ZONA KERJA: klik sudut area staff (belakang kasir/dapur). Orang di sini = STAFF, TIDAK dihitung.',
    roi:'AREA LANTAI: klik area lantai tempat orang jalan. Orang di luar ini (poster dinding) DIABAIKAN.'};
  $('#ehint').textContent=h[m];draw();
}
function evtXY(e){
  const r=cv.getBoundingClientRect();
  return [Math.round((e.clientX-r.left)/r.width*W),
          Math.round((e.clientY-r.top)/r.height*H)];
}
function arrow(s,en){
  const mx=(s[0]+en[0])/2,my=(s[1]+en[1])/2,vx=en[0]-s[0],vy=en[1]-s[1];
  const n=Math.hypot(vx,vy)||1,ax=vy/n*45,ay=-vx/n*45;
  ctx.beginPath();ctx.moveTo(mx,my);ctx.lineTo(mx+ax,my+ay);ctx.stroke();
  ctx.fillStyle='#ffff00';ctx.fillText('IN',mx+ax+4,my+ay);
}
function poly(p,col,close){
  if(!p.length)return;
  ctx.strokeStyle=col;ctx.lineWidth=2;ctx.beginPath();
  ctx.moveTo(p[0][0],p[0][1]);for(let i=1;i<p.length;i++)ctx.lineTo(p[i][0],p[i][1]);
  if(close)ctx.closePath();ctx.stroke();
  ctx.fillStyle=col;p.forEach(q=>{ctx.beginPath();ctx.arc(q[0],q[1],4,0,7);ctx.fill();});
}
function draw(){
  if(!W)return;
  ctx.clearRect(0,0,W,H);ctx.font='14px sans-serif';ctx.lineWidth=2;
  // existing
  for(const ln of cfg.lines){let s=ln.start,en=ln.end;if(ln.flip){[s,en]=[en,s];}
    ctx.strokeStyle=COL.line;ctx.beginPath();ctx.moveTo(s[0],s[1]);ctx.lineTo(en[0],en[1]);ctx.stroke();
    ctx.fillStyle=COL.line;ctx.fillText(ln.name,s[0],s[1]-6);arrow(s,en);}
  for(const z of cfg.zones){const c=COL[z.type]||COL.seat;
    poly(z.polygon,c,true);ctx.fillStyle=c;ctx.fillText(z.name,z.polygon[0][0],z.polygon[0][1]-6);}
  if(cfg.roi&&cfg.roi.length>=3){poly(cfg.roi,COL.roi,true);
    ctx.fillStyle=COL.roi;ctx.fillText('ROI',cfg.roi[0][0],cfg.roi[0][1]-6);}
  // in-progress
  if(pts.length){const c=COL[mode]||'#fff';
    poly(pts,c,false);
    if(mouse&&pts.length){ctx.strokeStyle=c;ctx.setLineDash([5,4]);ctx.beginPath();
      ctx.moveTo(pts[pts.length-1][0],pts[pts.length-1][1]);ctx.lineTo(mouse[0],mouse[1]);
      ctx.stroke();ctx.setLineDash([]);}}
}
async function post(url,body){
  const r=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify(body||{})});
  const d=await r.json().catch(()=>({}));
  if(!d.ok&&d.err)alert(d.err);
  await loadCfg();return d;
}
async function commitPoly(){
  if(mode==='view'||mode==='line')return;
  if(pts.length<3){alert('butuh ≥3 titik');return;}
  await post('/api/shape',{type:mode,points:pts});pts=[];
}
cv.addEventListener('click',async e=>{
  if(mode==='view')return;
  pts.push(evtXY(e));
  if(mode==='line'&&pts.length===2){await post('/api/shape',{type:'line',points:pts});pts=[];}
  draw();
});
cv.addEventListener('contextmenu',e=>{e.preventDefault();commitPoly();});
cv.addEventListener('mousemove',e=>{if(mode!=='view'){mouse=evtXY(e);draw();}});
document.querySelectorAll('.toolbar button[data-mode]').forEach(b=>
  b.onclick=()=>setMode(b.dataset.mode));
$('#b_close').onclick=commitPoly;
$('#b_undopt').onclick=()=>{pts.pop();draw();};
$('#b_undo').onclick=()=>post('/api/undo');
$('#b_clearroi').onclick=()=>post('/api/clear_roi');
$('#b_flip').onclick=()=>post('/api/flip');
$('#b_auto').onclick=()=>post('/api/toggle_auto');

// ----- panel setelan -----
const SLABEL={
 rtsp:'Sumber RTSP (URL CCTV)', file:'File video (mode file)',
 weights:'Model YOLO (.pt)', conf:'Confidence orang', iou:'IoU NMS',
 imgsz:'Ukuran inferensi (px)', device:'Device (0=GPU, cpu)',
 furniture_conf:'Conf furnitur (kursi/meja)',
 track_activation_threshold:'Tracker: ambang aktif', lost_track_buffer:'Tracker: buffer ilang (frame)',
 minimum_matching_threshold:'Tracker: ambang cocok', frame_rate:'Tracker: frame rate',
 frame_skip:'Proses tiap N frame', resize_width:'Resize lebar (px)',
 cashier_min_dwell_sec:'Kasir: min detik = transaksi',
 cashier_max_dwell_sec:'Kasir: max detik (lebih=staff)',
 seat_min_dwell_sec:'Kursi: min detik diam = duduk',
 seat_move_eps_px:'Kursi: gerak < px = diam',
 zone_exit_grace_sec:'Toleransi ilang sblm keluar (dtk)',
 seat_memory_sec:'Memori kursi: ingat brp detik',
 seat_memory_px:'Memori kursi: radius cocok (px)',
 min_box_area_px:'Buang kotak < px² (0=semua)',
 line_anchor:'Titik cek nyebrang garis',
 line_crossing_frames:'Frame nyebrang garis',
 auto_seat_from_chair:'Auto-kursi dari chair',
 auto_seat_ttl_sec:'Auto-kursi: slot ilang (dtk)',
 auto_seat_grow_up:'Auto-kursi: perbesar ke atas ×',
 auto_seat_ema:'Auto-kursi: smoothing (0-1)',
 auto_seat_iou:'Auto-kursi: ambang cocok IoU',
 queue_alert_count:'Alert antri kasir >= org',
 max_capacity:'Alert kapasitas penuh (0=off)',
 open_hours:'Jam buka (kosong=24j, 08:00-22:00)',
 snapshot_on_event:'Foto pas event',
 snapshot_events:'Event di-foto (pisah koma)',
 snapshot_retain_days:'Hapus snapshot > hari (0=off)',
 db_retain_days:'Hapus event DB > hari (0=off)',
 heatmap_decay:'Heatmap decay (0=off)',
 heatmap_gamma:'Heatmap gamma (<1 angkat sepi)',
 heatmap_blur:'Heatmap blur',
 show_boxes:'Tampil kotak orang',
 show_labels:'Tampil label id/detik',
 show_zones:'Tampil garis/zona',
 jpeg_quality:'Kualitas JPEG (1-100)',
 stream_fps:'FPS stream ke browser',
 heatmap_refresh_sec:'Refresh heatmap (dtk)',
};
const SECNAME={source:'Sumber',model:'Model',tracker:'Tracker',processing:'Proses',logic:'Logika',display:'Tampilan'};
let SFIELDS=[];
function fldInput(f,i){
  if(f.type==='bool')return `<input type=checkbox data-i=${i} ${f.val?'checked':''}>`;
  if(f.type==='list')return `<input type=text data-i=${i} value="${f.val??''}">`;
  if(f.type&&f.type.startsWith('enum:')){
    const opts=f.type.slice(5).split(',');
    return `<select data-i=${i}>`+opts.map(o=>`<option ${o==f.val?'selected':''}>${o}</option>`).join('')+`</select>`;
  }
  if(f.type==='str')return `<input type=text data-i=${i} value="${f.val??''}">`;
  return `<input type=number step=any data-i=${i} value="${f.val??0}">`;
}
async function openSettings(){
  const d=await(await fetch('/api/settings')).json();
  SFIELDS=d.fields||[];
  let html='',cur='';
  SFIELDS.forEach((f,i)=>{
    if(f.sec!==cur){cur=f.sec;html+=`<div class=ssec>${SECNAME[cur]||cur}</div>`;}
    const lab=SLABEL[f.key]||f.key;
    const badge=f.restart?'<span class=rbadge>restart</span>':'';
    html+=`<div class=srow><label>${lab}${badge}<span class=kk>${f.key}</span></label>${fldInput(f,i)}</div>`;
  });
  $('#s_fields').innerHTML=html;
  await loadCameras();
  await loadClasses();
  $('#settings_modal').classList.add('on');
}
function collectSettings(){
  const body={};
  $('#s_fields').querySelectorAll('[data-i]').forEach(el=>{
    const f=SFIELDS[+el.dataset.i];
    body[f.key]=f.type==='bool'?el.checked:el.value;
  });
  return body;
}
$('#b_settings').onclick=openSettings;
$('#s_close').onclick=()=>$('#settings_modal').classList.remove('on');
$('#settings_modal').addEventListener('click',e=>{
  if(e.target.id==='settings_modal')$('#settings_modal').classList.remove('on');});
$('#s_save').onclick=async()=>{
  const r=await post('/api/settings',collectSettings());
  if(r&&r.ok){
    if(r.need_restart){
      if(confirm('Ada setelan model/sumber/tracker berubah. Restart pipeline sekarang?')){
        await post('/api/restart');
      }
    }
    $('#settings_modal').classList.remove('on');
    setTimeout(()=>{$('#feed').src='/video?'+Date.now();},800);
  }
};
$('#s_restart').onclick=async()=>{
  await post('/api/settings',collectSettings());   // simpan dulu
  await post('/api/restart');
  $('#settings_modal').classList.remove('on');
  setTimeout(()=>{$('#feed').src='/video?'+Date.now();},1200);
};

// ----- kamera (switchable) -----
const esc=s=>String(s==null?'':s).replace(/"/g,'&quot;');
async function loadCameras(){
  let d; try{d=await(await fetch('/api/cameras')).json();}catch(e){return;}
  const cams=d.cameras||[],act=d.active;
  $('#cam_sel').innerHTML=cams.map(c=>`<option value="${c.id}" ${c.id==act?'selected':''}>${esc(c.name)}</option>`).join('');
  const mg=$('#cam_mgr'); if(!mg)return;
  mg.innerHTML=cams.map(c=>{
    const a=c.id==act;
    return `<div class=camrow data-id="${c.id}">`+
      `<input class=cn value="${esc(c.name)}" title=nama>`+
      `<select class=ct>${['rtsp','webcam','file'].map(t=>`<option ${t==c.type?'selected':''}>${t}</option>`).join('')}</select>`+
      `<input class=ca value="${esc(c.address)}" title=alamat>`+
      (a?`<span class=tag>AKTIF</span>`:`<button class=sw>Pakai</button>`)+
      `<button class=upd title=simpan>💾</button>`+
      (a?'':`<button class=del title=hapus>🗑</button>`)+
      `</div>`;
  }).join('')||'<div class=empty>belum ada kamera</div>';
}
function reloadFeed(ms){setTimeout(()=>{$('#feed').src='/video?'+Date.now();},ms||1000);}
$('#cam_sel').onchange=async()=>{
  await post('/api/camera/switch',{id:$('#cam_sel').value});
  await loadCameras();reloadFeed();
};
$('#cam_mgr').addEventListener('click',async e=>{
  const row=e.target.closest('.camrow'); if(!row)return;
  const id=row.dataset.id;
  if(e.target.classList.contains('sw')){
    await post('/api/camera/switch',{id});await loadCameras();reloadFeed();
  }else if(e.target.classList.contains('del')){
    if(confirm('Hapus kamera ini?')){await post('/api/camera/delete',{id});await loadCameras();}
  }else if(e.target.classList.contains('upd')){
    const r=await post('/api/camera/update',{id,
      name:row.querySelector('.cn').value,type:row.querySelector('.ct').value,
      address:row.querySelector('.ca').value});
    await loadCameras(); if(r&&r.restart)reloadFeed();
  }
});
$('#ca_add').onclick=async()=>{
  const name=$('#ca_name').value,type=$('#ca_type').value,address=$('#ca_addr').value;
  if(!address&&type!=='webcam'){alert('isi alamat dulu');return;}
  await post('/api/camera/add',{name,type,address});
  $('#ca_name').value='';$('#ca_addr').value='';
  await loadCameras();
};

// ----- objek/class on-off -----
async function loadClasses(){
  let d; try{d=await(await fetch('/api/classes')).json();}catch(e){return;}
  const on=new Set(d.furniture||[]);
  $('#cls_grid').innerHTML=(d.catalog||[]).map(c=>{
    const checked=(c.id===0)?d.person:on.has(c.id);
    return `<label class=clschip><input type=checkbox data-id=${c.id} ${checked?'checked':''}>`+
           `${c.name} <span class=cid>#${c.id}</span></label>`;
  }).join('');
}
$('#cls_save').onclick=async()=>{
  let person=true; const furniture=[];
  $('#cls_grid').querySelectorAll('input[data-id]').forEach(el=>{
    const id=+el.dataset.id;
    if(id===0)person=el.checked;
    else if(el.checked)furniture.push(id);
  });
  await post('/api/classes',{person,furniture});
  reloadFeed(1200);
};
window.addEventListener('keydown',e=>{
  if(e.key==='Enter')commitPoly();
  else if(e.key==='z'){pts.pop();draw();}
  else if(e.key==='Escape'){pts=[];setMode('view');}
});
loadCfg();setInterval(loadCfg,4000);   // sync shape kalau diedit dari tempat lain
loadCameras();   // isi dropdown kamera toolbar
</script></body></html>"""


@app.route("/")
def index():
    return render_template_string(PAGE)


@app.route("/video")
def video():
    fps = int((EDIT["cfg"] or {}).get("display", {}).get("stream_fps", 25)) if EDIT["cfg"] else 25
    delay = 1.0 / max(1, fps)
    def gen():
        boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
        while True:
            with LOCK:
                jpg = STATE["jpeg"]
            if jpg is None:
                time.sleep(0.05)
                continue
            yield boundary + jpg + b"\r\n"
            time.sleep(delay)   # cap fps kirim ke browser (config stream_fps)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/stats")
def stats():
    today = STORE.today()                  # metrik persist (dari SQLite)
    auto = bool(EDIT["cfg"]["logic"].get("auto_seat_from_chair", False)) if EDIT["cfg"] else False
    with LOCK:                              # semua baca STATE di dalam lock (anti race)
        occ = dict(STATE["occ"])
        payload = {
            "counts": dict(STATE["counts"]),
            "events": list(STATE["events"])[:60],
            "peak": today["peak"],
            "today": today,
            "alerts": list(STATE["alerts"]),
            "fps": STATE["fps"],
            "wh": STATE["wh"],
            "source": STATE["source"],
            "alive": STATE["alive"],
            "stale": STATE["stale"],
            "auto_seat": auto,
        }
    occ["inside"] = max(0, today["entered"] - today["exited"])   # okupansi = masuk - keluar
    payload["occ"] = occ
    return jsonify(payload)


@app.route("/api/history")
def history():
    return jsonify(STORE.history(int(request.args.get("days", 7))))


@app.route("/api/hourly")
def hourly():
    return jsonify(STORE.hourly(request.args.get("day") or None))


@app.route("/report.csv")
def report_csv():
    """Unduh laporan CSV. ?type=events (mentah, default) | summary (rekap harian)."""
    typ = request.args.get("type", "events")
    day = request.args.get("day") or datetime.now().strftime("%Y-%m-%d")
    buf = io.StringIO()
    w = csv.writer(buf)
    if typ == "summary":
        w.writerow(["hari", "masuk", "transaksi", "duduk", "rata2_duduk_dtk"])
        for r in STORE.history(int(request.args.get("days", 30))):
            w.writerow([r["day"], r["entered"], r["transactions"], r["seated"], r["avg_seat_sec"]])
        fname = "laporan_rekap.csv"
    else:
        w.writerow(["waktu", "event", "zona", "track_id", "dwell_detik"])
        for ts, ev, zn, tid, dw in STORE.events_for(day):
            w.writerow([ts, ev, zn, tid, "" if dw is None else round(dw, 1)])
        fname = f"laporan_{day}.csv"
    return Response(buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": f"attachment; filename={fname}"})


@app.route("/heatmap.png")
def heatmap():
    path = "output/heatmap.png"
    if os.path.exists(path):
        with open(path, "rb") as f:
            return Response(f.read(), mimetype="image/png")
    return Response(status=404)


# -------- editor zona (gambar di browser, koordinat = pixel frame W×H) --------
@app.route("/api/config")
def get_config():
    with ZLOCK:                         # snapshot di dalam lock (anti race dgn edit)
        cfg = EDIT["cfg"] or {}
        return jsonify({
            "lines": list(cfg.get("lines") or []),
            "zones": list(cfg.get("zones") or []),
            "roi": list(cfg.get("detect_roi") or []),
            "wh": list(EDIT["wh"]),
        })


@app.route("/api/shape", methods=["POST"])
def add_shape():
    if EDIT["cfg"] is None:
        return jsonify({"ok": False, "err": "pipeline belum siap"}), 409
    d = request.get_json(force=True)
    typ = d.get("type")
    pts = [[int(p[0]), int(p[1])] for p in d.get("points", [])]
    cfg = EDIT["cfg"]
    with ZLOCK:
        if typ == "line":
            if len(pts) < 2:
                return jsonify({"ok": False, "err": "garis butuh 2 titik"}), 400
            cfg.setdefault("lines", [])
            n = len(cfg["lines"]) + len(cfg.get("zones") or []) + 1
            cfg["lines"].append({"name": f"garis_{n}", "start": pts[0], "end": pts[1]})
        elif typ in ("cashier", "seat", "staff"):
            if len(pts) < 3:
                return jsonify({"ok": False, "err": "kotak butuh >=3 titik"}), 400
            cfg.setdefault("zones", [])
            n = len(cfg.get("lines") or []) + len(cfg["zones"]) + 1
            prefix = {"cashier": "kasir", "seat": "meja", "staff": "kerja"}[typ]
            nm = f"{prefix}_{n}"
            cfg["zones"].append({"name": nm, "type": typ, "polygon": pts})
        elif typ == "roi":
            if len(pts) < 3:
                return jsonify({"ok": False, "err": "ROI butuh >=3 titik"}), 400
            cfg["detect_roi"] = pts
        else:
            return jsonify({"ok": False, "err": f"tipe tak dikenal: {typ}"}), 400
        _save_cfg()
        _rebuild_zm()
    return jsonify({"ok": True})


@app.route("/api/toggle_auto", methods=["POST"])
def toggle_auto():
    """Nyala/mati auto-kursi (chair -> zona) live, tanpa restart."""
    if EDIT["cfg"] is None:
        return jsonify({"ok": False}), 409
    with ZLOCK:
        cur = bool(EDIT["cfg"]["logic"].get("auto_seat_from_chair", False))
        EDIT["cfg"]["logic"]["auto_seat_from_chair"] = not cur
        _save_cfg()
        _rebuild_zm()
    return jsonify({"ok": True, "auto": not cur})


@app.route("/api/undo", methods=["POST"])
def undo_shape():
    """Hapus shape terakhir: prioritas zona, lalu garis (mirip 'x' di setup_zones)."""
    if EDIT["cfg"] is None:
        return jsonify({"ok": False}), 409
    cfg = EDIT["cfg"]
    with ZLOCK:
        if cfg.get("zones"):
            cfg["zones"].pop()
        elif cfg.get("lines"):
            cfg["lines"].pop()
        elif cfg.get("detect_roi"):       # fallback: hapus ROI juga
            cfg["detect_roi"] = []
        else:
            return jsonify({"ok": False, "err": "kosong"}), 400
        _save_cfg()
        _rebuild_zm()
    return jsonify({"ok": True})


@app.route("/api/flip", methods=["POST"])
def flip_line():
    """Balik arah IN/OUT garis terakhir."""
    if EDIT["cfg"] is None:
        return jsonify({"ok": False}), 409
    cfg = EDIT["cfg"]
    with ZLOCK:
        if not cfg.get("lines"):
            return jsonify({"ok": False, "err": "belum ada garis"}), 400
        cfg["lines"][-1]["flip"] = not cfg["lines"][-1].get("flip", False)
        _save_cfg()
        _rebuild_zm()
    return jsonify({"ok": True, "flip": cfg["lines"][-1]["flip"]})


@app.route("/api/clear_roi", methods=["POST"])
def clear_roi():
    if EDIT["cfg"] is None:
        return jsonify({"ok": False}), 409
    with ZLOCK:
        EDIT["cfg"]["detect_roi"] = []
        _save_cfg()
        _rebuild_zm()
    return jsonify({"ok": True})


# SEMUA knob yg bisa diatur dari web. (section, key, tipe, restart).
#   restart=False -> langsung jalan (re-baca tiap frame / rebuild ZoneManager).
#   restart=True  -> butuh klik "Restart" (reload model/source/tracker/resolusi).
# Dikecualikan (file-only, bahaya kalau diubah remote): server.host/port.
SETTINGS_SPEC = [
    # ---- deteksi (RESTART) ----
    ("model", "conf", "num", True),                # sensitivitas deteksi orang
    # ---- logika kafe (LIVE) ----
    ("logic", "cashier_min_dwell_sec", "num", False),   # min detik di kasir = transaksi
    ("logic", "seat_min_dwell_sec", "num", False),      # min detik diam = duduk
    ("logic", "auto_seat_from_chair", "bool", False),   # kursi auto jadi zona duduk
    ("logic", "queue_alert_count", "num", False),       # alert antri kasir
    ("logic", "max_capacity", "num", False),            # alert kapasitas penuh
    ("logic", "open_hours", "str", False),              # jam operasi
    ("logic", "snapshot_on_event", "bool", False),      # foto pas event
    # ---- tampilan (LIVE) ----
    ("display", "show_boxes", "bool", False),
    ("display", "show_labels", "bool", False),
    ("display", "show_zones", "bool", False),
    ("display", "stream_fps", "num", False),
]
# Knob lanjutan (tracker, imgsz, seat_memory, heatmap, retention, dll) sengaja TIDAK
# di web -> diatur manual di config.yaml. Biar panel web bersih & fokus operasional.


@app.route("/api/settings", methods=["GET", "POST"])
def settings():
    if EDIT["cfg"] is None:
        return jsonify({"ok": False, "err": "pipeline belum siap"}), 409
    cfg = EDIT["cfg"]
    if request.method == "GET":
        with ZLOCK:
            out = []
            for sec, key, typ, rst in SETTINGS_SPEC:
                val = (cfg.get(sec) or {}).get(key)
                if typ == "list":
                    val = ", ".join(str(x) for x in (val or []))
                out.append({"sec": sec, "key": key, "type": typ, "val": val, "restart": rst})
        return jsonify({"fields": out})

    # POST: {key: val, ...}  (key unik antar section)
    d = request.get_json(force=True) or {}
    spec = {k: (sec, t, rst) for sec, k, t, rst in SETTINGS_SPEC}
    need_restart = False
    with ZLOCK:
        for k, v in d.items():
            if k not in spec:
                continue
            sec, typ, rst = spec[k]
            try:
                if typ == "num":
                    v = float(v)
                    if v == int(v):
                        v = int(v)
                elif typ == "bool":
                    v = bool(v)
                elif typ == "list":
                    v = [s.strip() for s in str(v).split(",") if s.strip()]
                elif typ.startswith("enum:"):
                    allowed = typ.split(":", 1)[1].split(",")
                    v = str(v)
                    if v not in allowed:
                        continue
                else:
                    v = str(v)
            except (TypeError, ValueError):
                continue
            old = cfg.setdefault(sec, {}).get(k)
            if old != v and rst:
                need_restart = True
            cfg.setdefault(sec, {})[k] = v
        _save_cfg()
        _rebuild_zm()          # knob live (dwell/auto-seat/anchor) masuk ZoneManager langsung
    return jsonify({"ok": True, "need_restart": need_restart})


@app.route("/api/restart", methods=["POST"])
def restart_pipeline():
    """Reload pipeline (model/source/tracker/resolusi) tanpa matiin proses Flask."""
    if EDIT["cfg"] is None:
        return jsonify({"ok": False}), 409
    RESTART.set()
    return jsonify({"ok": True})


@app.route("/api/classes", methods=["GET", "POST"])
def classes_cfg():
    """On/off class deteksi. person (0) -> counting; sisanya -> objek digambar (furniture_classes)."""
    if EDIT["cfg"] is None:
        return jsonify({"ok": False}), 409
    cfg = EDIT["cfg"]
    if request.method == "GET":
        with ZLOCK:
            person = 0 in (cfg["model"].get("classes") or [])
            furn = list(cfg["model"].get("furniture_classes") or [])
            catalog = [{"id": k, "name": v} for k, v in COCO_NAMES.items()]
        return jsonify({"catalog": catalog, "person": person, "furniture": furn})

    d = request.get_json(force=True) or {}
    with ZLOCK:
        cfg["model"]["classes"] = [0] if d.get("person", True) else []
        furn = []
        for x in (d.get("furniture") or []):
            try:
                furn.append(int(x))
            except (TypeError, ValueError):
                pass
        cfg["model"]["furniture_classes"] = sorted(set(furn))
        _save_cfg()
    RESTART.set()     # call_classes dibaca pas pipeline start -> perlu reload
    return jsonify({"ok": True})


# ---------------- kamera (switchable, zona per-kamera) ----------------
@app.route("/api/cameras")
def cameras_list():
    if EDIT["cfg"] is None:
        return jsonify({"cameras": [], "active": None})
    with ZLOCK:
        _ensure_cameras(EDIT["cfg"])
        cams = [{"id": c["id"], "name": c.get("name", c["id"]),
                 "type": c.get("type", "rtsp"), "address": c.get("address", ""),
                 "zones": len(c.get("zones") or []), "lines": len(c.get("lines") or [])}
                for c in EDIT["cfg"]["cameras"]]
        return jsonify({"cameras": cams, "active": EDIT["cfg"]["active_camera"]})


@app.route("/api/camera/add", methods=["POST"])
def camera_add():
    if EDIT["cfg"] is None:
        return jsonify({"ok": False}), 409
    d = request.get_json(force=True) or {}
    typ = d.get("type", "rtsp")
    if typ not in ("rtsp", "file", "webcam"):
        return jsonify({"ok": False, "err": "tipe harus rtsp/file/webcam"}), 400
    with ZLOCK:
        cfg = EDIT["cfg"]
        _ensure_cameras(cfg)
        cid = _new_cam_id(cfg)
        cfg["cameras"].append({
            "id": cid, "name": (d.get("name") or cid).strip(), "type": typ,
            "address": (d.get("address") or "").strip(),
            "lines": [], "zones": [], "detect_roi": [],
        })
        _save_cfg()
    return jsonify({"ok": True, "id": cid})


@app.route("/api/camera/update", methods=["POST"])
def camera_update():
    if EDIT["cfg"] is None:
        return jsonify({"ok": False}), 409
    d = request.get_json(force=True) or {}
    cid = d.get("id")
    with ZLOCK:
        cfg = EDIT["cfg"]
        _ensure_cameras(cfg)
        cam = next((c for c in cfg["cameras"] if c["id"] == cid), None)
        if not cam:
            return jsonify({"ok": False, "err": "kamera tak ada"}), 404
        if "name" in d:
            cam["name"] = (d["name"] or cam["id"]).strip()
        if d.get("type") in ("rtsp", "file", "webcam"):
            cam["type"] = d["type"]
        if "address" in d:
            cam["address"] = (d["address"] or "").strip()
        _save_cfg()
        restart = (cid == cfg["active_camera"])   # baca di dalam lock (anti race switch)
    if restart:                                   # kamera aktif diedit -> reload sumber
        RESTART.set()
    return jsonify({"ok": True, "restart": restart})


@app.route("/api/camera/delete", methods=["POST"])
def camera_delete():
    if EDIT["cfg"] is None:
        return jsonify({"ok": False}), 409
    cid = (request.get_json(force=True) or {}).get("id")
    with ZLOCK:
        cfg = EDIT["cfg"]
        _ensure_cameras(cfg)
        if len(cfg["cameras"]) <= 1:
            return jsonify({"ok": False, "err": "minimal 1 kamera"}), 400
        if cid == cfg["active_camera"]:
            return jsonify({"ok": False, "err": "kamera aktif, switch dulu sblm hapus"}), 400
        cfg["cameras"] = [c for c in cfg["cameras"] if c["id"] != cid]
        _save_cfg()
    return jsonify({"ok": True})


@app.route("/api/camera/switch", methods=["POST"])
def camera_switch():
    if EDIT["cfg"] is None:
        return jsonify({"ok": False}), 409
    cid = (request.get_json(force=True) or {}).get("id")
    with ZLOCK:
        cfg = EDIT["cfg"]
        _ensure_cameras(cfg)
        if cid not in [c["id"] for c in cfg["cameras"]]:
            return jsonify({"ok": False, "err": "kamera tak ada"}), 404
        if cid == cfg["active_camera"]:
            return jsonify({"ok": True, "active": cid})   # udah aktif
        _sync_active_zones(cfg)            # simpen zona kamera lama
        cfg["active_camera"] = cid
        _load_active_zones(cfg)            # load zona kamera baru -> top-level
        _save_cfg()
        _rebuild_zm()
    RESTART.set()                          # reopen source kamera baru
    return jsonify({"ok": True, "active": cid})


def cleanup_snapshots(folder, days):
    """Hapus foto snapshot lebih tua dari `days` hari (jaga disk gak penuh). 0=skip."""
    if days <= 0 or not os.path.isdir(folder):
        return 0
    cutoff = time.time() - days * 86400
    n = 0
    for fn in os.listdir(folder):
        p = os.path.join(folder, fn)
        try:
            if os.path.isfile(p) and os.path.getmtime(p) < cutoff:
                os.remove(p)
                n += 1
        except OSError:
            pass
    return n


def maintenance(cfg):
    """Thread retensi: hapus snapshot & event lama tiap jam (anti disk penuh)."""
    while not STOP.is_set():
        snap_days = int(cfg["logic"].get("snapshot_retain_days", 7))  # re-baca tiap jam -> live
        db_days = int(cfg["logic"].get("db_retain_days", 0))          # 0 = simpen history selamanya
        try:
            ns = cleanup_snapshots("output/snapshots", snap_days)
            nd = STORE.prune(db_days)
            if ns or nd:
                print(f"[maint] hapus {ns} snapshot, {nd} event lama")
        except Exception as ex:
            print("[maint] err:", ex)
        STOP.wait(3600)   # tiap jam


def pipeline_supervisor(cfg, args):
    """Watchdog: kalau pipeline crash/stream mati total, restart otomatis."""
    while not STOP.is_set():
        try:
            pipeline(cfg, args)
        except Exception as ex:
            print("[watchdog] pipeline crash:", ex)
        with LOCK:
            STATE["alive"] = False
        if STOP.is_set():
            break
        if RESTART.is_set():           # restart manual dari web: langsung re-enter, tanpa tunggu
            RESTART.clear()
            print("[restart] reload pipeline (config baru) dari web")
            continue
        print("[watchdog] restart pipeline 3s lagi...")
        STOP.wait(3.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["rtsp", "file", "webcam"], default="rtsp")
    ap.add_argument("--cam", type=int, default=0, help="index webcam (--source webcam), default 0")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--host", default=None, help="override config server.host")
    ap.add_argument("--port", type=int, default=None, help="override config server.port")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    srv = cfg.get("server", {}) or {}
    host = args.host or srv.get("host", "127.0.0.1")    # CLI override > config > default
    port = args.port or int(srv.get("port", 5000))
    th = threading.Thread(target=pipeline_supervisor, args=(cfg, args), daemon=True)
    th.start()
    threading.Thread(target=maintenance, args=(cfg,), daemon=True).start()

    print(f"[WEB] http://{host}:{port}")
    try:
        app.run(host=host, port=port, threaded=True, debug=False)
    finally:
        STOP.set()
        th.join(timeout=2.0)


if __name__ == "__main__":
    main()
