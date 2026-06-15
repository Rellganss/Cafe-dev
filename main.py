"""Cafe Detection System — entry count, cashier transaction, seat dwell, heatmap, peak hour.
Run: python main.py            (RTSP from config)
     python main.py --source file
     python main.py --no-window
"""
import os
import csv
import time
import argparse
import threading
from collections import defaultdict
from datetime import datetime

import cv2
import yaml
import numpy as np
import supervision as sv
from ultralytics import YOLO

from zones import ZoneManager

# RTSP low-latency: pakai TCP + buffer kecil (sebelum VideoCapture dibuat)
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS",
                      "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay")


def load_cfg(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def open_stream(cfg, use_file):
    src = cfg["source"]["file"] if use_file else cfg["source"]["rtsp"]
    cap = cv2.VideoCapture(src)
    if not use_file:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # low latency for RTSP
    return cap, src


class LatestFrame:
    """Thread baca RTSP terus, simpen cuma frame TERBARU. Proses lambat = drop
    frame basi, BUKAN numpuk delay. Kunci anti-lag buat live monitoring.
    opener(): callable -> cv2.VideoCapture baru, buat auto-reconnect kalau putus."""
    def __init__(self, cap, opener=None):
        self.cap = cap
        self.opener = opener
        self.lock = threading.Lock()
        self.frame = None
        self.ok = False
        self.stopped = False
        self.t = threading.Thread(target=self._loop, daemon=True)
        self.t.start()

    def _loop(self):
        fails = 0
        while not self.stopped:
            ok, f = self.cap.read()
            if not ok:
                fails += 1
                if self.opener and fails >= 25:    # RTSP putus -> reconnect
                    try:
                        self.cap.release()
                        self.cap = self.opener()
                    except Exception:
                        pass
                    fails = 0
                time.sleep(0.04)
                continue
            fails = 0
            with self.lock:
                self.ok, self.frame = True, f

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None
            return self.ok, self.frame.copy()

    def stop(self):
        self.stopped = True
        self.t.join(timeout=1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["rtsp", "file"], default="rtsp")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--no-window", action="store_true")
    ap.add_argument("--max-frames", type=int, default=0, help="stop after N processed frames (0=unlimited)")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    use_file = args.source == "file"
    os.makedirs("output", exist_ok=True)

    # --- model + tracker ---
    model = YOLO(cfg["model"]["weights"])
    tracker = sv.ByteTrack(
        track_activation_threshold=cfg["tracker"]["track_activation_threshold"],
        lost_track_buffer=cfg["tracker"]["lost_track_buffer"],
        minimum_matching_threshold=cfg["tracker"]["minimum_matching_threshold"],
        frame_rate=cfg["tracker"]["frame_rate"],
    )
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.4, text_thickness=1)

    # --- open stream, grab first frame to size everything ---
    cap, src = open_stream(cfg, use_file)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open source: {src}")
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("Cannot read first frame.")

    resize_w = cfg["processing"]["resize_width"]
    def prep(f):
        if resize_w and f.shape[1] != resize_w:
            h = int(f.shape[0] * resize_w / f.shape[1])
            return cv2.resize(f, (resize_w, h))
        return f
    frame = prep(frame)
    H, W = frame.shape[:2]
    src_fps = cap.get(cv2.CAP_PROP_FPS) or cfg["tracker"]["frame_rate"]

    zm = ZoneManager(cfg, (W, H))

    # RTSP: pakai grabber thread (selalu frame terbaru, anti delay numpuk).
    # File: baca sekuensial biasa (jangan drop frame).
    grabber = LatestFrame(cap, opener=lambda: open_stream(cfg, use_file)[0]) if not use_file else None

    # --- writers ---
    writer = None
    if cfg["output"]["save_video"]:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(cfg["output"]["annotated_video"], fourcc,
                                 max(1, src_fps / cfg["processing"]["frame_skip"]), (W, H))

    ev_file = open(cfg["output"]["events_csv"], "w", newline="", encoding="utf-8")
    ev_csv = csv.DictWriter(ev_file, fieldnames=["ts", "event", "zone", "track_id", "dwell_sec"])
    ev_csv.writeheader()

    peak = defaultdict(int)   # "YYYY-MM-DD HH:00" -> entries this hour
    show = cfg["output"]["show_window"] and not args.no_window
    skip = cfg["processing"]["frame_skip"]
    last_base = frame.copy()
    frame_idx = 0
    processed = 0
    t_start = time.time()
    conf, iou, imgsz, device, classes = (cfg["model"]["conf"], cfg["model"]["iou"],
                                         cfg["model"]["imgsz"], cfg["model"]["device"],
                                         cfg["model"]["classes"])
    min_area = float(cfg["logic"].get("min_box_area_px", 0) or 0)

    print(f"[INFO] {W}x{H} @ src_fps~{src_fps:.1f}  source={src}")
    print("[INFO] q to quit. Events -> output/events.csv")

    try:
        while True:
            if grabber is not None:
                ok, raw = grabber.read()
                if not ok:
                    time.sleep(0.01)   # belum ada frame baru
                    continue
            else:
                ok, raw = cap.read()
                if not ok:
                    break              # video ended

            frame_idx += 1
            if frame_idx % skip != 0:
                continue

            frame = prep(raw)
            last_base = frame.copy()
            processed += 1
            if args.max_frames and processed > args.max_frames:
                break

            # source-consistent time: wall clock for live, video-time for file
            if use_file:
                t_now = frame_idx / max(src_fps, 1e-6)
                iso_ts = f"frame_{frame_idx}"
            else:
                t_now = time.time()
                iso_ts = datetime.now().isoformat(timespec="seconds")

            # --- detect + track ---
            res = model(frame, conf=conf, iou=iou, imgsz=imgsz,
                        device=device, classes=classes, verbose=False)[0]
            det = sv.Detections.from_ultralytics(res)
            if min_area > 0 and len(det):            # buang bbox remeh (deteksi jauh palsu)
                wh = det.xyxy[:, 2:4] - det.xyxy[:, 0:2]
                det = det[(wh[:, 0] * wh[:, 1]) >= min_area]
            det = zm.filter_roi(det)                 # buang deteksi poster/dinding (kaki di luar ROI)
            det = tracker.update_with_detections(det)

            # --- zone logic ---
            events = zm.update(det, t_now, iso_ts)
            for e in events:
                ev_csv.writerow(e)
                if e["event"] == "enter":
                    hour_key = (datetime.now() if not use_file else datetime.now()).strftime("%Y-%m-%d %H:00")
                    peak[hour_key] += 1
                print(f"  [{e['event']}] {e['zone']} id={e['track_id']} dwell={e['dwell_sec']}")
            ev_file.flush()

            # --- annotate ---
            labels = []
            live = zm.live_dwell(t_now)
            for tid in (det.tracker_id if det.tracker_id is not None else []):
                tid = int(tid)
                if tid in live:
                    _, ztype, secs, sitting = live[tid]
                    tag = "DUDUK" if sitting else ("KASIR" if ztype == "cashier" else "")
                    labels.append(f"#{tid} {int(secs)}s {tag}".strip())
                else:
                    labels.append(f"#{tid}")
            annotated = box_annotator.annotate(frame.copy(), det)
            annotated = label_annotator.annotate(annotated, det, labels)
            annotated = zm.render_overlay(annotated)

            c = zm.counts
            cv2.putText(annotated, f"IN:{c['entered']} OUT:{c['exited']} now:{len(det)}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            if writer is not None:
                writer.write(annotated)
            if show:
                cv2.imshow("Cafe Detection", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        # flush: orang yg masih di zona pas berhenti -> log final dwell
        try:
            flush_ts = (f"frame_{frame_idx}" if use_file
                        else datetime.now().isoformat(timespec="seconds"))
            flush_t = (frame_idx / max(src_fps, 1e-6)) if use_file else time.time()
            for e in zm.flush(flush_t, flush_ts):
                ev_csv.writerow(e)
                print(f"  [flush:{e['event']}] {e['zone']} id={e['track_id']} dwell={e['dwell_sec']}")
        except Exception as ex:
            print("[warn] flush skipped:", ex)

        if grabber is not None:
            grabber.stop()
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        ev_file.close()

        # peak hour CSV
        with open(cfg["output"]["peak_csv"], "w", newline="", encoding="utf-8") as pf:
            pw = csv.writer(pf)
            pw.writerow(["hour", "entries"])
            for k in sorted(peak):
                pw.writerow([k, peak[k]])
        if peak:
            top = max(peak, key=peak.get)
            print(f"[PEAK] {top} -> {peak[top]} entries")

        zm.save_heatmap(cfg["output"]["heatmap_image"], base_frame=last_base)
        elapsed = max(time.time() - t_start, 1e-6)
        print(f"[PERF] processed {processed} frames in {elapsed:.1f}s -> {processed/elapsed:.1f} fps")
        print(f"[DONE] events={cfg['output']['events_csv']} "
              f"heatmap={cfg['output']['heatmap_image']} peak={cfg['output']['peak_csv']}")


if __name__ == "__main__":
    main()
