"""Zone & line management: entry counting, cashier/seat dwell, heatmap.

Logika dwell diperkuat:
- Debounce exit (grace): deteksi miss sesaat / jitter batas zona TIDAK reset dwell.
- "Duduk" = kaki DIAM (feet stationary) sekian detik, bukan sekadar ada di zona.
- Filter staff: dwell kasir > batas = staff/nongkrong, bukan transaksi.
- Satu event per kunjungan (anti dobel), dwell = total waktu di zona.
"""
import numpy as np
import cv2
import supervision as sv


def _feet(box):
    """Titik kaki = tengah-bawah bbox (posisi lantai)."""
    return (float((box[0] + box[2]) / 2.0), float(box[3]))


class ZoneManager:
    def __init__(self, cfg, frame_wh):
        w, h = frame_wh
        self.frame_wh = frame_wh
        L = cfg["logic"]
        self.cashier_min = float(L["cashier_min_dwell_sec"])
        self.cashier_max = float(L.get("cashier_max_dwell_sec", 0) or 0)   # 0 = no cap
        self.seat_min = float(L["seat_min_dwell_sec"])
        self.move_eps = float(L.get("seat_move_eps_px", 12))               # gerak antar-frame < ini = diam
        self.grace = float(L.get("zone_exit_grace_sec", 1.5))              # debounce keluar zona
        self.heat_decay = float(L.get("heatmap_decay", 0.0))
        self.heat_gamma = float(L.get("heatmap_gamma", 0.5))

        # anchor garis: titik bbox yg dicek nyebrang. RTSP top-down -> bottom_center (kaki).
        # webcam close-up (kaki ke-crop) -> center, biar badan nyebrang kehitung.
        anchor_map = {"bottom_center": sv.Position.BOTTOM_CENTER,
                      "center": sv.Position.CENTER,
                      "top_center": sv.Position.TOP_CENTER}
        line_anchor = anchor_map.get(str(L.get("line_anchor", "bottom_center")).lower(),
                                     sv.Position.BOTTOM_CENTER)
        # frame berturut di sisi seberang biar kehitung. 1 = paling responsif.
        line_frames = max(1, int(L.get("line_crossing_frames", 1)))

        # --- entry lines ---
        # flip: true -> tuker start/end (balik arah IN/OUT) tanpa gambar ulang.
        self.lines = []
        for ln in cfg.get("lines", []) or []:
            s, e = ln["start"], ln["end"]
            if ln.get("flip"):
                s, e = e, s
            self.lines.append({"name": ln["name"], "zone": sv.LineZone(
                start=sv.Point(*s), end=sv.Point(*e),
                triggering_anchors=[line_anchor],
                minimum_crossing_threshold=line_frames)})

        # --- polygon zones: anchor sama kayak garis (bottom_center kaki / center webcam) ---
        self.zone_anchor = line_anchor
        self.zones = []
        for z in cfg.get("zones", []) or []:
            self.zones.append({
                "name": z["name"], "type": z["type"],
                "zone": sv.PolygonZone(polygon=np.array(z["polygon"], dtype=np.int32),
                                       triggering_anchors=[line_anchor]),
                "tracks": {},   # tid -> state dict
            })

        # --- detect ROI: cuma deteksi yg KAKI di dalam poligon ini (buang poster dinding) ---
        roi = cfg.get("detect_roi") or []
        self.roi = np.array(roi, dtype=np.int32) if len(roi) >= 3 else None

        self.hm_scale = 8
        self.heat = np.zeros((h // self.hm_scale, w // self.hm_scale), dtype=np.float32)

    def filter_roi(self, det):
        """Buang deteksi yg kakinya di luar ROI (mis orang di poster/banner dinding)."""
        if self.roi is None or len(det) == 0:
            return det
        keep = np.empty(len(det), dtype=bool)
        for i, box in enumerate(det.xyxy):
            fx, fy = _feet(box)
            keep[i] = cv2.pointPolygonTest(self.roi, (float(fx), float(fy)), False) >= 0
        return det[keep]

    # ---------- per-frame ----------
    def update(self, detections, t_now, iso_ts):
        events = []
        ids = detections.tracker_id if detections.tracker_id is not None else np.array([])

        # ---- line crossings ----
        for Ln in self.lines:
            cin, cout = Ln["zone"].trigger(detections)
            for i, v in enumerate(cin):
                if v:
                    events.append({"ts": iso_ts, "event": "enter", "zone": Ln["name"],
                                   "track_id": int(ids[i]) if i < len(ids) else -1, "dwell_sec": ""})
            for i, v in enumerate(cout):
                if v:
                    events.append({"ts": iso_ts, "event": "exit", "zone": Ln["name"],
                                   "track_id": int(ids[i]) if i < len(ids) else -1, "dwell_sec": ""})

        # ---- polygon dwell (debounce + stillness) ----
        for Z in self.zones:
            mask = Z["zone"].trigger(detections)
            inside = {}   # tid -> feet pos this frame
            for i in range(len(mask)):
                if mask[i] and i < len(ids):
                    inside[int(ids[i])] = _feet(detections.xyxy[i])

            tr = Z["tracks"]
            for tid, pos in inside.items():
                st = tr.get(tid)
                if st is None:
                    tr[tid] = {"enter": t_now, "last_in": t_now, "last_pos": pos,
                               "still_since": t_now, "qualified": False}
                    st = tr[tid]
                else:
                    step = np.hypot(pos[0] - st["last_pos"][0], pos[1] - st["last_pos"][1])
                    if step > self.move_eps:          # kaki gerak -> reset hitungan diam
                        st["still_since"] = t_now
                    st["last_pos"] = pos
                    st["last_in"] = t_now
                # cek lolos syarat (real-time, biar gak ilang kalau gak keluar zona)
                if not st["qualified"]:
                    if Z["type"] == "cashier":
                        d = t_now - st["enter"]
                        if d >= self.cashier_min and (self.cashier_max <= 0 or d <= self.cashier_max):
                            st["qualified"] = True
                    else:  # seat: kaki diam >= seat_min
                        if (t_now - st["still_since"]) >= self.seat_min:
                            st["qualified"] = True

            # finalize keluar zona (lewat grace)
            for tid in list(tr.keys()):
                if tid in inside:
                    continue
                st = tr[tid]
                if t_now - st["last_in"] < self.grace:
                    continue                          # masih dalam toleransi, jangan reset
                ev = self._finalize(Z, st, iso_ts)
                if ev:
                    ev["track_id"] = tid
                    events.append(ev)
                del tr[tid]

        # ---- heatmap (titik kaki) ----
        if self.heat_decay > 0:
            self.heat *= (1.0 - self.heat_decay)
        for box in detections.xyxy:
            cx = int((box[0] + box[2]) / 2 / self.hm_scale)
            cy = int(box[3] / self.hm_scale)
            if 0 <= cy < self.heat.shape[0] and 0 <= cx < self.heat.shape[1]:
                self.heat[cy, cx] += 1.0

        return events

    def _finalize(self, Z, st, iso_ts):
        """Bikin event kalau kunjungan ini lolos syarat. dwell = total waktu di zona."""
        if not st["qualified"]:
            return None
        dwell = st["last_in"] - st["enter"]
        if Z["type"] == "cashier":
            if self.cashier_max > 0 and dwell > self.cashier_max:
                return None                           # staff/nongkrong, bukan transaksi
            ev = "transaction"
        else:
            ev = "seated"
        return {"ts": iso_ts, "event": ev, "zone": Z["name"],
                "track_id": -1, "dwell_sec": round(dwell, 1)}

    def flush(self, t_now, iso_ts):
        """Stream stop: orang yg masih di zona di-log kalau lolos syarat."""
        events = []
        for Z in self.zones:
            for tid, st in list(Z["tracks"].items()):
                st["last_in"] = max(st["last_in"], t_now)
                ev = self._finalize(Z, st, iso_ts)
                if ev:
                    ev["track_id"] = tid
                    events.append(ev)
            Z["tracks"].clear()
        return events

    def live_dwell(self, t_now):
        """Overlay: {tid: (zone, type, detik_di_zona, sitting_bool)}."""
        out = {}
        for Z in self.zones:
            for tid, st in Z["tracks"].items():
                if t_now - st["last_in"] < 0.2:       # lagi di zona frame ini
                    sitting = Z["type"] == "seat" and (t_now - st["still_since"]) >= self.seat_min
                    out[tid] = (Z["name"], Z["type"], t_now - st["enter"], sitting)
        return out

    @property
    def counts(self):
        c = {"entered": 0, "exited": 0}
        for Ln in self.lines:
            c["entered"] += Ln["zone"].in_count
            c["exited"] += Ln["zone"].out_count
        return c

    def render_overlay(self, frame):
        if self.roi is not None:
            cv2.polylines(frame, [self.roi.reshape((-1, 1, 2))], True, (255, 100, 0), 2)
            cv2.putText(frame, "ROI deteksi", tuple(self.roi[0]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 2)
        for Z in self.zones:
            poly = Z["zone"].polygon.reshape((-1, 1, 2))
            color = (0, 165, 255) if Z["type"] == "cashier" else (0, 255, 0)
            cv2.polylines(frame, [poly], True, color, 2)
            x, y = Z["zone"].polygon[0]
            cv2.putText(frame, Z["name"], (int(x), int(y) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        for Ln in self.lines:
            s, e = Ln["zone"].vector.start, Ln["zone"].vector.end
            cv2.line(frame, (int(s.x), int(s.y)), (int(e.x), int(e.y)), (255, 0, 255), 2)
            cv2.putText(frame, Ln["name"], (int(s.x), int(s.y) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
            # panah arah IN: normal (vy,-vx) dari start->end (konvensi supervision)
            mx, my = (s.x + e.x) / 2, (s.y + e.y) / 2
            vx, vy = e.x - s.x, e.y - s.y
            n = (vx ** 2 + vy ** 2) ** 0.5 or 1.0
            ax, ay = vy / n * 45, -vx / n * 45     # arah sisi IN
            cv2.arrowedLine(frame, (int(mx), int(my)), (int(mx + ax), int(my + ay)),
                            (0, 255, 255), 2, tipLength=0.3)
            cv2.putText(frame, "IN", (int(mx + ax), int(my + ay)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        return frame

    def save_heatmap(self, path, base_frame=None):
        """Normalisasi percentile-99 + gamma: 1 titik rame gak nelen area lain."""
        h = self.heat.copy()
        nz = h[h > 0]
        if nz.size:
            ceil = np.percentile(nz, 99)              # buang outlier (spot staff)
            h = np.clip(h / max(ceil, 1e-6), 0, 1)
            h = np.power(h, self.heat_gamma)          # gamma<1 -> angkat area sepi
        h8 = (h * 255).astype(np.uint8)
        h8 = cv2.resize(h8, (self.frame_wh[0], self.frame_wh[1]), interpolation=cv2.INTER_LINEAR)
        h8 = cv2.GaussianBlur(h8, (0, 0), sigmaX=15)
        colored = cv2.applyColorMap(h8, cv2.COLORMAP_JET)
        if base_frame is not None:
            colored = cv2.addWeighted(base_frame, 0.5, colored, 0.5, 0)
        cv2.imwrite(path, colored)
