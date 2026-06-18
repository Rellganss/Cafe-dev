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
        self.seat_memory_sec = float(L.get("seat_memory_sec", 30))         # ingat orang duduk yg ilang sekian detik
        self.seat_memory_px = float(L.get("seat_memory_px", 120))          # radius posisi buat nyocokin orang yg balik
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
        self.zones = []          # cashier/seat -> dihitung dwell
        self.staff_zones = []    # staff (zona kerja) -> orang di sini = STAFF, gak dihitung
        for z in cfg.get("zones", []) or []:
            pz = sv.PolygonZone(polygon=np.array(z["polygon"], dtype=np.int32),
                                triggering_anchors=[line_anchor])
            if z["type"] == "staff":
                self.staff_zones.append({"name": z["name"], "zone": pz})
            else:
                self.zones.append({"name": z["name"], "type": z["type"],
                                   "zone": pz, "tracks": {}})
        self.staff_ids = set()   # tid yg pernah kelihatan di zona kerja -> staff (sticky)

        # --- auto-seat: chair (COCO) yg distabilin -> zona kursi otomatis ---
        self.auto_seat = bool(L.get("auto_seat_from_chair", False))
        self.auto_seat_ttl = float(L.get("auto_seat_ttl_sec", 3.0))   # slot ilang kalau chair gak kelihatan sekian detik
        self.auto_seat_grow = float(L.get("auto_seat_grow_up", 0.6))  # perbesar zona ke ATAS x tinggi kursi (badan org duduk)
        self.auto_seats = []     # slot: {name,type,zone,tracks,box(EMA),last_seen}
        self._slot_n = 0

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

        # ---- staff: siapa di zona kerja -> tandai STAFF (sticky), gak dihitung customer ----
        for Sz in self.staff_zones:
            smask = Sz["zone"].trigger(detections)
            for i in range(len(smask)):
                if smask[i] and i < len(ids):
                    self.staff_ids.add(int(ids[i]))

        # ---- polygon dwell (zona manual + auto-seat dari chair) ----
        for Z in self.zones:
            self._dwell_zone(Z, ids, detections, t_now, iso_ts, events)
        for Z in self.auto_seats:
            self._dwell_zone(Z, ids, detections, t_now, iso_ts, events)

        # ---- heatmap (titik kaki) ----
        if self.heat_decay > 0:
            self.heat *= (1.0 - self.heat_decay)
        for box in detections.xyxy:
            cx = int((box[0] + box[2]) / 2 / self.hm_scale)
            cy = int(box[3] / self.hm_scale)
            if 0 <= cy < self.heat.shape[0] and 0 <= cx < self.heat.shape[1]:
                self.heat[cy, cx] += 1.0

        return events

    @staticmethod
    def _iou(a, b):
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter / ua if ua > 0 else 0.0

    def _seat_polygon(self, box):
        """Perbesar box chair ke atas (badan orang duduk) biar anchor center kena."""
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        W, H = self.frame_wh
        ex1, ex2 = max(0, x1 - 0.15 * w), min(W, x2 + 0.15 * w)
        ey1, ey2 = max(0, y1 - self.auto_seat_grow * h), min(H, y2 + 0.1 * h)
        return np.array([[ex1, ey1], [ex2, ey1], [ex2, ey2], [ex1, ey2]], dtype=np.int32)

    def sync_auto_seats(self, chair_boxes, t_now):
        """Chair (COCO) yg distabilin -> slot kursi otomatis. EMA smooth + TTL anti-kedip."""
        if not self.auto_seat:
            if self.auto_seats:        # baru dimatiin -> bersihin slot
                self.auto_seats = []
            return
        a = 0.6
        used = set()
        for cb in chair_boxes:
            cb = [float(v) for v in cb]
            best, best_iou = -1, 0.25
            for j, slot in enumerate(self.auto_seats):
                if j in used:
                    continue
                v = self._iou(cb, slot["box"])
                if v > best_iou:
                    best, best_iou = j, v
            if best >= 0:
                s = self.auto_seats[best]
                s["box"] = [a * s["box"][k] + (1 - a) * cb[k] for k in range(4)]   # EMA
                s["last_seen"] = t_now
                used.add(best)
            else:
                self._slot_n += 1
                self.auto_seats.append({"name": f"auto_{self._slot_n}", "type": "seat",
                                        "box": cb, "last_seen": t_now, "tracks": {}, "zone": None})
        # buang slot basi (chair lama gak kelihatan & gak ada orang aktif)
        self.auto_seats = [s for s in self.auto_seats
                           if t_now - s["last_seen"] <= self.auto_seat_ttl
                           or any(t_now - st["last_in"] < self.grace for st in s["tracks"].values())]
        # rebuild PolygonZone tiap slot dari box terbaru
        for s in self.auto_seats:
            s["zone"] = sv.PolygonZone(polygon=self._seat_polygon(s["box"]),
                                       triggering_anchors=[self.zone_anchor])

    def _dwell_zone(self, Z, ids, detections, t_now, iso_ts, events):
        """Logika dwell satu zona: debounce + stillness. Dipakai zona manual & auto-seat."""
        mask = Z["zone"].trigger(detections)
        inside = {}
        for i in range(len(mask)):
            if mask[i] and i < len(ids):
                tid = int(ids[i])
                if tid in self.staff_ids:        # staff -> gak dihitung
                    continue
                inside[tid] = _feet(detections.xyxy[i])

        tr = Z["tracks"]
        mem = Z.setdefault("memory", [])
        use_mem = Z["type"] == "seat" and self.seat_memory_sec > 0   # memori cuma buat kursi
        for tid, pos in inside.items():
            st = tr.get(tid)
            if st is None:
                if use_mem:        # coba LANJUTIN orang yg sama (abis ke-occlude / ID switch)
                    best, bj = self.seat_memory_px, -1
                    for j, m in enumerate(mem):
                        if t_now - m["vacated_at"] > self.seat_memory_sec:
                            continue
                        d = np.hypot(pos[0] - m["last_pos"][0], pos[1] - m["last_pos"][1])
                        if d <= best:
                            best, bj = d, j
                    if bj >= 0:
                        st = mem.pop(bj)            # RESUME: enter/still/qualified dipertahankan
                        st["last_pos"], st["last_in"] = pos, t_now
                        tr[tid] = st
                if st is None:
                    tr[tid] = {"enter": t_now, "last_in": t_now, "last_pos": pos,
                               "still_since": t_now, "qualified": False}
                    st = tr[tid]
            else:
                step = np.hypot(pos[0] - st["last_pos"][0], pos[1] - st["last_pos"][1])
                if step > self.move_eps:
                    st["still_since"] = t_now
                st["last_pos"] = pos
                st["last_in"] = t_now
            if not st["qualified"]:
                if Z["type"] == "cashier":
                    d = t_now - st["enter"]
                    if d >= self.cashier_min and (self.cashier_max <= 0 or d <= self.cashier_max):
                        st["qualified"] = True
                else:
                    if (t_now - st["still_since"]) >= self.seat_min:
                        st["qualified"] = True

        # keluar zona lewat grace
        for tid in list(tr.keys()):
            if tid in inside:
                continue
            st = tr.pop(tid)
            if t_now - st["last_in"] < self.grace:
                tr[tid] = st                       # masih toleransi, jangan apa-apa
                continue
            if use_mem:
                st["vacated_at"] = t_now           # SIMPAN ke memori, jangan finalize dulu
                mem.append(st)
            else:
                ev = self._finalize(Z, st, iso_ts)
                if ev:
                    ev["track_id"] = tid
                    events.append(ev)

        # memori kadaluarsa (orang beneran pergi) -> baru finalize event-nya
        if use_mem and mem:
            keep = []
            for m in mem:
                if t_now - m["vacated_at"] > self.seat_memory_sec:
                    ev = self._finalize(Z, m, iso_ts)
                    if ev:
                        events.append(ev)
                else:
                    keep.append(m)
            Z["memory"] = keep

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
        for Z in self.zones + self.auto_seats:
            for tid, st in list(Z["tracks"].items()):
                ev = self._finalize(Z, st, iso_ts)
                if ev:
                    ev["track_id"] = tid
                    events.append(ev)
            Z["tracks"].clear()
            for m in Z.get("memory", []):          # orang yg lagi "diingat" pas stream stop
                ev = self._finalize(Z, m, iso_ts)
                if ev:
                    events.append(ev)
            Z["memory"] = []
        return events

    def live_dwell(self, t_now):
        """Overlay: {tid: (zone, type, detik_di_zona, sitting_bool)}."""
        out = {}
        for Z in self.zones + self.auto_seats:
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

    def occupancy(self, t_now):
        """Metrik live: meja keisi (manual+auto), total, orang di kasir sekarang."""
        seat_zones = [Z for Z in self.zones if Z["type"] == "seat"] + self.auto_seats
        seats_total = len(seat_zones)
        seats_busy = sum(1 for Z in seat_zones
                         if any(t_now - st["last_in"] < 0.5 for st in Z["tracks"].values()))
        cashier_now = sum(sum(1 for st in Z["tracks"].values() if t_now - st["last_in"] < 0.5)
                          for Z in self.zones if Z["type"] == "cashier")
        return {"seats_total": seats_total, "seats_busy": seats_busy,
                "seats_pct": round(100 * seats_busy / seats_total) if seats_total else 0,
                "cashier_now": cashier_now}

    def render_overlay(self, frame):
        if self.roi is not None:
            cv2.polylines(frame, [self.roi.reshape((-1, 1, 2))], True, (255, 100, 0), 2)
            cv2.putText(frame, "ROI deteksi", tuple(self.roi[0]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 2)
        for Sz in self.staff_zones:               # zona kerja = ungu
            poly = Sz["zone"].polygon.reshape((-1, 1, 2))
            cv2.polylines(frame, [poly], True, (200, 0, 200), 2)
            x, y = Sz["zone"].polygon[0]
            cv2.putText(frame, Sz["name"] + " (staff)", (int(x), int(y) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 0, 200), 2)
        for Z in self.zones:
            poly = Z["zone"].polygon.reshape((-1, 1, 2))
            color = (0, 165, 255) if Z["type"] == "cashier" else (0, 255, 0)
            cv2.polylines(frame, [poly], True, color, 2)
            x, y = Z["zone"].polygon[0]
            cv2.putText(frame, Z["name"], (int(x), int(y) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        for Z in self.auto_seats:                 # kursi auto = ijo muda tipis
            poly = Z["zone"].polygon.reshape((-1, 1, 2))
            cv2.polylines(frame, [poly], True, (150, 255, 150), 1)
            x, y = Z["zone"].polygon[0]
            cv2.putText(frame, Z["name"], (int(x), int(y) - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 255, 150), 1)
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
