"""Interactive zone/line editor — AUTO-SAVE. Gambar ROI & garis pakai mouse.

Run: python setup_zones.py            (grab 1 frame dari RTSP)
     python setup_zones.py --source file
     python setup_zones.py --image snap.png

Kontrol:
  Pilih mode dulu (tombol):
    'l' = GARIS masuk   |  'p' = kotak KASIR   |  's' = kotak KURSI
    'r' = ROI DETEKSI (area lantai; orang di luar ini diabaikan -> buang poster dinding)
  Gambar:
    klik KIRI   : tambah titik
    GARIS  -> auto-save begitu klik titik ke-2
    KOTAK  -> klik KANAN (atau ENTER) buat tutup & save (min 3 titik)
  Lain:
    'z' = undo titik terakhir
    'x' = hapus shape terakhir (tersimpan)
    'f' = flip arah IN/OUT garis terakhir (liat panah kuning "IN")
    'q' = keluar
  Panah kuning "IN" = sisi yg dihitung MASUK. Salah arah? pencet 'f'.
  Tiap shape LANGSUNG ke-save ke config.yaml. Gak perlu tombol simpan.
"""
import argparse
import cv2
import yaml
import numpy as np

MODES = {"l": ("line", (255, 0, 255)),
         "p": ("cashier", (0, 165, 255)),
         "s": ("seat", (0, 255, 0)),
         "r": ("roi", (255, 100, 0))}


def grab_frame(args, cfg):
    if args.image:
        f = cv2.imread(args.image)
        if f is None:
            raise SystemExit(f"Cannot read image {args.image}")
        return f
    src = cfg["source"]["file"] if args.source == "file" else cfg["source"]["rtsp"]
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {src}")
    ok, f = cap.read()
    cap.release()
    if not ok:
        raise SystemExit("Cannot read frame")
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["rtsp", "file"], default="rtsp")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--image", default=None)
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    frame = grab_frame(args, cfg)
    rw = cfg["processing"]["resize_width"]
    if rw and frame.shape[1] != rw:
        h = int(frame.shape[0] * rw / frame.shape[1])
        frame = cv2.resize(frame, (rw, h))

    # preload existing shapes biar bisa nambah, bukan timpa
    state = {"mode": "line", "color": (255, 0, 255), "pts": [],
             "lines": list(cfg.get("lines") or []),
             "zones": list(cfg.get("zones") or []),
             "roi": list(cfg.get("detect_roi") or []), "n": 0}
    state["n"] = len(state["lines"]) + len(state["zones"])

    def save_config():
        cfg["lines"] = state["lines"]
        cfg["zones"] = state["zones"]
        cfg["detect_roi"] = state["roi"]
        with open(args.config, "w", encoding="utf-8") as fh:
            yaml.safe_dump(cfg, fh, sort_keys=False, allow_unicode=True)
        print(f"[SAVED] {len(state['lines'])} lines, {len(state['zones'])} zones, "
              f"roi:{len(state['roi'])}pts -> {args.config}", flush=True)

    def commit():
        pts, mode = state["pts"], state["mode"]
        if mode == "roi":
            if len(pts) < 3:
                print("[!] ROI butuh >=3 titik", flush=True)
                return
            state["roi"] = [list(p) for p in pts]
            print(f"[+] ROI deteksi ({len(pts)} titik)", flush=True)
            state["pts"] = []
            save_config()
            return
        if mode == "line":
            if len(pts) < 2:
                return
            state["n"] += 1
            state["lines"].append({"name": f"garis_{state['n']}", "start": pts[0], "end": pts[1]})
            print(f"[+] GARIS '{state['lines'][-1]['name']}' {pts[0]}->{pts[1]}", flush=True)
        else:
            if len(pts) < 3:
                print("[!] kotak butuh >=3 titik (klik lagi)", flush=True)
                return
            state["n"] += 1
            nm = ("kasir" if mode == "cashier" else "meja") + f"_{state['n']}"
            state["zones"].append({"name": nm, "type": mode, "polygon": [list(p) for p in pts]})
            print(f"[+] {mode.upper()} '{nm}' ({len(pts)} titik)", flush=True)
        state["pts"] = []
        save_config()

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["pts"].append([x, y])
            print(f"[click] pt {len(state['pts'])} @ ({x},{y}) mode={state['mode']}", flush=True)
            if state["mode"] == "line" and len(state["pts"]) == 2:
                commit()                                   # garis auto-save di titik ke-2
        elif event == cv2.EVENT_RBUTTONDOWN:
            if state["mode"] != "line":
                commit()                                   # kotak tutup + save

    cv2.namedWindow("setup")
    cv2.setMouseCallback("setup", on_mouse)
    print(__doc__, flush=True)
    print(f"[start] mode={state['mode']} (l=garis p=kasir s=kursi)", flush=True)

    while True:
        disp = frame.copy()
        for ln in state["lines"]:
            s, e = ln["start"], ln["end"]
            if ln.get("flip"):
                s, e = e, s
            cv2.line(disp, tuple(s), tuple(e), (255, 0, 255), 2)
            mx, my = (s[0] + e[0]) / 2, (s[1] + e[1]) / 2
            vx, vy = e[0] - s[0], e[1] - s[1]
            nn = (vx * vx + vy * vy) ** 0.5 or 1.0
            ax, ay = vy / nn * 40, -vx / nn * 40
            cv2.arrowedLine(disp, (int(mx), int(my)), (int(mx + ax), int(my + ay)),
                            (0, 255, 255), 2, tipLength=0.3)
            cv2.putText(disp, "IN", (int(mx + ax), int(my + ay)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        for z in state["zones"]:
            col = (0, 165, 255) if z["type"] == "cashier" else (0, 255, 0)
            cv2.polylines(disp, [np.array(z["polygon"], np.int32).reshape(-1, 1, 2)], True, col, 2)
            cv2.putText(disp, z["name"], tuple(z["polygon"][0]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
        if len(state["roi"]) >= 3:
            cv2.polylines(disp, [np.array(state["roi"], np.int32).reshape(-1, 1, 2)], True, (255, 100, 0), 2)
            cv2.putText(disp, "ROI", tuple(state["roi"][0]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 2)
        for p in state["pts"]:
            cv2.circle(disp, tuple(p), 5, state["color"], -1)
        if len(state["pts"]) > 1:
            cv2.polylines(disp, [np.array(state["pts"], np.int32).reshape(-1, 1, 2)],
                          False, state["color"], 1)
        hint = "garis: klik 2 titik" if state["mode"] == "line" else "kotak: klik titik, klik-KANAN tutup"
        cv2.putText(disp, f"MODE {state['mode'].upper()} | {hint}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, state["color"], 2)
        cv2.putText(disp, f"lines:{len(state['lines'])} zones:{len(state['zones'])} pts:{len(state['pts'])}",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.imshow("setup", disp)

        k = cv2.waitKey(20) & 0xFF
        if k == ord("q"):
            break
        elif k in (13, ord("c")):          # ENTER juga bisa tutup kotak
            commit()
        elif k == ord("z") and state["pts"]:
            state["pts"].pop()
        elif k == ord("f"):                # flip arah IN/OUT garis terakhir
            if state["lines"]:
                state["lines"][-1]["flip"] = not state["lines"][-1].get("flip", False)
                print(f"[flip] {state['lines'][-1]['name']} flip={state['lines'][-1]['flip']}", flush=True)
                save_config()
        elif k == ord("x"):
            if state["zones"]:
                print("[-] hapus", state["zones"].pop()["name"], flush=True); save_config()
            elif state["lines"]:
                print("[-] hapus", state["lines"].pop()["name"], flush=True); save_config()
        elif chr(k) in MODES:
            state["mode"], state["color"] = MODES[chr(k)]
            state["pts"] = []
            print(f"[mode] {state['mode']}", flush=True)

    cv2.destroyAllWindows()
    print(f"[exit] total {len(state['lines'])} lines, {len(state['zones'])} zones", flush=True)


if __name__ == "__main__":
    main()
