"""Penyimpanan SQLite: event persist + metrik harian + history. Anti-ilang pas restart."""
import os
import sqlite3
import threading
from datetime import datetime


class Store:
    def __init__(self, path="output/data.db"):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.lock = threading.Lock()
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute("""CREATE TABLE IF NOT EXISTS events(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT, day TEXT, hour TEXT,
            event TEXT, zone TEXT, track_id INTEGER, dwell_sec REAL)""")
        self.db.execute("CREATE INDEX IF NOT EXISTS ix_day ON events(day)")
        self.db.commit()

    def log(self, ev):
        now = datetime.now()
        ts = ev.get("ts") or now.isoformat(timespec="seconds")
        dwell = ev.get("dwell_sec")
        dwell = float(dwell) if dwell not in (None, "") else None
        with self.lock:
            self.db.execute(
                "INSERT INTO events(ts,day,hour,event,zone,track_id,dwell_sec) VALUES(?,?,?,?,?,?,?)",
                (ts, now.strftime("%Y-%m-%d"), now.strftime("%Y-%m-%d %H:00"),
                 ev["event"], ev["zone"], int(ev.get("track_id", -1)), dwell))
            self.db.commit()

    def _q(self, sql, args=()):
        with self.lock:
            return self.db.execute(sql, args).fetchall()

    def today(self):
        """Metrik hari ini buat dashboard."""
        day = datetime.now().strftime("%Y-%m-%d")
        c = lambda ev: self._q("SELECT COUNT(*) FROM events WHERE day=? AND event=?", (day, ev))[0][0]
        entered, exited = c("enter"), c("exit")
        trans, seated = c("transaction"), c("seated")
        avg = self._q("SELECT AVG(dwell_sec) FROM events WHERE day=? AND event='seated'", (day,))[0][0]
        pk = self._q("SELECT hour,COUNT(*) c FROM events WHERE day=? AND event='enter' "
                     "GROUP BY hour ORDER BY c DESC LIMIT 1", (day,))
        return {
            "entered": entered, "exited": exited,
            "transactions": trans, "seated": seated,
            "avg_seat_sec": round(avg, 1) if avg else 0,
            "conversion": round(trans / entered, 2) if entered else 0,
            "peak": {"hour": pk[0][0], "entries": pk[0][1]} if pk else None,
        }

    def prune(self, days):
        """Hapus event lebih tua dari `days` hari. Jaga DB gak balon. 0=skip."""
        if days <= 0:
            return 0
        with self.lock:
            cur = self.db.execute(
                "DELETE FROM events WHERE day < date('now', ?)", (f"-{int(days)} days",))
            self.db.commit()
            return cur.rowcount

    def history(self, days=7):
        """Rekap per hari (terbaru dulu)."""
        rows = self._q(
            "SELECT day,"
            " SUM(event='enter'), SUM(event='transaction'), SUM(event='seated'),"
            " AVG(CASE WHEN event='seated' THEN dwell_sec END)"
            " FROM events GROUP BY day ORDER BY day DESC LIMIT ?", (days,))
        return [{"day": r[0], "entered": r[1] or 0, "transactions": r[2] or 0,
                 "seated": r[3] or 0, "avg_seat_sec": round(r[4], 1) if r[4] else 0}
                for r in rows]
