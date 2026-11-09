"""Relatório de erros por momento (5 golos + lances de perigo): cobertura do mapa, jogadores, bola, linha de fora de jogo."""
import glob, json
import numpy as np

W = 105.0
ix = json.load(open("painel/data/index.json"))
fr = []
for s in ix["segments"]:
    fr += json.load(open("painel/data/" + s["file"]))
fr.sort(key=lambda f: f[0])
T = np.array([f[0] for f in fr])

moments = []
for e in ix["events"]:
    if e["tipo"] == "golo":
        tg = e["t"] + 6
        moments.append((f"GOAL {e['n']}", tg - 12, tg + 0.5))
for i, d in enumerate(ix["danger"]):
    if any(abs((e["t"] + 6) - d["t0"]) < 14 for e in ix["events"] if e["tipo"] == "golo"):
        continue
    moments.append((f"danger {i+1} {d['equipa']}", d["t0"] - 4, d["t1"] + 1))

rows = []
for name, a, b in moments:
    w = [f for f in fr[np.searchsorted(T, a):np.searchsorted(T, b)]]
    n = len(w)
    if n == 0:
        continue
    cal = np.mean([f[1] > 0 for f in w])
    wc = [f for f in w if f[1] > 0]
    seen = np.mean([sum(1 for s in f[6] if s[5]) for f in wc]) if wc else 0
    t0c = np.mean([sum(1 for s in f[6] if s[5] and s[1] == 0) for f in wc]) if wc else 0
    t1c = np.mean([sum(1 for s in f[6] if s[5] and s[1] == 1) for f in wc]) if wc else 0
    bpx = np.mean([f[2] is not None and f[2][2] is not None for f in w])
    bm = [f[2] for f in w if f[2] is not None and f[2][0] is not None]
    bmap = len(bm) / n
    jumps = 0
    last = None
    for f in w:
        b_ = f[7]
        if b_ is not None and b_[2] < 0.1:
            if last is not None and f[0] - last[2] < 0.1:
                if np.hypot(b_[0] - last[0], b_[1] - last[1]) / max(f[0] - last[2], 0.04) > 40:
                    jumps += 1
            last = (b_[0], b_[1], f[0])
    car = np.mean([f[8] is not None for f in wc]) if wc else 0
    off = [f[3][0] for f in w if f[3]]
    offj = int(np.sum(np.abs(np.diff(off)) > 3)) if len(off) > 1 else 0
    flags = []
    if cal < 0.6: flags.append("map gaps")
    if wc and seen < 8: flags.append("few players seen")
    if bpx < 0.6: flags.append("ball gaps")
    if jumps: flags.append(f"{jumps} ball jumps")
    if offj: flags.append(f"{offj} offside jumps")
    rows.append((name, a, b, n, cal, seen, t1c, t0c, bpx, bmap, car, flags))

lines = ["| moment | window (s) | map | players seen (POR/NAP) | ball in video | ball on map | carrier | problems |", "|---|---|---|---|---|---|---|---|"]
for name, a, b, n, cal, seen, t1c, t0c, bpx, bmap, car, flags in rows:
    lines.append(f"| {name} | {a:.0f}–{b:.0f} | {cal:.0%} | {seen:.1f} ({t1c:.1f}/{t0c:.1f}) | {bpx:.0%} | {bmap:.0%} | {car:.0%} | {', '.join(flags) or 'ok'} |")
ok = sum(1 for r in rows if not r[-1])
lines += ["", f"{ok} of {len(rows)} moments without flagged problems."]
open("qa_report.md", "w").write("\n".join(lines))
print("\n".join(lines))
