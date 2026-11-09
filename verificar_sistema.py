"""Verificação automática do sistema, nível a nível. Escreve verificacao.txt e devolve código 1 se algo crítico falhar."""
import glob, json, os, subprocess, sys
import numpy as np
import pandas as pd

R = []   # (nivel, nome, estado, detalhe)  estado: OK / AVISO / FALHA


def chk(nivel, nome, cond, detalhe="", critico=True):
    R.append((nivel, nome, "OK" if cond else ("FALHA" if critico else "AVISO"), detalhe))


def dur(path):
    return float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path]).decode().strip())


# ---- 1. ficheiros
chk("1 ficheiros", "vídeo original do resumo", os.path.exists("destaques/resumo.mp4"))
chk("1 ficheiros", "vídeo do painel (x264)", os.path.exists("painel/jogo.mp4"))
if os.path.exists("painel/jogo.mp4") and os.path.exists("destaques/resumo.mp4"):
    d1, d2 = dur("destaques/resumo.mp4"), dur("painel/jogo.mp4")
    chk("1 ficheiros", "duração painel = original", abs(d1 - d2) < 1, f"{d1:.1f}s vs {d2:.1f}s")
else:
    d1 = 618.95
log = open("destaques_out.log").read() if os.path.exists("destaques_out.log") else ""
n_h = len(glob.glob("destaques_out/c*_H.npz")); need_h = int(np.ceil(d1 / 60))
chk("1 ficheiros", "processamento terminou sem erros", "Traceback" not in log and ("FIM" in log or n_h >= need_h), f"{n_h}/{need_h} blocos completos" + ("" if log else " (sem log)"))

# ---- 2. blocos processados
cs = sorted(glob.glob("destaques_out/c*_tracks.csv"))
need = int(np.ceil(d1 / 60))
chk("2 processamento", "todos os blocos de 60 s existem", len(cs) == need, f"{len(cs)}/{need}")
tot = wide = wide_map = allmap = 0
rows = []
for f in cs:
    b = pd.read_csv(f.replace("_tracks", "_bola"))
    t0 = int(os.path.basename(f)[1:6])
    w = b[b.wide == 1]
    tot += len(b); wide += len(w); wide_map += int((w.calib != "nenhuma").sum()); allmap += int((b.calib != "nenhuma").sum())
    rows.append((t0, len(b), len(w), (w.calib != "nenhuma").mean() if len(w) else float("nan")))
chk("2 processamento", "frames analisados ≈ duração×25", abs(tot - d1 * 25) < 60, f"{tot} frames")
cov = wide_map / max(wide, 1)
chk("3 calibração", "jogo aberto com mapa ≥ 85%", cov >= 0.85, f"{cov:.0%} ({wide_map}/{wide} frames)", critico=False)
chk("3 calibração", "jogo aberto com mapa ≥ 70%", cov >= 0.70, f"{cov:.0%}")
for t0, n, nw, c in rows:
    if nw and c < 0.7:
        chk("3 calibração", f"bloco {t0}-{t0+60}s jogo aberto ≥ 70%", False, f"{c:.0%} de {nw} frames", critico=False)

# ---- 4. dados do painel
idx = json.load(open("painel/data/index.json")) if os.path.exists("painel/data/index.json") else {}
chk("4 painel", "index.json", bool(idx))
if idx:
    segs = idx["segments"]
    chk("4 painel", "segmentos cobrem o vídeo", len(segs) == need and abs(max(s["t1"] for s in segs) - d1) < 2, f"{len(segs)} segmentos")
    bad = 0
    for s in segs:
        fr = json.load(open("painel/data/" + s["file"]))
        ts = [f[0] for f in fr]
        if ts != sorted(ts) or any(len(f) < 9 for f in fr):
            bad += 1
    chk("4 painel", "ficheiros de segmentos íntegros", bad == 0, f"{bad} com problemas")
    ev = [e for e in idx["events"] if e["tipo"] == "golo"]
    chk("4 painel", "5 golos no painel", len(ev) == 5)
    chk("4 painel", "duração no índice = vídeo", abs(idx["duracao"] - d1) < 1)

# ---- 5. golos: dados para a repetição tática
if idx:
    allf = []
    for s in idx["segments"]:
        allf += json.load(open("painel/data/" + s["file"]))
    allf.sort(key=lambda f: f[0])
    T = np.array([f[0] for f in allf])
    for e in ev:
        tg = e["t"] + 6
        w = [f for f in allf if tg - 14 <= f[0] <= tg + 0.6]
        n = len(w)
        calib = np.mean([f[1] > 0 for f in w]) if n else 0
        seen = np.mean([sum(1 for s in f[6] if s[5]) >= 6 for f in w]) if n else 0
        ball = np.mean([f[2] is not None and f[2][0] is not None for f in w]) if n else 0
        carr = len({f[8] for f in w if f[8]})
        chk("5 golos", f"golo {e['n']}: frames de jogo ao vivo calibrados (≥50%)", calib >= 0.5, f"mapa {calib:.0%}, ≥6 jogadores à vista {seen:.0%}, bola {ball:.0%}, portadores distintos {carr}", critico=False)

# ---- 6. identidades
nm = json.load(open("painel/data/names.json")) if os.path.exists("painel/data/names.json") else {}
lab = Counter = {}
chk("6 identidades", "names.json existe", os.path.exists("painel/data/names.json"))
gr = sum(1 for v in nm.values() if "GK" in v)
chk("6 identidades", "guarda-redes identificados", gr > 0, f"{gr} jogadores-bloco", critico=False)
chk("6 identidades", "nomes lidos por OCR/propagação", len(nm) - gr > 0, f"{len(nm)} identidades no total", critico=False)

# ---- relatório
out = ["VERIFICAÇÃO DO SISTEMA", ""]
last = None
for nivel, nome, est, det in R:
    if nivel != last:
        out.append(f"[{nivel}]"); last = nivel
    out.append(f"  {est:6s} {nome}" + (f" — {det}" if det else ""))
nf = sum(1 for r in R if r[2] == "FALHA"); na = sum(1 for r in R if r[2] == "AVISO")
out += ["", f"RESUMO: {len(R) - nf - na} OK, {na} avisos, {nf} falhas"]
open("verificacao.txt", "w").write("\n".join(out))
print("\n".join(out))
sys.exit(1 if nf else 0)
