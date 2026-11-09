"""Exporta os dados processados para o painel (painel/data/*.json). Pode ser reexecutado à medida que o
processamento do jogo avança: o painel mostra o que já existe e a cobertura na linha do tempo.
Uso: .venv/bin/python build_dashboard.py
"""
import glob
import json
import os

import numpy as np
import pandas as pd

import halves
from slots import SlotTracker, sid_of

W, H = 105.0, 68.0
OUT = "painel/data"
import subprocess
DURACAO = float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", "destaques/resumo.mp4"]).decode().strip())
os.makedirs(OUT, exist_ok=True)
PORTO = 1


def second_half_start():
    d = pd.read_csv("timeline.csv").dropna(subset=["clock"])
    d = d[(d.t > 2800) & (d.clock > 2700)]
    return float(d.t.iloc[0] - (d.clock.iloc[0] - 2700)) if len(d) else 3100.0


def clock_to_video(minute, t2, second=False):
    """Minuto de jogo -> segundo de vídeo, a partir do marcador."""
    d = pd.read_csv("timeline.csv").dropna(subset=["clock"])
    d = d[(d.clock >= minute * 60) & (d.clock < minute * 60 + 60)]
    return float(d.t.median()) if len(d) else None


def read(path, cols):
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=cols)


def load_segment(path, src, tracker=None):
    try:
        lab = json.load(open(f"{OUT}/names.json"))
    except Exception:
        lab = {}
    tag = os.path.basename(path).split("_")[0]
    t0 = int(tag[1:])
    tr = read(path, ["frame", "id", "team", "x_m", "y_m"])
    ball = read(path.replace("_tracks", "_bola"), ["frame", "t_s", "bx_m", "by_m", "calib", "att_team"])
    off = read(path.replace("_tracks", "_foradejogo"), ["frame", "x_linha_m", "n_em_posicao_irregular"])
    fps = 1.0 / np.median(np.diff(sorted(ball.t_s.unique()))) if len(ball) > 3 else 12.5
    calib_map = {"nenhuma": 0, "propagada": 1, "ancora": 2}
    # linha de fora de jogo só vale dentro do campo e longe da linha de golo (aí não informa nada)
    offd = {int(r.frame): (round(r.x_linha_m, 1), int(r.n_em_posicao_irregular)) for r in off.itertuples()
            if 3.0 <= r.x_linha_m <= W - 3.0}
    by_frame = {f: g for f, g in tr.groupby("frame")}
    frames = []
    for b in ball.itertuples():
        calib = calib_map[b.calib]
        players = []
        g = by_frame.get(b.frame)
        if g is not None:
            for r in g.itertuples():
                if calib and -4 <= r.x_m <= W + 4 and -4 <= r.y_m <= H + 4:
                    xm, ym = round(r.x_m, 1), round(r.y_m, 1)
                else:
                    xm = ym = None
                if calib and xm is None:   # fora do campo (staff, banco, público): não etiquetar
                    continue
                if int(r.team) == 2 and not lab.get(f"{src}:{t0}:{int(r.id)}", "").split(" ")[0] in ("GK", "REF"):
                    continue   # classe 2 sem papel confirmado (staff, jogador mal classificado): não vai para o mapa
                players.append([int(r.id), int(r.team), xm, ym, int(r.x_px), int(r.y_px)])
        ball_xy = None
        if not np.isnan(b.bx_px):
            ball_xy = [round(b.bx_m, 1) if calib and not np.isnan(b.bx_m) else None,
                       round(b.by_m, 1) if calib and not np.isnan(b.by_m) else None, int(b.bx_px), int(b.by_px)]
        att = None if np.isnan(b.att_team) else int(b.att_team)
        t_abs = t0 + b.t_s
        dets = [(p[1], p[2], p[3]) for p in players if p[2] is not None]
        ball_m = (ball_xy[0], ball_xy[1]) if ball_xy and ball_xy[0] is not None else None
        sl, bh, car, fj = tracker.step(t_abs, calib > 0, dets, ball_m)
        slots_out = [[sid_of(s), s.team, int(round(s.dx * 10)), int(round(s.dy * 10)), int(round(min(t_abs - s.last, 99) * 10)), 1 if s.seen else 0] for s in sl]
        bh = [round(bh[0], 1), round(bh[1], 1), round(bh[2], 1)] if bh else None
        frames.append([round(t_abs, 2), calib, ball_xy, [fj, 0] if fj is not None else None, players, att, slots_out, bh, car])
    # quadros isolados sem deteções (1-3 seguidos, rodeados de quadros bons): repetem o quadro anterior em vez de piscar
    for i in range(1, len(frames) - 1):
        f = frames[i]
        if f[1] > 0 and not f[4] and frames[i - 1][4]:
            j = i
            while j < len(frames) and not frames[j][4] and frames[j][1] > 0 and j - i < 3:
                j += 1
            if j < len(frames) and frames[j][4] and j - i <= 3:
                for k in range(i, j):
                    g = frames[k]; pv = frames[i - 1]
                    g[4], g[6], g[3], g[7], g[8] = pv[4], pv[6], pv[3], pv[7], pv[8]
    fn = f"{src}_{tag}.json"
    json.dump(frames, open(f"{OUT}/{fn}", "w"), separators=(",", ":"))
    names = {}
    nf = path.replace("_tracks.csv", "_nomes.json")
    if os.path.exists(nf):
        names = {k: f"{v['num']} {v['nome']}" for k, v in json.load(open(nf)).items()}
    return dict(src=src, t0=t0, t1=round(t0 + ball.t_s.max(), 2), fps=round(fps, 1), file=fn, names=names), frames


def main():
    import identidades
    identidades.main()
    t2 = second_half_start()
    info = json.load(open("match_info.json"))
    segments, all_frames = [], {}
    for d in ["destaques_out"]:
        tracker = SlotTracker(halves.first_half)
        for p in sorted(glob.glob(f"{d}/c*_tracks.csv")):
            if not os.path.exists(p.replace("_tracks", "_bola")):
                continue
            seg, fr = load_segment(p, os.path.basename(d), tracker)
            segments.append(seg)
            all_frames[(seg["src"], seg["t0"])] = fr

    # marcador/relógio por segundo
    tl = pd.read_csv("timeline.csv")
    tl[["por", "nap", "clock"]] = tl[["por", "nap", "clock"]].ffill().bfill()
    timeline = [[int(r.t), int(r.clock), int(r.por), int(r.nap)] for r in tl.itertuples()]

    # estatísticas em bins de 10 s (só do jogo completo, sem duplicar clips)
    bins = {}
    for (src, t0), fr in all_frames.items():
        for t, calib, ball, line, players, att, *_ in fr:
            b = int(t // 10)
            s = bins.setdefault(b, dict(n=0, cal=0, att=[0, 0], z=[0, 0, 0], lh=[[], []], lc=[[], []], np=[0, 0]))
            s["n"] += 1
            if calib:
                s["cal"] += 1
                if att is not None:
                    s["att"][att] += 1
                if ball and ball[0] is not None:
                    x = ball[0] if halves.first_half(t) else W - ball[0]  # sentido de ataque do Porto
                    s["z"][0 if x < W / 3 else (1 if x < 2 * W / 3 else 2)] += 1
                for k in (0, 1):
                    xs = [p[2] for p in players if p[1] == k and p[2] is not None]
                    if len(xs) >= 4:
                        defends_left = halves.first_half(t) == (k == PORTO)
                        d = np.sort([x if defends_left else W - x for x in xs if (x if defends_left else W - x) > 3])
                        if len(d) >= 4:
                            s["lh"][k].append(float(d[0])); s["lc"][k].append(float(np.percentile(d, 90) - np.percentile(d, 10)))
                            s["np"][k] += 1
    series = []
    for b, s in sorted(bins.items()):
        series.append([b * 10, s["n"], s["cal"], s["att"], s["z"],
                       [round(float(np.median(s["lh"][k])), 1) if s["lh"][k] else None for k in (0, 1)],
                       [round(float(np.median(s["lc"][k])), 1) if s["lc"][k] else None for k in (0, 1)]])

    # lances de perigo: bola dentro da grande área adversária >= 1.2 s (juntando falhas < 1.5 s)
    danger = []
    for (src, t0), fr in all_frames.items():
        cur = None
        for t, calib, ball, line, players, att, *_ in fr:
            inbox = None
            clk = tl.clock.iloc[min(int(t), len(tl) - 1)]
            if ball and ball[0] is not None and sum(p[2] is not None for p in players) >= 8 and clk >= 90 and 13.84 <= ball[1] <= 54.16:
                if ball[0] >= W - 16.5:
                    inbox = "dir"
                elif ball[0] <= 16.5:
                    inbox = "esq"
            if inbox:
                if cur and cur["lado"] == inbox and t - cur["t1"] < 1.5:
                    cur["t1"] = t
                else:
                    if cur and cur["t1"] - cur["t0"] >= 2.0:
                        danger.append(cur)
                    cur = dict(t0=t, t1=t, lado=inbox)
        if cur and cur["t1"] - cur["t0"] >= 2.0:
            danger.append(cur)
    for d in danger:
        right = d["lado"] == "dir"
        porto_attacks_right = halves.first_half(d["t0"])
        d["equipa"] = "POR" if right == porto_attacks_right else "NAP"
        d["tipo"] = "perigo"

    # eventos verificados + Flashscore
    events = []
    for g in info["golos"]:
        t = g.get("t_gol_video") or {1: 848.5, 5: 5766.0}.get(g["n"])
        events.append(dict(t=t - 6 if t else None, tipo="golo", n=g["n"], equipa=g["equipa"], minuto=g["minuto"],
                           marcador=g["marcador"], assistencia=g["assistencia"],
                           titulo=f"Goal {g['n']} · {g['minuto']}′ {g['marcador']}", detalhe=g.get("descricao_verificada") or g["tipo"]
                           + (f" · assist: {g['assistencia']}" if g["assistencia"] else "")))
    extra = [(21, "cartão/falta", "Anguissa (Napoli) — foul", False), (35, "cartão/falta", "Rrahmani (Napoli) — misconduct", False),
             (36, "cartão/falta", "McTominay (Napoli) — holding", False), (56, "substituição", "Samu for André Silva", True),
             (67, "substituição", "Gimenez for Sainz; Zaidu for M. Fernandes", True), (78, "substituição", "Hwang for Veiga; Fofana for Pepê; Lang for Anguissa; Giovane for Vergara", True),
             (81, "substituição", "Lucca for Politano", True), (88, "cartão/falta", "Marin (Napoli) — foul", True)]
    for m, tipo, txt, second in extra:
        t = clock_to_video(m, t2, second)
        if t is not None and not (0 <= t <= DURACAO):
            t = None
        if t:
            events.append(dict(t=t, tipo=tipo, titulo=f"{m}′ {txt}", detalhe="Source: Flashscore"))
    events = [e for e in events if e["t"] is not None]
    events.sort(key=lambda e: e["t"])
    index = dict(jogo=info["jogo"], duracao=DURACAO, t2=0, segments=segments, timeline=timeline, series=series,
                 danger=[dict(t0=round(d["t0"], 1), t1=round(d["t1"], 1), equipa=d["equipa"]) for d in danger],
                 events=events, stats_flashscore=info["stats_flashscore"], elenco=info["elenco"])
    json.dump(index, open(f"{OUT}/index.json", "w"), separators=(",", ":"), ensure_ascii=False)
    print(f"segmentos={len(segments)} bins_stats={len(series)} perigo={len(danger)} eventos={len(events)} 2ªparte={t2:.0f}s")


if __name__ == "__main__":
    main()
