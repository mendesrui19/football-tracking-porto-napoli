"""Preenche falhas de calibração em frames de JOGO ABERTO, só onde existe um frame calibrado vizinho:
propaga a homografia pelo movimento da câmara (nos dois sentidos), afina pelas linhas e aceita com limiar mais baixo
(0,20) porque a continuidade já está provada. Atualiza *_H.npz, *_bola.csv e as posições em metros dos jogadores."""
import glob, os, subprocess, sys
import cv2
import numpy as np
import pandas as pd
import track as T

V = "destaques/resumo.mp4"
OUT = "destaques_out"
MIN_SCORE = 0.20


def frames(start, n, fps=25):
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-ss", str(start), "-i", V, "-vf", f"fps={fps},scale=1280:-2", "-frames:v", str(n),
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    sz = 1280 * 720 * 3
    while True:
        b = p.stdout.read(sz)
        if len(b) < sz:
            break
        yield np.frombuffer(b, np.uint8).reshape(720, 1280, 3)
    p.stdout.close(); p.wait()


def propagate(fr, Hs, order, step_sign):
    """Propaga de frames calibrados para vizinhos em falta. order: lista de índices na ordem de propagação."""
    out = {}
    cur = None; prev_i = None
    for i in order:
        if not np.isnan(Hs[i][0, 0]):
            cur, prev_i = Hs[i], i
            continue
        if cur is None or prev_i is None:
            continue
        a, b = (prev_i, i) if step_sign > 0 else (i, prev_i)
        # movimento câmara: G mapeia frame posterior -> anterior
        G = T.frame_motion(fr[a], fr[b])
        if G is None:
            cur = None; continue
        cand = cur @ G if step_sign > 0 else cur @ np.linalg.inv(G)
        cand = cand / cand[2, 2]
        dt = T.line_mask(fr[i])
        H2, s = T.refine_homography(fr[i], [cand], dt)
        if H2 is not None and s >= MIN_SCORE and T.plausible(H2, fr[i].shape):
            out[i] = (H2 / H2[2, 2], s)
            cur, prev_i = H2 / H2[2, 2], i
        else:
            cur = None
    return out


def process(t0):
    b = pd.read_csv(f"{OUT}/c{t0:05d}_bola.csv")
    Hs = np.load(f"{OUT}/c{t0:05d}_H.npz")["H"].copy()
    n = len(b)
    gap = ((b.wide == 1) & (b.calib == "nenhuma")).values
    if not gap.any():
        return 0, 0
    # só intervalos curtos/médios (≤ 4 s) com vizinho calibrado
    idx = np.where(gap)[0]
    need = set()
    for i in idx:
        need.update(range(max(0, i - 100), min(n, i + 101)))
    lo, hi = min(need), max(need)
    fr = {}
    for k, f in enumerate(frames(t0 + lo / 25.0, hi - lo + 1)):
        fr[lo + k] = f
    rng = [i for i in range(lo, hi + 1) if i in fr]
    fw = propagate(fr, Hs, rng, +1)
    bw = propagate(fr, Hs, rng[::-1], -1)
    filled = 0
    for i in idx:
        c = [x for x in (fw.get(i), bw.get(i)) if x is not None]
        if c:
            Hs[i] = max(c, key=lambda x: x[1])[0]; filled += 1
    if filled:
        tr = pd.read_csv(f"{OUT}/c{t0:05d}_tracks.csv")
        for i in idx:
            if np.isnan(Hs[i][0, 0]) or b.at[i, "calib"] != "nenhuma":
                continue
            b.at[i, "calib"] = "propagada"
            m = tr.frame == i
            if m.any():
                xy = T.project(Hs[i], tr.loc[m, ["x_px", "y_px"]].values)
                tr.loc[m, "x_m"], tr.loc[m, "y_m"] = xy[:, 0], xy[:, 1]
            if not np.isnan(b.at[i, "bx_px"]):
                bxy = T.project(Hs[i], [[b.at[i, "bx_px"], b.at[i, "by_px"]]])[0]
                b.at[i, "bx_m"], b.at[i, "by_m"] = bxy
        tr.to_csv(f"{OUT}/c{t0:05d}_tracks.csv", index=False)
        b.to_csv(f"{OUT}/c{t0:05d}_bola.csv", index=False)
        np.savez_compressed(f"{OUT}/c{t0:05d}_H.npz", H=Hs)
    return int(gap.sum()), filled


if __name__ == "__main__":
    blocks = [int(x) for x in sys.argv[1:]] or sorted(int(os.path.basename(f)[1:6]) for f in glob.glob(f"{OUT}/c*_bola.csv"))
    tg = tf = 0
    for t0 in blocks:
        g, f = process(t0); tg += g; tf += f
        print(f"[c{t0:05d}] frames de jogo aberto sem mapa: {g} -> preenchidos {f}", flush=True)
    print(f"TOTAL: {tg} -> preenchidos {tf}")
