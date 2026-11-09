"""Corrige frames com homografia espelhada (esquerda/direita trocadas): o sentido da orientação (sinal do determinante)
tem de ser sempre o mesmo numa câmara real. Espelha de volta x_m -> 105 - x_m (o campo é simétrico) em H, jogadores e bola."""
import glob, os
import numpy as np
import pandas as pd
import track as T

W = 105.0
OUT = "destaques_out"
FX = np.array([[-1, 0, W], [0, 1, 0], [0, 0, 1.0]])


def det_sign(H):
    c = np.array([[640, 400], [641, 400], [640, 401]], float)
    p = T.project(H, c)
    J = np.array([[p[1, 0] - p[0, 0], p[2, 0] - p[0, 0]], [p[1, 1] - p[0, 1], p[2, 1] - p[0, 1]]])
    return np.sign(np.linalg.det(J))


if __name__ == "__main__":
    tot = 0
    for f in sorted(glob.glob(f"{OUT}/c*_H.npz")):
        t0 = int(os.path.basename(f)[1:6]); Hs = np.load(f)["H"].copy()
        bad = [i for i, H in enumerate(Hs) if not np.isnan(H[0, 0]) and det_sign(H) != 1]
        if not bad:
            continue
        tr = pd.read_csv(f"{OUT}/c{t0:05d}_tracks.csv"); b = pd.read_csv(f"{OUT}/c{t0:05d}_bola.csv")
        bs = set(bad)
        m = tr.frame.isin(bs) & tr.x_m.notna()
        tr.loc[m, "x_m"] = W - tr.loc[m, "x_m"]
        mb = b.frame.isin(bs) & b.bx_m.notna()
        b.loc[mb, "bx_m"] = W - b.loc[mb, "bx_m"]
        for i in bad:
            Hs[i] = FX @ Hs[i]
        tr.to_csv(f"{OUT}/c{t0:05d}_tracks.csv", index=False); b.to_csv(f"{OUT}/c{t0:05d}_bola.csv", index=False)
        np.savez_compressed(f"{OUT}/c{t0:05d}_H.npz", H=Hs)
        tot += len(bad)
        print(f"[c{t0:05d}] frames espelhados corrigidos: {len(bad)}", flush=True)
    print("TOTAL corrigidos:", tot)
