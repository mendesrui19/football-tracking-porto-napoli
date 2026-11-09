"""Auditoria da localização no campo: encaixe das linhas do campo (score) em cada frame calibrado."""
import glob, os, subprocess, sys
import numpy as np, pandas as pd
import track as T

V, OUT = "destaques/resumo.mp4", "destaques_out"


def frames(start, n):
    p = subprocess.Popen(["ffmpeg", "-nostdin", "-v", "error", "-ss", str(start), "-i", V, "-vf", "fps=25,scale=1280:-2", "-frames:v", str(n),
                          "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE)
    sz = 1280 * 720 * 3
    while True:
        b = p.stdout.read(sz)
        if len(b) < sz:
            break
        yield np.frombuffer(b, np.uint8).reshape(720, 1280, 3)
    p.stdout.close(); p.wait()


if __name__ == "__main__":
    rows = []
    for f in sorted(glob.glob(f"{OUT}/c*_bola.csv")):
        t0 = int(os.path.basename(f)[1:6]); b = pd.read_csv(f); Hs = np.load(f"{OUT}/c{t0:05d}_H.npz")["H"]
        for i, fr in enumerate(frames(t0, len(b))):
            if i >= len(b) or np.isnan(Hs[i][0, 0]) or i % 2:
                continue
            s = T.line_score(T.line_mask(fr), Hs[i])
            rows.append((t0, i, t0 + i / 25.0, b.at[i, "calib"], int(b.at[i, "wide"]), round(s, 3)))
        print(f"[c{t0:05d}] auditado", flush=True)
    d = pd.DataFrame(rows, columns=["t0", "frame", "T", "calib", "wide", "score"])
    d.to_csv("audit_pos.csv", index=False)
    print(d.score.describe().round(2).to_dict())
    for lo, hi, nome in ((0.0, 0.3, "<0,30 (mal)"), (0.3, 0.45, "0,30-0,45 (fraco)"), (0.45, 0.6, "0,45-0,60 (razoável)"), (0.6, 0.8, "0,60-0,80 (bom)"), (0.8, 2, ">0,80 (muito bom)")):
        m = (d.score >= lo) & (d.score < hi)
        print(f"  {nome:22s}: {m.sum():5d} frames ({m.mean():.0%})")
