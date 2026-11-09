"""Linha temporal do jogo a partir do marcador (placar) da transmissão: relógio, resultado e golos."""
import glob, os, re, subprocess, sys
import cv2
import pandas as pd
import jersey_ocr as J


def parse(texts):
    joined = " ".join(texts)
    clock = re.search(r"(\d{1,3}):(\d{2})", joined)
    m = re.search(r"POR\W*([\dOo]{1,2})\W*([\dOo]{1,2})", joined) or re.search(r"([\dOo])\s+([\dOo])\s*NAP", joined)
    if not clock or not m:
        return None
    f = lambda s: int(s.replace("O", "0").replace("o", "0"))
    return int(clock.group(1)) * 60 + int(clock.group(2)), f(m.group(1)), f(m.group(2))


def build(video, out_csv="timeline.csv", crop="crop=700:90:30:20", every=1):
    os.makedirs("sb_frames", exist_ok=True)
    if not glob.glob("sb_frames/*.png"):
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", video, "-vf", f"fps=1/{every},{crop}", "-an", "sb_frames/s%05d.png"], check=True)
    rows = []
    for i, p in enumerate(sorted(glob.glob("sb_frames/*.png"))):
        im = cv2.resize(cv2.imread(p), None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        r = parse([t for t, c in J._recognize(im)])
        rows.append(dict(t=i * every, clock=r[0] if r else None, por=r[1] if r else None, nap=r[2] if r else None))
    df = pd.DataFrame(rows); df.to_csv(out_csv, index=False); return df


def goal_events(df, stable=4):
    """Um golo = mudança de resultado que se mantém `stable` leituras seguidas."""
    d = df.dropna(subset=["por", "nap"]).copy()
    d["score"] = list(zip(d.por.astype(int), d.nap.astype(int)))
    ev, cur = [], (0, 0)
    sc = list(d.score); ts = list(d.t); cl = list(d.clock)
    for i in range(len(sc) - stable):
        s = sc[i]
        if s != cur and all(x == s for x in sc[i:i + stable]) and sum(s) == sum(cur) + 1:
            ev.append(dict(t_video=ts[i], minuto=round(cl[i] / 60, 1), equipa="POR" if s[0] > cur[0] else "NAP", resultado=f"{s[0]}-{s[1]}"))
            cur = s
    return pd.DataFrame(ev)


if __name__ == "__main__":
    df = build(sys.argv[1])
    print("leituras válidas:", df.dropna().shape[0], "de", len(df))
    g = goal_events(df); print(g.to_string(index=False)); g.to_csv("golos.csv", index=False)
