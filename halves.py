"""Metade do jogo em cada instante do vídeo, a partir do relógio do marcador (timeline.csv)."""
import numpy as np
import pandas as pd

_C = {}


def first_half(t):
    if "d" not in _C:
        d = pd.read_csv("timeline.csv").dropna(subset=["clock"])
        _C["d"] = (d.t.values.astype(float), d.clock.values.astype(float))
    ts, cl = _C["d"]
    if len(ts) == 0:
        return True
    i = int(np.argmin(np.abs(ts - t)))
    return bool(cl[i] <= 2760)  # até ao 46:00 (descontos incluídos)
