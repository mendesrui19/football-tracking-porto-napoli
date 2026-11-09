"""Une tracklets que o tracker partiu (oclusões, cortes de câmara) para reduzir trocas de ID."""
import numpy as np


def stitch_ids(per_frame, max_gap=40, base_dist=90, per_frame_dist=10, cut_frames=()):
    """per_frame: lista (por frame) de (tracker_ids, xyxy, team). Devolve dict id -> id final."""
    tl = {}
    for f, (ids, boxes, teams) in enumerate(per_frame):
        for i, b, t in zip(ids, boxes, teams):
            tl.setdefault(int(i), []).append((f, b, t))
    info = {i: dict(s=v[0][0], e=v[-1][0], first=v[0][1], last=v[-1][1],
                    team=int(np.bincount([x[2] for x in v]).argmax()),
                    frames={x[0] for x in v}) for i, v in tl.items()}
    parent = {i: i for i in info}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    group_frames = {i: set(info[i]["frames"]) for i in info}
    for b_id in sorted(info, key=lambda i: info[i]["s"]):
        B = info[b_id]
        best, best_d = None, 1e9
        for a_id in info:
            A = info[a_id]
            if find(a_id) == find(b_id) or not (0 < B["s"] - A["e"] <= max_gap) or A["team"] != B["team"] \
                    or any(A["e"] < c <= B["s"] for c in cut_frames):
                continue
            ca = ((A["last"][0] + A["last"][2]) / 2, (A["last"][1] + A["last"][3]) / 2)
            cb = ((B["first"][0] + B["first"][2]) / 2, (B["first"][1] + B["first"][3]) / 2)
            d = np.hypot(ca[0] - cb[0], ca[1] - cb[1])
            ha, hb = A["last"][3] - A["last"][1], B["first"][3] - B["first"][1]
            if d > base_dist + per_frame_dist * (B["s"] - A["e"]) or not (0.6 < hb / max(ha, 1) < 1.6):
                continue
            ra, rb = find(a_id), find(b_id)
            if group_frames[ra] & group_frames[rb]:
                continue
            if d < best_d:
                best, best_d = a_id, d
        if best is not None:
            ra, rb = find(best), find(b_id)
            parent[rb] = ra
            group_frames[ra] |= group_frames[rb]
    return {i: find(i) for i in info}
