"""Análise de futebol por blocos (jogo inteiro ou troços): tracking, equipas, bola, mapa 2D,
linha de fora de jogo e exportação de dados.

  .venv/bin/python analyze.py VIDEO --start 1500 --dur 60 --fps 25 --out out_teste
  .venv/bin/python analyze.py VIDEO --fps 12.5 --out out_jogo          # jogo inteiro (retoma se interrompido)
"""
import argparse
import os
import json
import pickle
import subprocess

import cv2
import numpy as np
import pandas as pd
import supervision as sv
from trackers import ByteTrackTracker
from ultralytics import YOLO

import track as T
from pitch_template import H as PH, W as PW
from stitch import stitch_ids
from identify import identify
from teams import TeamClassifier, jersey_feature

CHUNK_S = 40
MODELS = dict(person="yolo11m.pt", ball="ball_best.pt", pitch="pitch_pretrained/Model/weights/best.pt")
TEAM_BGR = [(255, 191, 0), (147, 20, 255), (0, 215, 255)]


# ---------------------------------------------------------------- utilitários
def duration(video):
    out = subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", video])
    return float(out.decode().strip())


def cut(video, start, dur, fps, out):
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", str(start), "-i", video, "-t", str(dur),
                    "-vf", f"fps={fps},scale=1280:-2", "-an", out], check=True)


def fit_team_classifier(video, total, model, device, cache="teams.pkl", n=80):
    if os.path.exists(cache):
        return pickle.load(open(cache, "rb"))
    feats = []
    os.makedirs("tmp_teams", exist_ok=True)
    for i in range(n):
        t = total * (i + 0.5) / n
        p = f"tmp_teams/f{i}.jpg"
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", str(t), "-i", video, "-frames:v", "1",
                        "-vf", "scale=1280:-2", p], check=True)
        frame = cv2.imread(p)
        det = T.detect_people(model, frame, device)
        for b, c in zip(det.xyxy, det.confidence):
            f = jersey_feature(frame, b) if c > 0.4 else None
            if f is not None:
                feats.append(f)
    tc = TeamClassifier().fit(feats)
    pickle.dump(tc, open(cache, "wb"))
    return tc


# ---------------------------------------------------------------- calibração do campo
def solve_homographies(frames, pitch_model, device, stride, max_prop):
    """Calibração do campo por frame, sequencial e validada pelas linhas brancas reais:
    cada frame parte da calibração do anterior (movimento da câmara) e das candidatas do modelo de pontos
    (com simetrias), é afinado pelas linhas e só é aceite se encaixar. Passa para a frente e para trás."""
    n = len(frames)
    G, last, fail = [None] * n, None, 0
    for t in range(1, n):
        g = T.frame_motion(frames[t - 1], frames[t])
        if g is None:
            fail += 1
            g = last if (last is not None and fail <= 3) else None
        else:
            fail, last = 0, g
        G[t] = g
    raw = {}
    for t in range(0, n, stride):
        h = T.model_homography_raw(pitch_model, frames[t], device)
        if h is not None:
            raw[t] = h
    sh = frames[0].shape

    def run(order, step):
        out, cur, age = [None] * n, None, 0
        for t in order:
            tracked = None
            s_ = step(t)
            if cur is not None and s_ is not None and age < max_prop:
                c = cur @ s_
                tracked = c / c[2, 2]
            cands = []
            if t in raw:
                cands.append(raw[t])
                cands += [F @ raw[t] for F in T._FLIPS[1:]]
            if tracked is None and not cands:
                cur = None
                continue
            dt = T.line_mask(frames[t])
            best_t = (T.line_score(dt, tracked), tracked) if tracked is not None else (-1.0, None)
            sc = sorted(((T.line_score(dt, H), H) for H in cands), key=lambda x: -x[0])
            best_m = sc[0] if sc else (-1.0, None)
            # mantém a continuidade (tracked) salvo se o modelo encaixar claramente melhor
            s0, H0 = best_t if best_t[0] >= best_m[0] - 0.08 and best_t[1] is not None else best_m
            if s0 < 0.75:
                pool = [h for h in (H0, tracked, best_m[1]) if h is not None]
                Hr, sr = T.refine_homography(frames[t], pool, dt)
                if Hr is not None and sr > s0:
                    s0, H0 = sr, Hr
            if H0 is not None and s0 >= T.LINE_MIN and T.plausible(H0, sh):
                out[t] = (H0, s0)
                cur, age = H0, (0 if t in raw and H0 is not tracked else age + 1)
            else:
                cur = None
        return out

    fw = run(range(n), lambda t: G[t])
    bw = run(range(n - 1, -1, -1), lambda t: np.linalg.inv(G[t + 1]) if t + 1 < n and G[t + 1] is not None else None)
    out, anchors = [None] * n, {}
    for t in range(n):
        c = [x for x in (fw[t], bw[t]) if x is not None]
        if c:
            out[t] = max(c, key=lambda x: x[1])[0]
            if t in raw:
                anchors[t] = out[t]
    return out, anchors


# ---------------------------------------------------------------- fora de jogo
def attacking_team(xy, team, ball_m):
    """Equipa atacante = a do jogador (0/1) mais próximo da bola."""
    if ball_m is None or len(xy) < 6:
        return None
    m = team < 2
    if m.sum() < 4:
        return None
    d = np.linalg.norm(xy - ball_m, axis=1)
    d[~m] = 1e9
    return int(team[int(np.argmin(d))]) if d.min() < 8 else None


def offside_state(xy, team, ball_m, att=None):
    """Devolve (x_linha_m, índices_atacantes_em_posição_irregular, equipa_defensora) ou None."""
    if ball_m is None or len(xy) < 8:
        return None
    ok = (xy[:, 0] > -3) & (xy[:, 0] < PW + 3) & (xy[:, 1] > -3) & (xy[:, 1] < PH + 3)
    if ok.sum() < 8:
        return None
    left = ball_m[0] < PW / 2
    dist = np.where(left, xy[:, 0], PW - xy[:, 0])
    near = ok & (dist < 40)
    if att is not None:
        dteam = 1 - att
    else:
        cnt = [int((near & (team == k)).sum()) for k in (0, 1)]
        if cnt[0] == cnt[1]:
            return None
        dteam = int(np.argmax(cnt))
    dd = np.sort(dist[ok & (team == dteam) & (dist > 1.0)])
    if len(dd) < 2:
        return None
    second_last = dd[0]  # o guarda-redes conta como último; o 2º último é o defesa mais recuado
    ball_d = ball_m[0] if left else PW - ball_m[0]
    line_d = min(second_last, ball_d)
    if line_d > PW / 2:
        return None
    flagged = np.where(ok & (team == 1 - dteam) & (dist < line_d - 0.5) & (dist < PW / 2))[0]
    x_line = line_d if left else PW - line_d
    return x_line, flagged, dteam


# ---------------------------------------------------------------- desenho
def draw_line_on_frame(frame, Hm, x_line, color=(0, 0, 255)):
    inv = np.linalg.inv(Hm)
    pts = np.array([[x_line, y] for y in np.linspace(-5, PH + 5, 40)], np.float32)
    img = T.project(inv, pts)
    h, w = frame.shape[:2]
    seg = [tuple(map(int, p)) for p in img if -50 < p[0] < w + 50 and -50 < p[1] < h + 50]
    for a, b in zip(seg, seg[1:]):
        cv2.line(frame, a, b, color, 3, cv2.LINE_AA)
    return frame


def render_map(base, xy, cls, ball_m, line, label):
    img = base.copy()
    if line is not None:
        x = int(line[0] * T.SCALE) + 20
        cv2.line(img, (x, 20), (x, int(PH * T.SCALE) + 20), (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(img, "Linha de fora de jogo (aprox.)", (min(x + 6, 760), 36), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    for k, ((x, y), c) in enumerate(zip(xy, cls)):
        if -5 <= x <= PW + 5 and -5 <= y <= PH + 5:
            p = (int(x * T.SCALE) + 20, int(y * T.SCALE) + 20)
            cv2.circle(img, p, 9, TEAM_BGR[c], -1)
            cv2.circle(img, p, 9, (0, 0, 0), 1)
            if line is not None and k in line[1]:
                cv2.circle(img, p, 14, (0, 0, 255), 2, cv2.LINE_AA)
    if ball_m is not None:
        cv2.circle(img, (int(ball_m[0] * T.SCALE) + 20, int(ball_m[1] * T.SCALE) + 20), 6, (255, 255, 255), -1)
    cv2.putText(img, label, (24, 710), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return img



def track_ball(frames, ball_c, homs, ball_m, device, fps):
    """Seguimento da bola: candidatos dentro do campo (via homografia), previsão por velocidade e, quando falha,
    nova deteção numa zona ampliada em volta da posição prevista (remates rápidos, bola pequena)."""
    n = len(frames)
    h, w = frames[0].shape[:2]
    allp = np.array([c[:2] for f in ball_c for c in f]) if any(len(f) for f in ball_c) else np.zeros((0, 2))
    static = set()
    if len(allp):
        keys, counts = np.unique((allp // 25).astype(int), axis=0, return_counts=True)
        static = {tuple(k) for k, c_ in zip(keys, counts) if c_ > 0.2 * n}   # falsos positivos fixos (bolas suplentes)

    def in_pitch(t, x, y, m=1.5):
        if homs[t] is None:
            return True
        px, py = T.project(homs[t], [[x, y]])[0]
        return -m <= px <= PW + m and -m <= py <= PH + m

    out = [[np.nan, np.nan] for _ in range(n)]
    pts = []          # últimas posições aceites (frame, x, y)
    gap = 0
    for t in range(n):
        cands = [c for c in ball_c[t] if tuple((np.array(c[:2]) // 25).astype(int)) not in static and in_pitch(t, c[0], c[1])]
        pred = None
        if pts:
            f0, x0, y0 = pts[-1]
            vx = vy = 0.0
            if len(pts) >= 2 and pts[-1][0] - pts[-2][0] <= 5:
                df = pts[-1][0] - pts[-2][0]
                vx, vy = (pts[-1][1] - pts[-2][1]) / df, (pts[-1][2] - pts[-2][2]) / df
            k = min(t - f0, 8)
            pred = (x0 + vx * k, y0 + vy * k)
        rad = min(45 + 30 * gap, 300) if pred else None
        pick = None
        if pred and gap <= 40:
            g = [c for c in cands if np.hypot(c[0] - pred[0], c[1] - pred[1]) <= rad]
            if g:
                pick = max(g, key=lambda c: c[2] - 0.5 * np.hypot(c[0] - pred[0], c[1] - pred[1]) / rad)
            elif gap <= 25:   # segunda deteção, ampliada, em volta da posição prevista
                S = 300
                cx = int(min(max(pred[0], S // 2), w - S // 2)); cy = int(min(max(pred[1], S // 2), h - S // 2))
                crop = frames[t][cy - S // 2: cy + S // 2, cx - S // 2: cx + S // 2]
                r = ball_m(cv2.resize(crop, (960, 960)), classes=[1], conf=0.04, imgsz=960, max_det=5, device=device, verbose=False)[0]
                bd = sv.Detections.from_ultralytics(r)
                z = []
                for xy, c in zip(bd.get_anchors_coordinates(sv.Position.CENTER), bd.confidence):
                    x_, y_ = cx - S // 2 + xy[0] * S / 960, cy - S // 2 + xy[1] * S / 960
                    if np.hypot(x_ - pred[0], y_ - pred[1]) <= rad and in_pitch(t, x_, y_):
                        z.append([x_, y_, float(c)])
                if z:
                    pick = max(z, key=lambda c: c[2])
        else:
            g = [c for c in cands if c[2] >= 0.2]
            if g:
                pick = max(g, key=lambda c: c[2])
        if pick is not None:
            out[t] = [pick[0], pick[1]]
            pts.append((t, pick[0], pick[1])); pts = pts[-3:]
            gap = 0
        else:
            gap += 1
    return out

# ---------------------------------------------------------------- bloco
def process_chunk(path, tag, fps, models, teams, device, out_dir, stride_pitch, ocr=False):
    frames = list(sv.get_video_frames_generator(path))
    n = len(frames)
    if n < 5:
        return
    person, ball_m, pitch_m = models
    def new_tracker():
        return ByteTrackTracker(frame_rate=fps, lost_track_buffer=int(6 * fps), track_activation_threshold=0.6,
                                minimum_consecutive_frames=max(2, int(0.2 * fps)), minimum_iou_threshold=0.05,
                                high_conf_det_threshold=0.5)
    tracker, id_off, max_tid, cuts = new_tracker(), 0, 0, []
    peos, raw_teams, ball_c = [], [], []
    idv = {}  # votos de identificação por tracker id
    for fi_, f in enumerate(frames):
        det = T.detect_people(person, f, device)
        rb = ball_m(f, classes=[1], conf=0.1, imgsz=1280, max_det=8, device=device, verbose=False)[0]
        bd = sv.Detections.from_ultralytics(rb)
        ball_c.append([[*xy, c] for xy, c in zip(bd.get_anchors_coordinates(sv.Position.CENTER), bd.confidence) if T.on_grass(f, *xy)])
        if fi_ > 0 and T.scene_cut(frames[fi_ - 1], f):  # corte de plano: IDs não atravessam cortes
            cuts.append(fi_)
            tracker, id_off = new_tracker(), max_tid
        p = tracker.update(det)
        p = p[p.tracker_id >= 0]
        if len(p):
            p.tracker_id = p.tracker_id + id_off
            max_tid = max(max_tid, int(p.tracker_id.max()))
        peos.append(p)
        raw_teams.append([teams.predict(jersey_feature(f, b)) for b in p.xyxy])
        if ocr and fi_ % 3 == 0:
            for b, tid, tm in zip(p.xyxy, p.tracker_id, raw_teams[-1]):
                if tm in (0, 1) and (b[3] - b[1]) >= 0.22 * f.shape[0]:
                    r_ = identify(f, b, tm)
                    if r_:
                        idv.setdefault(int(tid), []).append((r_[0], r_[1], r_[2], tm))

    homs, anchors = solve_homographies(frames, pitch_m, device, stride_pitch, max_prop=int(25 * fps))
    # planos fechados (jogadores muito grandes): a geometria do campo não é fiável, não mostrar mapa
    fh = frames[0].shape[0]
    for t_ in range(n):
        if len(peos[t_]) and np.median(peos[t_].xyxy[:, 3] - peos[t_].xyxy[:, 1]) > 0.28 * fh:
            homs[t_] = None
    ball_px = T.interpolate_ball(track_ball(frames, ball_c, homs, ball_m, device, fps), max_gap=max(3, int(0.24 * fps)))
    np.savez_compressed(f"{out_dir}/{tag}_H.npz", H=np.array([h_ if h_ is not None else np.full((3, 3), np.nan) for h_ in homs]))

    votes = {}
    for p, raw in zip(peos, raw_teams):
        for tid, t in zip(p.tracker_id, raw):
            if t is not None:
                votes.setdefault(int(tid), []).append(t)
    team_by_id = {i: int(np.bincount(v, minlength=3).argmax()) for i, v in votes.items()}
    pf = [(p.tracker_id, p.xyxy, [team_by_id.get(int(t), 0) for t in p.tracker_id]) for p in peos]
    merged = stitch_ids(pf, max_gap=int(1.6 * fps), per_frame_dist=int(10 * 25 / fps), cut_frames=cuts) if any(len(p) for p in peos) else {}
    renum = {}
    for p in peos:
        for t in p.tracker_id:
            renum.setdefault(merged[int(t)], len(renum) + 1)
    fid = {i: renum[g] for i, g in merged.items()}
    gv = {}
    for i, v in votes.items():
        gv.setdefault(fid[i], []).extend(v)
    team_of = {g: int(np.bincount(v, minlength=3).argmax()) for g, v in gv.items()}

    names = {}
    for tid, v in idv.items():
        g = fid[tid]
        names.setdefault(g, []).extend(v)
    nomes = {}
    for g, v in names.items():
        nums = {}
        for num, nome, conf, tm in v:
            d = nums.setdefault((num, nome, tm), [0, 0.0]); d[0] += 1; d[1] = max(d[1], conf)
        (num, nome, tm), (c, cmax) = max(nums.items(), key=lambda kv: (kv[1][0], kv[1][1]))
        if c >= 2 or cmax >= 0.88:
            nomes[int(g)] = dict(num=num, nome=nome, equipa="POR" if tm == 1 else "NAP", leituras=c)
    json.dump(nomes, open(f"{out_dir}/{tag}_nomes.json", "w"), ensure_ascii=False)

    # métricas por frame (posições no campo) e linha de fora de jogo
    base = T.draw_pitch()
    info = sv.VideoInfo(width=1280 + base.shape[1], height=720, fps=fps)
    box_ann = sv.EllipseAnnotator(color=sv.ColorPalette.from_hex(T.TEAM_COLORS), thickness=2)
    lab_ann = sv.LabelAnnotator(color=sv.ColorPalette.from_hex(T.TEAM_COLORS), text_position=sv.Position.BOTTOM_CENTER)
    ball_ann = sv.TriangleAnnotator(color=sv.Color.WHITE, base=20, height=17, color_lookup=sv.ColorLookup.INDEX)
    rows, off_rows, states, atts, ball_rows = [], [], [], [], []
    for t in range(n):
        p, Hm = peos[t], homs[t]
        ids = np.array([fid[int(i)] for i in p.tracker_id], dtype=int)
        cls = np.array([team_of.get(int(i), 0) for i in ids], dtype=int)
        if Hm is not None and len(p):
            xy = T.project(Hm, p.get_anchors_coordinates(sv.Position.BOTTOM_CENTER))
        else:
            xy = np.zeros((0, 2))
        bx = None
        if Hm is not None and not np.isnan(ball_px[t][0]):
            bx = T.project(Hm, [ball_px[t]])[0]
        ok_xy = len(xy) == len(cls) and len(xy) > 0
        atts.append(attacking_team(xy, cls, bx) if ok_xy else None)
        states.append([p, ids, cls, xy, bx, None])
    # equipa atacante suavizada (maioria numa janela de ~1.5 s) e depois a linha
    a = pd.Series([np.nan if x is None else x for x in atts])
    a = a.rolling(int(max(3, 1.5 * fps)) | 1, center=True, min_periods=1).apply(lambda w: np.nanmean(w) if np.isfinite(w).any() else np.nan).round()
    for t, s_ in enumerate(states):
        p, ids, cls, xy, bx, _ = s_
        if len(xy) == len(cls) and len(xy):
            att = None if np.isnan(a.iloc[t]) else int(a.iloc[t])
            s_[5] = offside_state(xy, cls, bx, att)
    # suaviza a linha de fora de jogo (mediana temporal) para não tremer
    xs = np.array([s[5][0] if s[5] else np.nan for s in states])
    xs_s = pd.Series(xs).rolling(int(max(3, fps // 2)) | 1, center=True, min_periods=1).median().values
    tmp_video = f"{out_dir}/{tag}.mp4"
    with sv.VideoSink(tmp_video, info) as sink:
        for t, (p, ids, cls, xy, bx, st) in enumerate(states):
            f = frames[t].copy()
            Hm = homs[t]
            p2 = sv.Detections(xyxy=p.xyxy, tracker_id=ids, class_id=cls)
            f = box_ann.annotate(f, p2)
            f = lab_ann.annotate(f, p2, [f"#{i}" for i in ids])
            if not np.isnan(ball_px[t][0]):
                bp = ball_px[t]
                f = ball_ann.annotate(f, sv.Detections(xyxy=np.array([[bp[0] - 6, bp[1] - 6, bp[0] + 6, bp[1] + 6]])))
            line = None
            if st is not None and not np.isnan(xs_s[t]):
                line = (xs_s[t], st[1])
                f = draw_line_on_frame(f, Hm, xs_s[t])
                for k in st[1]:
                    x1, y1, x2, y2 = map(int, p.xyxy[k])
                    cv2.rectangle(f, (x1, y1), (x2, y2), (0, 0, 255), 3)
            lab = "calibracao: ancora" if t in anchors else ("calibracao: propagada" if Hm is not None else "sem calibracao (plano fechado/sem linhas)")
            m = render_map(base, xy, cls, bx, line, lab) if Hm is not None else render_map(base, np.zeros((0, 2)), [], None, None, lab)
            sink.write_frame(np.hstack([f, m]))
            wide_ = int(len(p) >= 8 and np.median(p.xyxy[:, 3] - p.xyxy[:, 1]) < 0.22 * f.shape[0])
            ball_rows.append(dict(chunk=tag, frame=t, t_s=round(t / fps, 2), wide=wide_, bx_px=float(ball_px[t][0]), by_px=float(ball_px[t][1]),
                                  bx_m=float(bx[0]) if bx is not None else np.nan, by_m=float(bx[1]) if bx is not None else np.nan,
                                  calib="ancora" if t in anchors else ("propagada" if Hm is not None else "nenhuma"),
                                  att_team=float(a.iloc[t]) if not np.isnan(a.iloc[t]) else np.nan))
            for k, i in enumerate(ids):
                rows.append(dict(chunk=tag, frame=t, t_s=round(t / fps, 2), id=int(i), team=int(cls[k]),
                                 x_px=float(p.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)[k][0]),
                                 y_px=float(p.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)[k][1]),
                                 x_m=float(xy[k][0]) if len(xy) else np.nan, y_m=float(xy[k][1]) if len(xy) else np.nan))
            if line is not None:
                off_rows.append(dict(chunk=tag, frame=t, t_s=round(t / fps, 2), x_linha_m=round(float(line[0]), 1), n_em_posicao_irregular=int(len(line[1]))))
    pd.DataFrame(rows).to_csv(f"{out_dir}/{tag}_tracks.csv", index=False)
    pd.DataFrame(off_rows).to_csv(f"{out_dir}/{tag}_foradejogo.csv", index=False)
    pd.DataFrame(ball_rows).to_csv(f"{out_dir}/{tag}_bola.csv", index=False)
    cov = sum(h is not None for h in homs) / n
    print(f"[{tag}] frames={n} mapa={cov:.0%} ids={len(renum)} linhaFJ={len(off_rows)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--start", type=float, default=0)
    ap.add_argument("--dur", type=float, default=None)
    ap.add_argument("--fps", type=float, default=12.5)
    ap.add_argument("--out", default="out")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--chunk", type=float, default=CHUNK_S, help="duração de cada bloco em segundos (IDs só são contínuos dentro do bloco)")
    ap.add_argument("--ocr", action="store_true", help="ler números/nomes das camisolas (planos próximos)")
    ap.add_argument("--pitch-every", type=int, default=2, help="calibrar o campo a cada N frames")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    total = duration(a.video)
    dur = a.dur if a.dur else total - a.start
    person, ball, pitch = (YOLO(MODELS[k]) for k in ("person", "ball", "pitch"))
    teams = fit_team_classifier(a.video, total, person, a.device)
    t0 = a.start
    while t0 < a.start + dur - 1:
        d = min(a.chunk, a.start + dur - t0)
        tag = f"c{int(t0):05d}"
        if not os.path.exists(f"{a.out}/{tag}_foradejogo.csv"):
            cut(a.video, t0, d, a.fps, f"{a.out}/_tmp.mp4")
            process_chunk(f"{a.out}/_tmp.mp4", tag, a.fps, (person, ball, pitch), teams, a.device, a.out, a.pitch_every, a.ocr)
        t0 += d


if __name__ == "__main__":
    main()
