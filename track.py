"""Tracking de futebol a partir de vídeo broadcast.

Deteta e segue jogadores (YOLO + ByteTrack), separa equipas pela cor da camisola,
deteta a bola com um modelo próprio e, se houver modelo do campo, projeta tudo num mapa 2D.

Uso:
  .venv/bin/python track.py entrada.mp4 saida.mp4 \
      [--ball-model runs/ball/weights/best.pt] \
      [--pitch-model runs/pitch/weights/best.pt --map-out mapa2d.mp4]
"""
import argparse

import cv2
import numpy as np
import pandas as pd
import supervision as sv
from trackers import ByteTrackTracker
from ultralytics import YOLO

from pitch_template import H as PITCH_H, PITCH_PTS, W as PITCH_W
from jersey_ocr import read_number
from stitch import stitch_ids
from teams import TeamClassifier, jersey_feature

PERSON, COCO_BALL = 0, 32  # classes COCO
BALL_CLASS = 1  # no modelo treinado: 0 jogador, 1 bola, 2 árbitro
TEAM_COLORS = ["#00BFFF", "#FF1493", "#FFD700"]
MAP_BGR = [(255, 191, 0), (147, 20, 255), (0, 215, 255)]
SCALE = 10  # píxeis por metro no mapa 2D


# ---------- equipas ----------
def feet_on_grass(frame, box, thr=0.3):
    """Descarta pessoas fora do relvado (público, bancos, planos fechados)."""
    x1, y1, x2, y2 = map(int, box)
    h = y2 - y1
    win = frame[max(0, y2 - int(0.12 * h)): y2 + int(0.12 * h), max(0, x1): x2]
    if win.size == 0:
        return False
    hsv = cv2.cvtColor(win, cv2.COLOR_BGR2HSV)
    green = (hsv[..., 0] > 30) & (hsv[..., 0] < 90) & (hsv[..., 1] > 50)
    return green.mean() > thr


def detect_people(model, frame, device):
    r = model(frame, classes=[PERSON], conf=0.25, imgsz=1280, device=device, verbose=False)[0]   # 1280 px: apanha 99% dos jogadores (640 px só 86%)
    det = sv.Detections.from_ultralytics(r).with_nms(0.5)
    keep = np.array([feet_on_grass(frame, b) for b in det.xyxy], dtype=bool) if len(det) else np.zeros(0, bool)
    return det[keep]


def fit_teams(video, model, device, samples=60):
    info = sv.VideoInfo.from_video_path(video)
    stride = max(1, (info.total_frames or samples * 15) // samples)
    feats = []
    for frame in sv.get_video_frames_generator(video, stride=stride):
        det = detect_people(model, frame, device)
        for b, c in zip(det.xyxy, det.confidence):
            f = jersey_feature(frame, b) if c > 0.4 else None
            if f is not None:
                feats.append(f)
    return TeamClassifier().fit(feats)


# ---------- bola ----------
def on_grass(frame, x, y, r=28):
    """A bola de jogo tem relva à volta; bolas suplentes junto às placas não."""
    h, w = frame.shape[:2]
    win = frame[max(0, int(y) - r): int(y) + r, max(0, int(x) - r): min(w, int(x) + r)]
    if win.size == 0:
        return False
    hsv = cv2.cvtColor(win, cv2.COLOR_BGR2HSV)
    green = (hsv[..., 0] > 30) & (hsv[..., 0] < 90) & (hsv[..., 1] > 60)
    return green.mean() > 0.5


def pick_ball(cands):
    """cands: por frame, lista de [x, y, conf]. Escolhe a trajetória plausível da bola."""
    allp = np.array([c[:2] for f in cands for c in f]) if any(len(f) for f in cands) else np.zeros((0, 2))
    static = set()
    if len(allp):
        keys, counts = np.unique((allp // 25).astype(int), axis=0, return_counts=True)
        static = {tuple(k) for k, n in zip(keys, counts) if n > 0.2 * len(cands)}  # falsos positivos fixos
    out, last, gap = [], None, 0
    for f in cands:
        f = [c for c in f if tuple((np.array(c[:2]) // 25).astype(int)) not in static]
        if last is not None and gap <= 30:
            rad = min(60 + 40 * gap, 400)
            f = [c for c in f if np.hypot(c[0] - last[0], c[1] - last[1]) <= rad]
        elif f:
            f = [c for c in f if c[2] >= 0.2]
        if f:
            best = max(f, key=lambda c: c[2])
            last, gap = best[:2], 0
            out.append([best[0], best[1]])
        else:
            gap += 1
            out.append([np.nan, np.nan])
    return out


def interpolate_ball(rows, max_gap=12):
    df = pd.DataFrame(rows, columns=["x", "y"])
    return df.interpolate(limit=max_gap, limit_area="inside").values


# ---------- campo / mapa 2D ----------
def homography_for_frame(pitch_model, frame, device, prev):
    """Homografia imagem -> campo (metros) a partir dos keypoints; reutiliza a anterior se falhar."""
    r = pitch_model(frame, imgsz=640, conf=0.25, device=device, verbose=False)[0]
    if r.keypoints is not None and len(r.keypoints.xy):
        k = r.keypoints.xy[0].cpu().numpy()
        c = r.keypoints.conf[0].cpu().numpy() if r.keypoints.conf is not None else np.ones(len(k))
        m = (c > 0.35) & (k.sum(axis=1) > 0)
        if m.sum() >= 5:
            Hm, inl = cv2.findHomography(k[m], PITCH_PTS[m], cv2.RANSAC, 1.5)
            if Hm is not None and inl.sum() >= 4 and plausible(Hm, frame.shape):
                Hm, sc = best_homography(frame, Hm)
                if sc >= LINE_MIN:
                    return Hm
    return prev


def scene_cut(prev, cur):
    """True se houver corte de plano entre dois frames consecutivos (histograma de cor)."""
    hp = cv2.calcHist([cv2.cvtColor(cv2.resize(prev, (320, 180)), cv2.COLOR_BGR2HSV)], [0, 1], None, [16, 16], [0, 180, 0, 256])
    hc = cv2.calcHist([cv2.cvtColor(cv2.resize(cur, (320, 180)), cv2.COLOR_BGR2HSV)], [0, 1], None, [16, 16], [0, 180, 0, 256])
    return cv2.compareHist(hp, hc, cv2.HISTCMP_CORREL) < 0.6


def frame_motion(prev, cur):
    """Homografia cur -> prev (movimento da câmara). Fluxo ótico sobre a relva; se falhar, câmara parada
    (diferença mínima) ou alinhamento ECC; devolve None só em cortes de plano."""
    sp, sc = (cv2.resize(x, None, fx=0.5, fy=0.5) for x in (prev, cur))
    gp, gc = (cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in (sp, sc))
    hsv = cv2.cvtColor(sc, cv2.COLOR_BGR2HSV)
    hp = cv2.calcHist([cv2.cvtColor(sp, cv2.COLOR_BGR2HSV)], [0, 1], None, [16, 16], [0, 180, 0, 256])
    hc = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
    if cv2.compareHist(hp, hc, cv2.HISTCMP_CORREL) < 0.7:  # corte de plano
        return None
    mask = (((hsv[..., 0] > 30) & (hsv[..., 0] < 90) & (hsv[..., 1] > 40)) * 255).astype(np.uint8)
    mask = cv2.erode(mask, np.ones((5, 5), np.uint8))
    pc = cv2.goodFeaturesToTrack(gc, 800, 0.003, 5, mask=mask)
    S = np.diag([0.5, 0.5, 1.0])
    if pc is not None and len(pc) >= 25:
        pp, st, _ = cv2.calcOpticalFlowPyrLK(gc, gp, pc, None)
        ok = st.ravel() == 1
        if ok.sum() >= 25:
            G, inl = cv2.findHomography(pc[ok], pp[ok], cv2.RANSAC, 2.0)
            if G is not None and inl.sum() >= 20:
                return np.linalg.inv(S) @ G @ S
    # câmara parada: imagens quase iguais
    ta, tb = (cv2.resize(x, (160, 90)) for x in (gp, gc))
    if float(np.mean(np.abs(ta.astype(np.float32) - tb.astype(np.float32)))) < 4.0:
        return np.eye(3)
    # alinhamento ECC (afim) a baixa resolução
    qa, qb = (cv2.resize(x, (320, 180)) for x in (gp, gc))
    warp = np.eye(2, 3, dtype=np.float32)
    try:
        cc, warp = cv2.findTransformECC(qb, qa, warp, cv2.MOTION_AFFINE,
                                        (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 1e-4), None, 5)
    except cv2.error:
        return None
    if cc < 0.85:
        return None
    G = np.vstack([warp, [0, 0, 1]]).astype(np.float64)
    Q = np.diag([0.25, 0.25, 1.0])  # coordenadas de 1280x720 -> 320x180
    return np.linalg.inv(Q) @ G @ Q


def plausible(Hm, shape):
    """Rejeita homografias degeneradas: a área do campo visível tem de ser razoável."""
    h, w = shape[:2]
    quad = project(Hm, [[0, 0.35 * h], [w, 0.35 * h], [w, h], [0, h]])
    if not np.isfinite(quad).all():
        return False
    x, y = quad[:, 0], quad[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    cx, cy = quad.mean(axis=0)
    return 300 < area < 9000 and -30 < cx < PITCH_W + 30 and -30 < cy < PITCH_H + 30



# ---------- validação da calibração pelas linhas do campo ----------
def _pitch_lines(step=0.5):
    pts = []
    def seg(a, b):
        n = int(np.hypot(b[0] - a[0], b[1] - a[1]) / step) + 2
        pts.extend(zip(np.linspace(a[0], b[0], n), np.linspace(a[1], b[1], n)))
    Wd, Hd = PITCH_W, PITCH_H
    for a, b in (((0, 0), (Wd, 0)), ((Wd, 0), (Wd, Hd)), ((Wd, Hd), (0, Hd)), ((0, Hd), (0, 0)), ((Wd / 2, 0), (Wd / 2, Hd))):
        seg(a, b)
    th = np.linspace(0, 2 * np.pi, int(2 * np.pi * 9.15 / step))
    pts.extend((Wd / 2 + 9.15 * np.cos(t), Hd / 2 + 9.15 * np.sin(t)) for t in th)
    for x0, x1, y0, y1 in ((0, 16.5, 13.84, 54.16), (Wd - 16.5, Wd, 13.84, 54.16), (0, 5.5, 24.84, 43.16), (Wd - 5.5, Wd, 24.84, 43.16)):
        xi = x1 if x0 == 0 else x0
        seg((x0, y0), (x1, y0)); seg((x0, y1), (x1, y1)); seg((xi, y0), (xi, y1))
    for cx, sgn in ((11.0, 1), (Wd - 11.0, -1)):  # arcos da grande área (fora da área)
        for t in np.linspace(-1.0, 1.0, 60):
            x, y = cx + sgn * 9.15 * np.cos(t), 34 + 9.15 * np.sin(t)
            if (sgn == 1 and x > 16.5) or (sgn == -1 and x < Wd - 16.5):
                pts.append((x, y))
    return np.array(pts, dtype=np.float64)


LINE_MIN = 0.30  # fração mínima de linhas do campo que têm de coincidir com linhas brancas reais
_LINES = _pitch_lines()
# variantes de simetria do campo (o campo é simétrico: o modelo pode trocar esquerda/direita)
_FLIPS = [np.eye(3),
          np.array([[-1, 0, PITCH_W], [0, -1, PITCH_H], [0, 0, 1.0]])]   # só a rotação de 180°: uma câmara real nunca produz um espelho
_FX = np.array([[-1, 0, PITCH_W], [0, 1, 0], [0, 0, 1.0]])


def det_sign(H):
    c = np.array([[640, 400], [641, 400], [640, 401]], float)
    p = project(H, c)
    return float(np.sign(np.linalg.det(np.array([[p[1, 0] - p[0, 0], p[2, 0] - p[0, 0]], [p[1, 1] - p[0, 1], p[2, 1] - p[0, 1]]]))))


def make_proper(H):
    """Homografia com orientação física (sinal +1); se vier espelhada, espelha de volta (o campo é simétrico)."""
    return H if det_sign(H) == 1 else _FX @ H


def line_mask(frame):
    small = cv2.resize(frame, (640, 360))
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    gray = hsv[..., 2]
    th = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)))
    white = (th > 20) & (hsv[..., 1] < 95)
    grass = ((hsv[..., 0] > 30) & (hsv[..., 0] < 90) & (hsv[..., 1] > 45)).astype(np.uint8)
    near = cv2.dilate(grass, np.ones((9, 9), np.uint8)) > 0
    mask = (white & near).astype(np.uint8)
    return cv2.distanceTransform(1 - mask, cv2.DIST_L2, 3)


def line_score(dt, H_img2pitch):
    """Fração dos pontos das linhas do campo projetadas que caem em cima de linhas brancas reais."""
    try:
        inv = np.linalg.inv(H_img2pitch)
    except np.linalg.LinAlgError:
        return 0.0
    p = (inv @ np.c_[_LINES, np.ones(len(_LINES))].T).T
    ok = p[:, 2] > 1e-6
    x, y = p[ok, 0] / p[ok, 2] * 0.5, p[ok, 1] / p[ok, 2] * 0.5
    ins = (x >= 0) & (x < 640) & (y >= 0) & (y < 360)
    if ins.sum() < 40:
        return 0.0
    d = dt[y[ins].astype(int), x[ins].astype(int)]
    return float((d < 3).mean())


def refine_homography(frame, cands, dt=None):
    """Afina homografias candidatas (imagem->campo) para maximizar o encaixe das linhas do campo nas linhas brancas.
    Devolve (H, score) da melhor."""
    from scipy.optimize import minimize
    if dt is None:
        dt = line_mask(frame)
    P3 = np.c_[_LINES, np.ones(len(_LINES))].T
    scored = sorted(((line_score(dt, H), H) for H in cands if H is not None), key=lambda x: -x[0])
    best = (None, 0.0)
    for s0, H in scored[:2]:
        try:
            Hi0 = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            continue

        def pts(Hi):
            p = (Hi @ P3).T
            ok = p[:, 2] > 1e-6
            x, y = p[ok, 0] / p[ok, 2] * 0.5, p[ok, 1] / p[ok, 2] * 0.5
            ins = (x >= 0) & (x < 640) & (y >= 0) & (y < 360)
            return x[ins], y[ins]

        x0, _ = pts(Hi0)
        n0 = max(len(x0), 1)

        def cost(th):
            P = np.array([[1 + th[0] * 0.05, th[1] * 0.05, th[2] * 20], [th[3] * 0.05, 1 + th[4] * 0.05, th[5] * 20],
                          [th[6] * 5e-4, th[7] * 5e-4, 1.0]])
            x, y = pts(P @ Hi0)
            if len(x) < 40:
                return 30.0
            d = dt[y.astype(int), x.astype(int)]
            return float(np.minimum(d, 10).mean()) + 4.0 * max(0.0, 1 - len(x) / n0)

        res = minimize(cost, np.zeros(8), method="Powell", options=dict(maxiter=5, xtol=1e-2, ftol=1e-3))
        th = res.x
        P = np.array([[1 + th[0] * 0.05, th[1] * 0.05, th[2] * 20], [th[3] * 0.05, 1 + th[4] * 0.05, th[5] * 20],
                      [th[6] * 5e-4, th[7] * 5e-4, 1.0]])
        try:
            Hr = np.linalg.inv(P @ Hi0)
        except np.linalg.LinAlgError:
            continue
        Hr = Hr / Hr[2, 2]
        s = line_score(dt, Hr)
        if s > best[1] and plausible(Hr, frame.shape):
            best = (Hr, s)
        if s0 > best[1]:
            best = (H, s0)
    return best


def best_homography(frame, H_img2pitch, min_score=0.0):
    """Escolhe, entre a homografia do modelo e as suas simetrias, a que melhor encaixa nas linhas; devolve (H, score)."""
    dt = line_mask(frame)
    s0 = line_score(dt, H_img2pitch)
    best = (H_img2pitch, s0)
    for F in _FLIPS[1:]:
        Hc = F @ H_img2pitch
        s = line_score(dt, Hc)
        if s > best[1] + 0.08:  # só troca de orientação se encaixar claramente melhor
            best = (Hc, s)
    return best


def model_homography_raw(pitch_model, frame, device):
    """Homografia do modelo de pontos sem validação (a validação é feita pelas linhas, no solver)."""
    r = pitch_model(frame, imgsz=640, conf=0.25, device=device, verbose=False)[0]
    if r.keypoints is None or not len(r.keypoints.xy):
        return None
    k = r.keypoints.xy[0].cpu().numpy()
    c = r.keypoints.conf[0].cpu().numpy() if r.keypoints.conf is not None else np.ones(len(k))
    m = (c > 0.3) & (k.sum(axis=1) > 0)
    if m.sum() < 4:
        return None
    Hm, inl = cv2.findHomography(k[m], PITCH_PTS[m], cv2.RANSAC, 1.5)
    if Hm is None or inl.sum() < 4:
        return None
    return make_proper(Hm / Hm[2, 2])


def project(Hm, pts):
    if Hm is None or len(pts) == 0:
        return np.zeros((0, 2))
    return cv2.perspectiveTransform(np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2), Hm).reshape(-1, 2)


def draw_pitch():
    img = np.full((int(PITCH_H * SCALE) + 40, int(PITCH_W * SCALE) + 40, 3), (60, 120, 45), np.uint8)
    p = lambda x, y: (int(x * SCALE) + 20, int(y * SCALE) + 20)
    L = (255, 255, 255)
    cv2.rectangle(img, p(0, 0), p(PITCH_W, PITCH_H), L, 2)
    cv2.line(img, p(PITCH_W / 2, 0), p(PITCH_W / 2, PITCH_H), L, 2)
    cv2.circle(img, p(PITCH_W / 2, PITCH_H / 2), int(9.15 * SCALE), L, 2)
    for x0, x1 in ((0, 16.5), (PITCH_W - 16.5, PITCH_W)):
        cv2.rectangle(img, p(x0, 13.84), p(x1, 54.16), L, 2)
    for x0, x1 in ((0, 5.5), (PITCH_W - 5.5, PITCH_W)):
        cv2.rectangle(img, p(x0, 24.84), p(x1, 43.16), L, 2)
    return img


def draw_map(base, xy, teams, ball_xy):
    img = base.copy()
    for (x, y), t in zip(xy, teams):
        if -5 <= x <= PITCH_W + 5 and -5 <= y <= PITCH_H + 5:
            cv2.circle(img, (int(x * SCALE) + 20, int(y * SCALE) + 20), 9, MAP_BGR[t], -1)
            cv2.circle(img, (int(x * SCALE) + 20, int(y * SCALE) + 20), 9, (0, 0, 0), 1)
    if ball_xy is not None:
        cv2.circle(img, (int(ball_xy[0] * SCALE) + 20, int(ball_xy[1] * SCALE) + 20), 6, (255, 255, 255), -1)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("out")
    ap.add_argument("--model", default="yolo11m.pt", help="detetor de pessoas")
    ap.add_argument("--ball-model", default=None, help="pesos do detetor de bola treinado")
    ap.add_argument("--pitch-model", default=None, help="pesos YOLO-pose dos pontos do campo")
    ap.add_argument("--map-out", default="mapa2d.mp4")
    ap.add_argument("--labels", default=None, help="CSV id,numero,nome para rotular jogadores à mão")
    ap.add_argument("--csv", default="tracks.csv", help="exporta posições por frame")
    ap.add_argument("--stats", default="stats.csv", help="exporta estatísticas por jogador")
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    model = YOLO(args.model)
    ball_model = YOLO(args.ball_model) if args.ball_model else model
    ball_cls = BALL_CLASS if args.ball_model else COCO_BALL
    pitch_model = YOLO(args.pitch_model) if args.pitch_model else None

    teams = fit_teams(args.video, model, args.device)
    info = sv.VideoInfo.from_video_path(args.video)
    tracker = ByteTrackTracker(frame_rate=info.fps, lost_track_buffer=150, track_activation_threshold=0.6,
                               minimum_consecutive_frames=5, minimum_iou_threshold=0.05, high_conf_det_threshold=0.5)

    box_ann = sv.EllipseAnnotator(color=sv.ColorPalette.from_hex(TEAM_COLORS), thickness=2)
    label_ann = sv.LabelAnnotator(color=sv.ColorPalette.from_hex(TEAM_COLORS), text_position=sv.Position.BOTTOM_CENTER)
    ball_ann = sv.TriangleAnnotator(color=sv.Color.WHITE, base=20, height=17, color_lookup=sv.ColorLookup.INDEX)

    # Passo 1: deteção, tracking, bola e homografia
    frames_data, ball_cands, homs, raw_teams, ocr_reads = [], [], [], [], []
    prev_h, stale, prev_frame = None, 0, None
    for frame in sv.get_video_frames_generator(args.video):
        det = detect_people(model, frame, args.device)
        rb = ball_model(frame, classes=[ball_cls], conf=0.05, imgsz=1280, device=args.device, verbose=False)[0]
        bd = sv.Detections.from_ultralytics(rb)
        ball_cands.append([[*xy, c] for xy, c in zip(bd.get_anchors_coordinates(sv.Position.CENTER), bd.confidence)
                           if on_grass(frame, *xy)])
        people = tracker.update(det)
        people = people[people.tracker_id >= 0]
        raw = [teams.predict(jersey_feature(frame, b)) for b in people.xyxy]
        raw_teams.append(raw)
        reads = []
        if len(ocr_reads) % 5 == 0:  # OCR só em planos próximos e a cada 5 frames
            for tid, b in zip(people.tracker_id, people.xyxy):
                if b[3] - b[1] >= 200:
                    n = read_number(frame, b)
                    if n:
                        reads.append((int(tid), n[0]))
        ocr_reads.append(reads)
        if pitch_model is not None:
            new_h = homography_for_frame(pitch_model, frame, args.device, None)
            if new_h is not None:
                prev_h, stale = new_h, 0
            elif prev_h is not None and prev_frame is not None and stale < 250:
                G = frame_motion(prev_frame, frame)  # propaga a calibração pelo movimento da câmara
                cand = prev_h @ G if G is not None else None
                if cand is not None and plausible(cand / cand[2, 2], frame.shape):
                    prev_h, stale = cand / cand[2, 2], stale + 1
                else:
                    prev_h = None  # corte de câmara ou plano sem campo: não mostrar posições erradas
            else:
                prev_h = None
        prev_frame = frame
        homs.append(prev_h)
        frames_data.append((frame.copy(), people))

    ball_pos = interpolate_ball(pick_ball(ball_cands))

    # Equipa por jogador (voto sobre todos os frames) e união de tracklets partidos
    votes = {}
    for (frame, people), raw in zip(frames_data, raw_teams):
        for tid, t in zip(people.tracker_id, raw):
            if t is not None:
                votes.setdefault(int(tid), []).append(t)
    team_by_id = {i: int(np.bincount(v, minlength=3).argmax()) for i, v in votes.items()}
    per_frame = [(p.tracker_id, p.xyxy, [team_by_id.get(int(t), 0) for t in p.tracker_id]) for _, p in frames_data]
    merged = stitch_ids(per_frame)
    renum = {}
    for _, p in frames_data:  # numeração final sequencial por ordem de aparição
        for t in p.tracker_id:
            renum.setdefault(merged[int(t)], len(renum) + 1)
    final_id = {i: renum[g] for i, g in merged.items()}
    group_votes = {}
    for i, v in votes.items():
        group_votes.setdefault(final_id[i], []).extend(v)
    team_of = {g: int(np.bincount(v, minlength=3).argmax()) for g, v in group_votes.items()}
    print("IDs antes/depois da união:", len(merged), "/", len(renum))

    # Números de camisola lidos por OCR (voto) + rótulos manuais
    num_votes = {}
    for reads in ocr_reads:
        for tid, n in reads:
            num_votes.setdefault(final_id[tid], []).append(n)
    names = {}
    for g, v in num_votes.items():
        n, c = np.unique(v, return_counts=True)
        if c.max() >= 2:
            names[g] = f"{int(n[c.argmax()])}"
    if args.labels:
        for row in pd.read_csv(args.labels, dtype=str).fillna("").itertuples():
            names[int(row.id)] = f"{row.numero} {row.nome}".strip()
    print("Números lidos por OCR:", {k: v for k, v in names.items()} or "nenhum (números ilegíveis neste vídeo)")
    rows, track_xy = [], {}

    # Passo 2: anotar e escrever
    base = draw_pitch() if pitch_model is not None else None
    map_sink = None
    with sv.VideoSink(args.out, info) as sink:
        for fi, ((frame, people), (bx, by), Hm) in enumerate(zip(frames_data, ball_pos, homs)):
            people.tracker_id = np.array([final_id[int(t)] for t in people.tracker_id], dtype=int)
            cls = np.array([team_of.get(int(t), 0) for t in people.tracker_id], dtype=int)
            people.class_id = cls
            frame = box_ann.annotate(frame, people)
            frame = label_ann.annotate(frame, people, [names.get(int(t), f"#{t}") for t in people.tracker_id])
            has_ball = not np.isnan(bx)
            if has_ball:
                frame = ball_ann.annotate(frame, sv.Detections(xyxy=np.array([[bx - 6, by - 6, bx + 6, by + 6]])))
            sink.write_frame(frame)

            if base is not None:
                feet = people.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
                xy = project(Hm, feet)
                for k, (tid, t) in enumerate(zip(people.tracker_id, cls)):
                    rows.append(dict(frame=fi, id=int(tid), team=int(t), x_px=feet[k][0], y_px=feet[k][1],
                                     x_m=xy[k][0] if len(xy) else np.nan, y_m=xy[k][1] if len(xy) else np.nan))
                bxy = project(Hm, [[bx, by]])[0] if has_ball and Hm is not None else None
                mimg = draw_map(base, xy, cls, bxy)
                if map_sink is None:
                    map_sink = sv.VideoSink(args.map_out, sv.VideoInfo(width=mimg.shape[1], height=mimg.shape[0], fps=info.fps))
                    map_sink.__enter__()
                map_sink.write_frame(mimg)
    if map_sink is not None:
        map_sink.__exit__(None, None, None)
        print("Mapa 2D guardado em", args.map_out)
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(args.csv, index=False)
        stats = []
        for pid, g in df.groupby("id"):
            g = g.dropna(subset=["x_m"]).sort_values("frame")
            dist = 0.0
            for a, b in zip(g.itertuples(), list(g.itertuples())[1:]):
                dt = (b.frame - a.frame) / info.fps
                d = float(np.hypot(b.x_m - a.x_m, b.y_m - a.y_m))
                if 0 < dt <= 0.5 and d / dt <= 12:  # ignora saltos impossíveis (>12 m/s)
                    dist += d
            stats.append(dict(id=pid, nome=names.get(pid, ""), equipa=int(df[df.id == pid].team.mode()[0]),
                              frames=len(df[df.id == pid]), frames_no_mapa=len(g), distancia_m=round(dist, 1)))
        pd.DataFrame(stats).sort_values("frames", ascending=False).to_csv(args.stats, index=False)
        print("Dados em", args.csv, "e estatísticas em", args.stats)
    print("Vídeo guardado em", args.out)


if __name__ == "__main__":
    main()
