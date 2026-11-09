"""Seguimento persistente em coordenadas do campo: 11 'lugares' por equipa + árbitro + bola.

- Cada lugar guarda a posição filtrada (alfa-beta) do jogador; quem não é visto mantém a posição e acompanha,
  atenuado, o movimento coletivo dos colegas visíveis (estimado, marcado como não visto).
- Salto comum de calibração (todos os jogadores deslocados ao mesmo tempo) é removido.
- Linha de fora de jogo estável: equipa atacante com histerese, defesa mais recuado dos lugares,
  limitada pela bola, suavizada no tempo e só mostrada com a bola no meio-campo atacante.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment

W, H = 105.0, 68.0
MAX_OUT = 10  # jogadores de campo por equipa (o 11.º é o guarda-redes)
PORTO = 1
VMAX = 9.0     # m/s: velocidade máxima plausível de um jogador
ALPHA, BETA = 0.42, 0.10   # ganhos do filtro a 25 fps


class Slot:
    def __init__(self, sid, team, x, y, t):
        self.sid, self.team, self.x, self.y, self.last, self.first = sid, team, x, y, t, t
        self.vx = self.vy = 0.0
        self.seen = True
        self.dx, self.dy = x, y   # posição mostrada (desliza até à posição real, sem teletransportes)


class SlotTracker:
    def __init__(self, porto_left):
        self.slots = {0: [], 1: []}
        self.gk = {}
        self.ref = None
        self.next_id = {0: 0, 1: 0}
        self.ball = None
        self.porto_left = porto_left
        self.prev_t = None
        # fora de jogo
        self.att = None
        self.cand, self.cand_t = None, 0.0
        self.line = None

    # ---------------------------------------------------------------- utilitários
    @staticmethod
    def _gate(unseen):
        return min(4.0 + 6.0 * unseen, 45.0)

    @staticmethod
    def _predict(s, t):
        dt = min(t - s.last, 1.0)
        return s.x + s.vx * dt, s.y + s.vy * dt

    def _all_field_slots(self):
        return self.slots[0] + self.slots[1]

    def _common_shift(self, dets, t):
        """Deslocamento comum (salto de calibração) entre este frame e as posições previstas dos lugares."""
        res = []
        used = set()
        for tm, x, y in dets:
            if tm not in (0, 1):
                continue
            best = None
            for j, s in enumerate(self.slots[tm]):
                if (tm, j) in used or t - s.last > 0.3:
                    continue
                px, py = self._predict(s, t)
                d = np.hypot(x - px, y - py)
                if d < 3.0 and (best is None or d < best[0]):
                    best = (d, j, x - px, y - py)
            if best:
                used.add((tm, best[1])); res.append((best[2], best[3]))
        if len(res) < 4:
            return 0.0, 0.0
        r = np.array(res)
        mx, my = float(np.median(r[:, 0])), float(np.median(r[:, 1]))
        n = np.hypot(mx, my)
        lim = 0.22 * max(min(t - (self.prev_t or t), 0.2) / 0.04, 1.0)  # movimento coletivo plausível por frame
        if n <= lim:
            return 0.0, 0.0
        k = (n - lim) / n
        return mx * k, my * k

    def _match(self, team, dets, t):
        slots = self.slots[team]
        used = set()
        keep = []
        for d in sorted(dets, key=lambda d: d[0]):
            if all(np.hypot(d[0] - k[0], d[1] - k[1]) > 1.2 for k in keep):
                keep.append(d)
        dets = keep
        if slots and dets:
            C = np.full((len(dets), len(slots)), 1e6)
            for i, d in enumerate(dets):
                for j, s in enumerate(slots):
                    px, py = self._predict(s, t)
                    dist = np.hypot(d[0] - px, d[1] - py)
                    if dist <= self._gate(t - s.last):
                        C[i, j] = dist
            ri, ci = linear_sum_assignment(C)
            for i, j in zip(ri, ci):
                if C[i, j] < 1e5:
                    s = slots[j]
                    dt = max(t - s.last, 1e-3)
                    px, py = self._predict(s, t)
                    rx, ry = dets[i][0] - px, dets[i][1] - py
                    if dt <= 0.25:  # filtro alfa-beta (suaviza o ruído da deteção)
                        a, b = ALPHA, BETA
                        s.x, s.y = px + a * rx, py + a * ry
                        s.vx += b * rx / dt; s.vy += b * ry / dt
                        v = np.hypot(s.vx, s.vy)
                        if v > VMAX:
                            s.vx *= VMAX / v; s.vy *= VMAX / v
                    else:
                        s.x, s.y, s.vx, s.vy = dets[i][0], dets[i][1], 0.0, 0.0
                    s.last, s.seen = t, True
                    used.add(j)
                    dets[i] = None
        for d in dets:
            if d is None:
                continue
            if len(slots) < MAX_OUT:
                s = Slot(self.next_id[team], team, d[0], d[1], t)
                self.next_id[team] += 1
                slots.append(s); used.add(len(slots) - 1)
            else:
                cand = [(t - s.last, j) for j, s in enumerate(slots) if j not in used and t - s.last > 2.0]
                if cand:
                    _, j = max(cand)
                    s = slots[j]; s.x, s.y, s.last, s.seen = d[0], d[1], t, True; s.vx = s.vy = 0.0
                    used.add(j)

    # ---------------------------------------------------------------- passo principal
    def step(self, t, calibrated, dets, ball_xy):
        """dets: [(team, x, y)]; ball_xy: (x, y) ou None. Devolve (lugares, bola, portador, linha_fj)."""
        before = {id(s): (s.x, s.y) for s in self._all_field_slots()}
        for lst in self.slots.values():
            for s in lst:
                s.seen = False
        for s in list(self.gk.values()) + ([self.ref] if self.ref else []):
            s.seen = False
        drifted = set()
        if calibrated:
            sx, sy = self._common_shift(dets, t)           # remove o salto comum da calibração
            dets = [(tm, x - sx, y - sy) for tm, x, y in dets]
            if ball_xy is not None:
                ball_xy = (ball_xy[0] - sx, ball_xy[1] - sy)
            for k in (0, 1):
                self._match(k, [(x, y) for tm, x, y in dets if tm == k], t)
            porto_left = self.porto_left(t)
            for tm, x, y in dets:
                if tm != 2 or not (-3 <= x <= W + 3 and -3 <= y <= H + 3):   # fora do campo (banco, staff): ignorar
                    continue
                if min(x, W - x) < 16 and abs(y - H / 2) < 20:
                    side = "L" if x < W / 2 else "R"
                    s = self.gk.get(side)
                    team = (1 if porto_left else 0) if side == "L" else (0 if porto_left else 1)
                    if s is None:
                        self.gk[side] = Slot("GK" + side, team, x, y, t)
                    else:
                        s.x, s.y, s.last, s.seen, s.team = 0.5 * s.x + 0.5 * x, 0.5 * s.y + 0.5 * y, t, True, team
                else:
                    if self.ref is None:
                        self.ref = Slot("REF", 2, x, y, t)
                    elif np.hypot(x - self.ref.x, y - self.ref.y) < self._gate(t - self.ref.last) * 1.5:
                        self.ref.x, self.ref.y, self.ref.last, self.ref.seen = 0.5 * self.ref.x + 0.5 * x, 0.5 * self.ref.y + 0.5 * y, t, True
            # movimento coletivo estimado: quem não é visto acompanha (atenuado) os colegas visíveis
            for k in (0, 1):
                moved = [(s.x - before[id(s)][0], s.y - before[id(s)][1]) for s in self.slots[k]
                         if s.seen and id(s) in before and t - 0 < 1e9]
                if len(moved) >= 3:
                    mdx, mdy = np.median([m[0] for m in moved]), np.median([m[1] for m in moved])
                    drifted.add(k)
                    for s in self.slots[k]:
                        if not s.seen and t - s.last < 8.0:
                            w = 0.55 * max(0.0, 1.0 - (t - s.last) / 8.0)
                            s.x = min(max(s.x + mdx * w, -1.0), W + 1.0)
                            s.y = min(max(s.y + mdy * w, -1.0), H + 1.0)
        if ball_xy is not None and calibrated and -6 <= ball_xy[0] <= W + 6 and -6 <= ball_xy[1] <= H + 6:
            nb = [min(max(ball_xy[0], 0.0), W), min(max(ball_xy[1], 0.0), H), t]
            if self.ball is not None and 0 < t - self.ball[2] <= 0.5:   # bola no ar projetada no chão exagera a velocidade: limita a 38 m/s
                dt_b = t - self.ball[2]
                d_b = float(np.hypot(nb[0] - self.ball[0], nb[1] - self.ball[1]))
                lim = 38.0 * dt_b
                if d_b > lim:
                    k_ = lim / d_b
                    nb[0] = self.ball[0] + (nb[0] - self.ball[0]) * k_
                    nb[1] = self.ball[1] + (nb[1] - self.ball[1]) * k_
            if self.ball is not None and 0 < t - self.ball[2] < 0.3:
                dbx, dby = nb[0] - self.ball[0], nb[1] - self.ball[1]
                if np.hypot(dbx, dby) < 2.0:   # as equipas sobem/descem com a bola: quem não é visto acompanha-a (atenuado)
                    for k in (0, 1):
                        if k in drifted:
                            continue
                        for s_ in self.slots[k]:
                            if not s_.seen and t - s_.last < 10.0:
                                w_ = 0.30 * max(0.0, 1.0 - (t - s_.last) / 10.0)
                                s_.x = min(max(s_.x + dbx * w_, -1.0), W + 1.0)
                                s_.y = min(max(s_.y + dby * w_ * 0.6, -1.0), H + 1.0)
            self.ball = nb
        out = list(self._all_field_slots()) + list(self.gk.values()) + ([self.ref] if self.ref else [])
        dt_ = 0.04 if self.prev_t is None else min(max(t - self.prev_t, 0.0), 1.0)
        for s_ in out:   # deslize suave: no máximo 14 m/s em direção à posição real
            ex, ey = s_.x - s_.dx, s_.y - s_.dy
            d_ = float(np.hypot(ex, ey))
            if d_ > 45:
                s_.dx, s_.dy = s_.x, s_.y
            else:
                mx_ = 14.0 * max(dt_, 0.04)
                if d_ > mx_:
                    s_.dx += ex * mx_ / d_; s_.dy += ey * mx_ / d_
                else:
                    s_.dx, s_.dy = s_.x, s_.y
        ball = carrier = None
        if self.ball:
            age = t - self.ball[2]
            ball = [self.ball[0], self.ball[1], age]
            best = None
            for s in out:
                if s.team == 2 or t - s.last > 1.0:
                    continue
                d = np.hypot(s.x - self.ball[0], s.y - self.ball[1])
                if d < 3.0 and age < 1.0 and (best is None or d < best[0]):
                    best = (d, s.sid if isinstance(s.sid, str) else f"{s.team}_{s.sid}", s.team)
            carrier = best[1] if best else None
            self._update_att(t, best[2] if best else None)
        line = self._offside(t, ball) if calibrated else None
        self.prev_t = t
        return out, ball, carrier, line

    # ---------------------------------------------------------------- fora de jogo coerente
    def _update_att(self, t, carrier_team):
        if carrier_team not in (0, 1):
            return
        if self.att is None:
            self.att = carrier_team
        elif carrier_team != self.att:
            if self.cand == carrier_team:
                if t - self.cand_t >= 0.8:   # histerese: só muda de equipa atacante após 0,8 s de posse
                    self.att, self.cand = carrier_team, None
            else:
                self.cand, self.cand_t = carrier_team, t
        else:
            self.cand = None

    def _offside(self, t, ball):
        if self.att is None or ball is None or ball[2] > 2.0:
            return None
        first_half = self.porto_left(t)
        attacks_right = (self.att == PORTO) == first_half
        defs = [s for s in self.slots[1 - self.att] if t - s.last < 4.0]
        if len(defs) < 4:
            return None
        D = np.array([(W - s.x) if attacks_right else s.x for s in defs])
        ball_d = (W - ball[0]) if attacks_right else ball[0]
        if ball_d > W / 2 + 6:      # bola longe da baliza adversária: sem linha
            self.line = None
            return None
        line_d = min(float(D.min()), ball_d)
        if line_d < 10.0 or line_d > W - 10.0:   # colada à linha de golo não informa nada
            return None
        target = (W - line_d) if attacks_right else line_d
        dt = 0.04 if self.prev_t is None else min(max(t - self.prev_t, 0.01), 0.3)
        if self.line is not None and abs(target - self.line) > 25:  # mudou o sentido do ataque: esconde um instante
            self.line = None
            return None
        if self.line is None:
            self.line = target
        else:  # suavização + limite de velocidade (6 m/s)
            step = np.clip(0.18 * (target - self.line), -6.0 * dt, 6.0 * dt)
            self.line += step
        return round(float(self.line), 1)


def sid_of(s):
    return s.sid if isinstance(s.sid, str) else f"{s.team}_{s.sid}"
