"""Atribuição AUTOMÁTICA de identidades (sem intervenção manual) a partir dos dados do tracking.

Fontes, por ordem de fiabilidade:
 1. Guarda-redes e árbitro: papel + posição (junto da baliza / fora das equipas) + metade do jogo.
 2. Números/nomes lidos por OCR em planos próximos (votos já agregados por jogador em *_nomes.json).
 3. Propagação: o mesmo jogador é ligado entre blocos consecutivos (continuidade de posição) e herda o nome.
Em planos largos de 1080p o número da camisola tem ~12 px e não é legível: esses jogadores ficam só com #ID.
Saída: painel/data/names.json  {"src:t0:id": "10 Veiga"}
"""
import glob
import json
import os
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

import halves

OUT = "painel/data/names.json"
W = 105.0


def second_half_start():
    d = pd.read_csv("timeline.csv").dropna(subset=["clock"])
    d = d[(d.t > 2800) & (d.clock > 2700)]
    return float(d.t.iloc[0] - (d.clock.iloc[0] - 2700)) if len(d) else 3100.0


class UF:
    def __init__(self):
        self.p = {}

    def find(self, a):
        self.p.setdefault(a, a)
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def read(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def chunks(src):
    out = []
    for f in sorted(glob.glob(f"{src}/c*_tracks.csv")):
        tag = os.path.basename(f).split("_")[0]
        out.append((int(tag[1:]), f))
    return out


def main():
    info = json.load(open("match_info.json"))
    names, source = {}, {}

    def put(key, label, why):
        # fiabilidade: golo(4) > ocr(3) > papel(2) > propagado(1)
        rank = {"golo": 4, "ocr": 3, "papel": 2, "propagado": 1}[why]
        if key not in names or rank > source[key]:
            names[key], source[key] = label, rank

    # ---------- 1) guarda-redes e árbitro por papel/posição ----------
    gk = {"POR": "99 Diogo Costa", "NAP": "1 Meret"}
    for src in ["destaques_out"]:
        for t0, f in chunks(src):
            tr = read(f)
            if tr.empty:
                continue
            refs = []   # candidatos a árbitro (equipa 2, dentro do campo, não é guarda-redes)
            for pid, g in tr[(tr.team == 2)].groupby("id"):
                m = g.dropna(subset=["x_m"])
                if len(g) < 25 or len(m) < 25:
                    continue
                key = f"{src}:{t0}:{pid}"
                near = ((np.minimum(m.x_m, W - m.x_m) < 14) & ((m.y_m - 34).abs() < 16)).mean()
                if near >= 0.5:
                    left = (m.x_m < W / 2).mean() > 0.5
                    porto_left = halves.first_half(t0 + g.t_s.mean())   # 1.ª parte: Porto defende a esquerda
                    team = "POR" if left == porto_left else "NAP"
                    put(key, "GK " + gk[team], "papel")
                    continue
                inside = ((m.x_m.between(-3, W + 3)) & (m.y_m.between(-3, 71))).mean()
                if inside >= 0.6:
                    refs.append((len(g), key, set(g.frame)))
            chosen = []   # só um árbitro em campo de cada vez: IDs sem sobreposição temporal
            for n_, key, frs in sorted(refs, key=lambda x: -x[0]):
                if all(not (frs & c) for c in chosen):
                    chosen.append(frs); put(key, "REF", "papel")

    # equipa observada (mais frequente) de cada jogador, para validar leituras
    team_obs = {}
    for src in ["destaques_out"]:
        for t0, f in chunks(src):
            tr = read(f)
            if tr.empty:
                continue
            for pid, g in tr.groupby("id"):
                if len(g) >= 10:
                    team_obs[(src, t0, int(pid))] = int(g.team.mode().iloc[0])

    # ---------- 2) OCR + 3) propagação entre blocos consecutivos ----------
    for src in ["destaques_out"]:
        cs = chunks(src)
        uf = UF()
        ocr = {}
        for t0, f in cs:
            nf = f.replace("_tracks.csv", "_nomes.json")
            if os.path.exists(nf):
                for pid, v in json.load(open(nf)).items():
                    ocr[f"{t0}:{pid}"] = v
        # ligar o fim de um bloco ao início do seguinte (plano contínuo)
        for (ta, fa), (tb, fb) in zip(cs, cs[1:]):
            A, B = read(fa), read(fb)
            if A.empty or B.empty:
                continue
            fa_last = A.frame.max(); fb_first = B.frame.min()
            a = A[A.frame >= fa_last - 1].groupby("id").last().reset_index()
            b = B[B.frame <= fb_first + 1].groupby("id").first().reset_index()
            a = a[a.team < 2]; b = b[b.team < 2]
            if a.empty or b.empty:
                continue
            cost = np.hypot(a.x_px.values[:, None] - b.x_px.values[None, :], a.y_px.values[:, None] - b.y_px.values[None, :])
            cost[a.team.values[:, None] != b.team.values[None, :]] = 1e6
            ri, ci = linear_sum_assignment(cost)
            for i, j in zip(ri, ci):
                if cost[i, j] < 40:
                    uf.union(f"{ta}:{int(a.id.values[i])}", f"{tb}:{int(b.id.values[j])}")
        groups = defaultdict(list)
        for t0, f in cs:
            tr = read(f)
            if tr.empty:
                continue
            for pid in tr.id.unique():
                groups[uf.find(f"{t0}:{int(pid)}")].append(f"{t0}:{int(pid)}")
        for root, members in groups.items():
            votes = Counter()
            for mbr in members:
                if mbr in ocr:
                    v = ocr[mbr]
                    t0_, pid_ = mbr.split(":")
                    obs = team_obs.get((src, int(t0_), int(pid_)))
                    if obs is not None and obs != (1 if v["equipa"] == "POR" else 0):
                        continue  # equipa observada ≠ equipa do nome: leitura rejeitada
                    votes[(v["num"], v["nome"], v["equipa"])] += v["leituras"]
            if not votes:
                continue
            (num, nome, eq), n = votes.most_common(1)[0]
            for mbr in members:
                put(f"{src}:{mbr}", f"{num} {nome}", "ocr" if mbr in ocr else "propagado")

    json.dump(names, open(OUT, "w"), ensure_ascii=False)
    por = Counter(v.split()[0] if v[0].isdigit() else v.split()[0] for v in names.values())
    print(f"identidades atribuídas: {len(names)} (ocr={sum(1 for k in source if source[k]==3)}, "
          f"papel={sum(1 for k in source if source[k]==2)}, propagadas={sum(1 for k in source if source[k]==1)}, "
          f"golo={sum(1 for k in source if source[k]==4)})")


if __name__ == "__main__":
    main()
