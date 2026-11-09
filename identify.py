"""Identificação de jogadores por OCR (Vision da Apple) limitada ao elenco real do jogo."""
import difflib
import json
import re

import cv2
import numpy as np

import jersey_ocr as J

ROSTER = json.load(open("match_info.json"))["elenco"]
KEY = {1: "POR", 0: "NAP"}


def _clean(t):
    return re.sub(r"[^A-Z0-9 ]", "", t.upper()).strip()


def identify(frame, box, team):
    """Devolve (numero, nome, confianca) se conseguir ler número/nome do elenco da equipa; senão None."""
    if team not in KEY:
        return None
    roster = ROSTER[KEY[team]]
    x1, y1, x2, y2 = map(int, box)
    h = y2 - y1
    crop = frame[max(0, y1): y2, max(0, x1): x2]
    if crop.size == 0 or h < 60:
        return None
    sc = max(1.0, 520 / crop.shape[0])
    big = cv2.resize(crop, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC)
    best = None
    for img in (big, cv2.cvtColor(cv2.cvtColor(big, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)):
        for text, conf in J._recognize(img):
            t = _clean(text)
            if not t or conf < 0.45:
                continue
            for tok in t.split():
                if tok.isdigit() and tok.lstrip("0") in roster and len(tok) <= 2:
                    cand = (int(tok), roster[tok.lstrip("0")], conf)
                    if best is None or conf > best[2]:
                        best = cand
            if len(t) >= 4 and not t.isdigit():
                for num, name in roster.items():
                    last = _clean(name).split()[-1]
                    if len(last) >= 4 and difflib.SequenceMatcher(None, t.split()[-1], last).ratio() >= 0.85:
                        cand = (int(num), name, conf * 0.9)
                        if best is None or cand[2] > best[2]:
                            best = cand
    return best
