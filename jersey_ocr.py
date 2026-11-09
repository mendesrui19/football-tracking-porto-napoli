"""OCR do número da camisola com o Vision da Apple (local, sem descarregar modelos)."""
import re

import cv2
import numpy as np
import Quartz
import Vision
from Foundation import NSData


def _recognize(img_bgr):
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        return []
    data = NSData.dataWithBytes_length_(buf.tobytes(), len(buf))
    req = Vision.VNRecognizeTextRequest.alloc().init()
    req.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    req.setUsesLanguageCorrection_(False)
    handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(data, None)
    handler.performRequests_error_([req], None)
    out = []
    for r in req.results() or []:
        cand = r.topCandidates_(1)
        if cand:
            out.append((str(cand[0].string()), float(cand[0].confidence())))
    return out


def read_number(frame, box):
    """Tenta ler o número (1-99) nas costas/peito. Devolve (numero, confianca) ou None."""
    x1, y1, x2, y2 = map(int, box)
    h, w = y2 - y1, x2 - x1
    if h < 70 or w < 20:  # demasiado pequeno para ler
        return None
    crop = frame[max(0, y1 + int(h * 0.12)): y1 + int(h * 0.6), max(0, x1): x2]
    if crop.size == 0:
        return None
    scale = max(1.0, 160 / crop.shape[0])
    crop = cv2.resize(crop, None, fx=scale * 2, fy=scale * 2, interpolation=cv2.INTER_CUBIC)
    best = None
    for variant in (crop, cv2.cvtColor(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)):
        for text, conf in _recognize(variant):
            t = re.sub(r"\D", "", text)
            if t and len(t) <= 2 and 1 <= int(t) <= 99 and conf > 0.4:
                if best is None or conf > best[1]:
                    best = (int(t), conf)
    return best
