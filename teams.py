"""Classificação de equipas: 2 clusters por cor da camisola (sem relva) + outliers (árbitro/guarda-redes)."""
import cv2
import numpy as np
from sklearn.cluster import KMeans


def jersey_feature(frame, box):
    x1, y1, x2, y2 = map(int, box)
    h, w = y2 - y1, x2 - x1
    if h < 25 or w < 8:
        return None
    crop = frame[max(y1, 0) + int(h * 0.18): y1 + int(h * 0.55), x1 + int(w * 0.2): x2 - int(w * 0.2)]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).reshape(-1, 3)
    grass = (hsv[:, 0] > 30) & (hsv[:, 0] < 90) & (hsv[:, 1] > 50)
    keep = lab[~grass]
    if len(keep) < 0.2 * len(lab):
        return None
    return np.median(keep, axis=0)


class TeamClassifier:
    def __init__(self, outlier_pct=86, outlier_mult=1.15):
        self.pct, self.mult = outlier_pct, outlier_mult

    def fit(self, feats):
        X = np.array(feats)
        self.km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(X)
        d = np.linalg.norm(X - self.km.cluster_centers_[self.km.labels_], axis=1)
        self.thr = np.percentile(d, self.pct) * self.mult
        return self

    def predict(self, f):
        if f is None:
            return None
        c = self.km.cluster_centers_
        d = np.linalg.norm(c - f, axis=1)
        return 2 if d.min() > self.thr else int(d.argmin())
