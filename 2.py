"""
CENG 391 - Fixed-camera parking occupancy system (2.py)

Task: identify parking spaces, determine their occupancy, track occupancy changes over time and
present availability statistics; compare geometric space-based classification with vehicle
detection; add robustness to shadows, weather, partial occlusion and camera viewpoint changes.

Pipeline
  1. Parking-space identification
       - installation map: space polygons annotated once for the camera (PKLot XML)
       - viewpoint compensation: SIFT keypoints + RANSAC homography (reference frame -> current
         frame) re-projects the map when the camera moves (falls back to the last good estimate)
       - automatic discovery: persistent YOLO vehicle boxes clustered over a day, scored against
         the annotated map
  2. Occupancy classification per space - four methods compared
       A  Canny edge density + calibrated threshold      (geometric space-based baseline, as 1.py)
       B  Space features + SVM: every ROI is resampled to 48x48 (scale normalisation) and described
          with illumination-robust features: LBP texture, Sobel gradients, auto/fixed Canny edges,
          FFT high-frequency energy, Shi-Tomasi corners, histogram entropy, HSV colour, HOG
       C  YOLOv8 vehicle detection on aspect-ratio-restored frames plus a 2x2 tiled pass for small,
          distant cars; one-to-one box<->space assignment by IoU, calibrated IoU threshold
       D  Hybrid: B's features + C's detection scores in one SVM
  3. Temporal tracking: per-space two-state HMM forward filter. The switch probability grows with
     the time gap between frames, so dense frames are smoothed strongly and sparse snapshots
     weakly; state flips are logged as occupancy change events
  4. Availability statistics: free spaces over time, busiest hour, per-space occupancy rate,
     turnover and estimated occupied time (CSV + dashboard figures)
  5. Robustness experiments: weather (Sunny/Cloudy/Rainy sequences), synthetic cast shadows,
     synthetic partial occlusion, synthetic camera shake with/without registration, the real
     UFPR04 camera move between December and January, and an unseen camera (UFPR05)

Data (create with: python prepare_dataset.py)
  PKLot/<Camera>/<Weather>/<date>/*.jpg+xml   test sequences, one day each
  PKLot_calibration/<Camera>/<date>/...       training + threshold calibration (other dates only)
  PKLot_unseen/<Camera>/<date>/...            camera never used for training

Run from the Project folder:  python 2.py        (outputs are written to ./outputs_2)
"""
import csv
import glob
import os
import time
import warnings
import zlib
from datetime import datetime

import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from shapely.geometry import Polygon, box as shapely_box
from skimage.feature import hog, local_binary_pattern
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from ultralytics import YOLO
except ImportError:  # methods C and D are skipped without ultralytics
    YOLO = None

# ================================ configuration ================================
DATASET_ROOT = "PKLot"
CALIB_ROOT = "PKLot_calibration"
UNSEEN_ROOT = "PKLot_unseen"
OUTPUT_ROOT = "outputs_2"

YOLO_WEIGHTS = "yolov8n.pt"
VEHICLE_CLASSES = [2, 3, 5, 7]          # COCO: car, motorcycle, bus, truck
YOLO_CONF = 0.15
DETECTION_SIZE = (1280, 720)            # PKLot frames are 1280x720; the Roboflow export stretched them to 640x640
DETECTION_TILES = 2                     # extra NxN tiled pass for small cars (1 = whole frame only)
TILE_OVERLAP, TILE_IMGSZ = 0.2, 960

PATCH_SIZE = 48                         # every parking ROI is resampled to 48x48 (scale normalisation)
TRACK_MEAN_DWELL_MIN = 60.0             # HMM tracker: expected minutes between state changes of a space
ROBUSTNESS_FRAME_STEP = 2               # robustness experiments use every 2nd frame of each sequence
SEED = 391

METHODS = {
    "A": "Canny edge density",
    "B": "Space features + SVM",
    "C": "YOLOv8 detection",
    "D": "Hybrid (B + C)",
}

# Reference data-viz palette (light chart surface)
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a", "D": "#eda100"}
HEAT_COLORS = ["#f0efec", "#86b6ef", "#eb6834", "#6250d6"]  # free ok, occupied ok, false occupied, missed
HEAT_NAMES = ["Free (correct)", "Occupied (correct)", "False occupied", "Missed occupied"]
GOOD_BGR, CRITICAL_BGR, WARNING_BGR, BLUE_BGR = (12, 163, 12), (59, 59, 208), (25, 178, 250), (214, 120, 42)

plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "axes.titlesize": 11,
                     "axes.labelcolor": INK_2, "text.color": INK})
warnings.filterwarnings("ignore", message=".*y_pred contains classes not in y_true.*")


# ================================ data loading =================================
def imread(path):
    """cv2.imread cannot open non-ASCII Windows paths (e.g. 'Sinif'); decode from bytes instead."""
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


def imwrite(path, img):
    ok, buf = cv2.imencode(os.path.splitext(path)[1], img)
    if ok:
        buf.tofile(path)


def parse_pklot_xml(xml_path):
    """Space polygons (contour, or rotatedRect fallback) and occupancy labels of one PKLot XML."""
    import xml.etree.ElementTree as ET
    spaces = []
    for space in ET.parse(xml_path).getroot().findall("space"):
        if "occupied" not in space.attrib:
            continue
        pts = [[int(float(p.attrib["x"])), int(float(p.attrib["y"]))]
               for p in (space.find("contour") if space.find("contour") is not None else [])]
        if len(pts) < 3 and space.find("rotatedRect") is not None:
            rr = space.find("rotatedRect")
            pts = np.int32(cv2.boxPoints((
                (float(rr.find("center").attrib["x"]), float(rr.find("center").attrib["y"])),
                (float(rr.find("size").attrib["w"]), float(rr.find("size").attrib["h"])),
                float(rr.find("angle").attrib["d"])))).tolist()
        if len(pts) >= 3:
            spaces.append((np.array(pts, dtype=np.int32), int(space.attrib["occupied"])))
    return spaces


def load_frames(folder):
    """All annotated frames under folder, sorted by capture time."""
    frames = []
    for xml_path in glob.glob(os.path.join(folder, "**", "*.xml"), recursive=True):
        img_path = os.path.splitext(xml_path)[0] + ".jpg"
        spaces = parse_pklot_xml(xml_path)
        if not os.path.exists(img_path) or not spaces:
            continue
        frames.append({
            "img_path": img_path,
            "time": datetime.strptime(os.path.basename(xml_path)[:19], "%Y-%m-%d_%H_%M_%S"),
            "polys": [p for p, _ in spaces],
            "labels": np.array([o for _, o in spaces], dtype=np.int32),
        })
    return sorted(frames, key=lambda f: f["time"])


def load_sequences(root, default_condition=None):
    """root/<Camera>/<Condition>/... -> one sequence per folder (or root/<Camera>/... if default_condition)."""
    sequences = []
    for cam_dir in sorted(d for d in glob.glob(os.path.join(root, "*")) if os.path.isdir(d)):
        cond_dirs = [cam_dir] if default_condition else \
            sorted(d for d in glob.glob(os.path.join(cam_dir, "*")) if os.path.isdir(d))
        for cond_dir in cond_dirs:
            frames = load_frames(cond_dir)
            if frames:
                sequences.append({"camera": os.path.basename(cam_dir),
                                  "condition": default_condition or os.path.basename(cond_dir),
                                  "frames": frames})
    return sequences


# =============================== geometry helpers ==============================
def crop_box(poly, shape):
    h, w = shape[:2]
    x, y, bw, bh = cv2.boundingRect(np.asarray(poly, dtype=np.int32))
    return max(x, 0), max(y, 0), min(x + bw, w), min(y + bh, h)


def transform_polys(polys, M):
    """Apply a 3x3 homography / affine matrix to every polygon."""
    return [np.round(cv2.perspectiveTransform(np.asarray(p, np.float32).reshape(-1, 1, 2), M)
                     .reshape(-1, 2)).astype(np.int32) for p in polys]


def to_shapely(poly):
    shp = Polygon(np.asarray(poly, dtype=float))
    return shp if shp.is_valid else shp.buffer(0)


def bounds_of(polys):
    return np.array([[p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()] for p in polys], float)


def overlapping_pairs(a, b):
    """Index pairs whose bounding boxes (x0, y0, x1, y1) overlap."""
    if len(a) == 0 or len(b) == 0:
        return []
    ov = ((a[:, None, 0] < b[None, :, 2]) & (a[:, None, 2] > b[None, :, 0]) &
          (a[:, None, 1] < b[None, :, 3]) & (a[:, None, 3] > b[None, :, 1]))
    return list(zip(*np.nonzero(ov)))


def shapely_iou(a, b):
    inter = a.intersection(b).area
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def greedy_assignment(score):
    """One-to-one assignment, highest score first. Returns column index per row (-1 if none)."""
    idx = -np.ones(score.shape[0], dtype=int)
    used = np.zeros(score.shape[1], dtype=bool)
    for flat in np.argsort(-score, axis=None):
        i, j = divmod(int(flat), score.shape[1])
        if score[i, j] <= 0:
            break
        if idx[i] < 0 and not used[j]:
            idx[i], used[j] = j, True
    return idx


def match_polys(src, dst, min_iou=0.3):
    """Greedy one-to-one polygon matching. Returns (dst index or -1, IoU) for every src polygon."""
    iou = np.zeros((len(src), len(dst)))
    S, D = [to_shapely(p) for p in src], [to_shapely(p) for p in dst]
    for i, j in overlapping_pairs(bounds_of(src), bounds_of(dst)):
        iou[i, j] = shapely_iou(S[i], D[j])
    iou[iou < min_iou] = 0
    idx = greedy_assignment(iou) if len(dst) else -np.ones(len(src), dtype=int)
    best = np.array([iou[i, j] if j >= 0 else 0.0 for i, j in enumerate(idx)])
    return idx, best


def box_to_poly(b):
    x0, y0, x1, y1 = b
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.int32)


# ============================== image preprocessing ============================
CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def image_view(img):
    """Illumination normalisation (CLAHE on the L channel) + maps shared by the space-based methods."""
    l, a, b = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2LAB))
    enhanced = cv2.cvtColor(cv2.merge((CLAHE.apply(l), a, b)), cv2.COLOR_LAB2BGR)
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    return {
        "gray": gray,
        "edges": cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 50, 150),  # same edge map as 1.py
        "hsv": cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV),
        "gray_mean": float(gray.mean()) + 1e-6,
    }


# ===================== method A: Canny edge density (baseline) =================
def canny_densities(view, polys):
    out = np.zeros(len(polys), dtype=np.float32)
    for i, p in enumerate(polys):
        x0, y0, x1, y1 = crop_box(p, view["edges"].shape)
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue
        mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.fillPoly(mask, [np.asarray(p, np.int32) - [x0, y0]], 255)
        area = np.count_nonzero(mask)
        out[i] = np.count_nonzero(view["edges"][y0:y1, x0:x1][mask > 0]) / area if area else 0.0
    return out


# ================= method B: scale-normalised space features + SVM =============
_YY, _XX = np.mgrid[:PATCH_SIZE, :PATCH_SIZE] - PATCH_SIZE / 2
HIGH_FREQ_MASK = np.hypot(_XX, _YY) > PATCH_SIZE / 8


def describe_patch(g, hsv, gray_mean, area):
    """Feature vector of one 48x48 parking patch (g: gray uint8, hsv: HSV uint8)."""
    gf = g.astype(np.float32)
    # edges: fixed thresholds + automatic thresholds from the patch median (illumination adaptive)
    med = float(np.median(g))
    edges = [cv2.Canny(g, 50, 150).mean() / 255,
             cv2.Canny(g, int(max(0, 0.67 * med)), int(min(255, 1.33 * med))).mean() / 255]
    # spatial filtering: Sobel gradient statistics
    mag = np.hypot(cv2.Sobel(gf, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(gf, cv2.CV_32F, 0, 1, ksize=3))
    grad = [mag.mean() / 255, mag.std() / 255, (mag > 60).mean()]
    # histogram processing: contrast, entropy, brightness relative to the whole frame (shadow aware)
    hist = np.bincount(g.ravel() // 16, minlength=16) / g.size
    entropy = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0]))
    intensity = [gf.std() / 255, entropy / 4, gf.mean() / gray_mean]
    # texture: rotation-invariant uniform LBP at two radii (invariant to monotonic illumination change)
    lbp1 = np.bincount(local_binary_pattern(g, 8, 1, "uniform").astype(int).ravel(), minlength=10) / g.size
    lbp2 = np.bincount(local_binary_pattern(g, 16, 2, "uniform").astype(int).ravel(), minlength=18) / g.size
    # frequency domain: share of high-frequency energy
    power = np.abs(np.fft.fftshift(np.fft.fft2(gf - gf.mean()))) ** 2
    high_freq = [power[HIGH_FREQ_MASK].sum() / (power.sum() + 1e-9)]
    # corner detection: Shi-Tomasi corner density
    corners = cv2.goodFeaturesToTrack(g, maxCorners=50, qualityLevel=0.05, minDistance=3)
    corner_density = [0.0 if corners is None else len(corners) / 50]
    # colour: cars are often saturated, asphalt is grey
    s, v = hsv[..., 1].astype(np.float32), hsv[..., 2].astype(np.float32)
    colour = [s.mean() / 255, s.std() / 255, v.std() / 255]
    # shape: HOG on a 3x3 cell grid
    shape = hog(g, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(1, 1), feature_vector=True)
    scale = [np.log1p(area) / 10]
    return np.concatenate([edges, grad, intensity, lbp1, lbp2, high_freq, corner_density,
                           colour, shape, scale]).astype(np.float32)


FEATURE_LEN = len(describe_patch(np.zeros((PATCH_SIZE, PATCH_SIZE), np.uint8),
                                 np.zeros((PATCH_SIZE, PATCH_SIZE, 3), np.uint8), 1.0, 1.0))


def space_features(view, polys):
    feats = np.zeros((len(polys), FEATURE_LEN), dtype=np.float32)
    for i, p in enumerate(polys):
        x0, y0, x1, y1 = crop_box(p, view["gray"].shape)
        if x1 - x0 < 3 or y1 - y0 < 3:
            continue
        g = cv2.resize(view["gray"][y0:y1, x0:x1], (PATCH_SIZE, PATCH_SIZE), interpolation=cv2.INTER_AREA)
        hsv = cv2.resize(view["hsv"][y0:y1, x0:x1], (PATCH_SIZE, PATCH_SIZE), interpolation=cv2.INTER_AREA)
        feats[i] = describe_patch(g, hsv, view["gray_mean"], (x1 - x0) * (y1 - y0))
    return feats


def make_svm():
    """RBF-SVM with sigmoid (Platt) probability calibration on 5 folds."""
    svm = SVC(C=10, gamma="scale", class_weight="balanced")
    return make_pipeline(StandardScaler(), CalibratedClassifierCV(svm, method="sigmoid", cv=5, ensemble=False))


# ======================= method C: YOLOv8 vehicle detection =====================
class VehicleDetector:
    def __init__(self, weights):
        self.model = None
        if YOLO is None:
            print("[warn] ultralytics is not installed -> methods C and D are skipped")
            return
        try:
            self.model = YOLO(weights)
        except Exception as exc:  # e.g. no internet to download the weights
            print(f"[warn] YOLO could not be loaded ({exc}) -> methods C and D are skipped")

    @property
    def available(self):
        return self.model is not None

    def _predict(self, img, imgsz):
        r = self.model.predict(img, classes=VEHICLE_CLASSES, conf=YOLO_CONF, imgsz=imgsz, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return np.zeros((0, 5), dtype=np.float32)
        return np.hstack([r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()[:, None]]).astype(np.float32)

    def detect(self, img):
        """Vehicle boxes (x0, y0, x1, y1, conf) in the coordinates of img."""
        h, w = img.shape[:2]
        # restore the original 16:9 geometry: cars are no longer squeezed and small cars get more pixels
        big = cv2.resize(img, DETECTION_SIZE, interpolation=cv2.INTER_CUBIC)
        H, W = big.shape[:2]
        boxes = [self._predict(big, W)]
        if DETECTION_TILES > 1:
            # overlapping tiles are upsampled by YOLO -> distant cars become large enough to detect
            n = DETECTION_TILES
            tw, th = int(W / n * (1 + TILE_OVERLAP)), int(H / n * (1 + TILE_OVERLAP))
            for gy in range(n):
                for gx in range(n):
                    x0 = int(np.clip(gx * W / n - TILE_OVERLAP / 2 * W / n, 0, W - tw))
                    y0 = int(np.clip(gy * H / n - TILE_OVERLAP / 2 * H / n, 0, H - th))
                    b = self._predict(big[y0:y0 + th, x0:x0 + tw], TILE_IMGSZ)
                    b[:, [0, 2]] += x0
                    b[:, [1, 3]] += y0
                    boxes.append(b)
        boxes = np.vstack(boxes)
        if len(boxes) and DETECTION_TILES > 1:  # merge duplicates from the whole frame and the tiles
            keep = cv2.dnn.NMSBoxes([[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])]
                                     for b in boxes], boxes[:, 4].tolist(), YOLO_CONF, 0.5)
            boxes = boxes[np.array(keep, dtype=int).ravel()]
        boxes[:, [0, 2]] *= w / W
        boxes[:, [1, 3]] *= h / H
        return boxes


def detection_scores(polys, dets):
    """Per space: [IoU of the one-to-one assigned vehicle, max coverage by any vehicle, assigned conf]."""
    out = np.zeros((len(polys), 3), dtype=np.float32)
    if dets is None or len(dets) == 0:
        return out
    S = [to_shapely(p) for p in polys]
    iou = np.zeros((len(polys), len(dets)))
    cover = np.zeros_like(iou)
    for i, j in overlapping_pairs(bounds_of(polys), dets[:, :4].astype(float)):
        b = shapely_box(*dets[j, :4])
        inter = S[i].intersection(b).area
        if inter > 0:
            iou[i, j] = inter / (S[i].area + b.area - inter)
            cover[i, j] = inter / max(S[i].area, 1e-6)
    # one car occupies one space: neighbours touched by the same oblique box are not marked occupied
    idx = greedy_assignment(iou)
    for i, j in enumerate(idx):
        if j >= 0:
            out[i, 0], out[i, 2] = iou[i, j], dets[j, 4]
    out[:, 1] = cover.max(axis=1)
    return out


# ============================ analysis & prediction ============================
def analyze(img, polys, detector, cache, cache_key):
    """Everything the four methods need for one frame (detections are cached per image)."""
    view = image_view(img)
    result = {"dens": canny_densities(view, polys), "feats": space_features(view, polys),
              "dets": None, "dscore": None}
    if detector.available:
        if cache_key not in cache:
            cache[cache_key] = detector.detect(img)
        result["dets"] = cache[cache_key]
        result["dscore"] = detection_scores(polys, result["dets"])
    return result


def best_threshold(values, labels):
    """Threshold on a 1-D score that maximises balanced accuracy (robust to class imbalance)."""
    values, labels = np.asarray(values, float), np.asarray(labels)
    if len(labels) == 0 or labels.min() == labels.max():
        return 0.5
    cands = np.linspace(0.0, max(values.max(), 1e-6), 500)
    pred = values[None, :] >= cands[:, None]
    bal = (pred[:, labels == 1].mean(1) + (~pred[:, labels == 0]).mean(1)) / 2
    return float(cands[int(np.argmax(bal))])


def available_methods(models):
    return ["A", "B", "C", "D"] if models["hybrid"] is not None else ["A", "B"]


def error_rates(pred, labels):
    """(true positive rate, false positive rate) of a hard decision, clipped away from 0 and 1."""
    tpr = pred[labels == 1].mean() if (labels == 1).any() else 0.5
    fpr = pred[labels == 0].mean() if (labels == 0).any() else 0.5
    return float(np.clip(tpr, 0.02, 0.98)), float(np.clip(fpr, 0.02, 0.98))


def likelihood_ratio(models, method, probs, camera):
    """P(observation | occupied) / P(observation | free) for one frame - the tracker's input.
    B/D: calibrated probability with the training prior divided out. A/C: from calibration TPR/FPR."""
    if method in ("B", "D"):
        p = np.clip(probs, 0.02, 0.98)
        return p / (1 - p) / models["prior_odds"]
    tpr, fpr = models["rates"][method].get(camera, models["rates"][method]["*"])
    return np.where(probs >= 0.5, tpr / fpr, (1 - tpr) / (1 - fpr))


def predict(models, a, camera):
    """Occupancy probability per space for every method (A and C are hard 0/1 decisions)."""
    thr_a = models["thr_canny"].get(camera, models["thr_canny"]["*"])
    out = {"A": (a["dens"] >= thr_a).astype(float),
           "B": models["space"].predict_proba(a["feats"])[:, 1]}
    if models["hybrid"] is not None:
        thr_c = models["thr_det"].get(camera, models["thr_det"]["*"])
        out["C"] = (a["dscore"][:, 0] >= thr_c).astype(float)
        out["D"] = models["hybrid"].predict_proba(np.hstack([a["feats"], a["dscore"]]))[:, 1]
    return out


def train_models(calib, detector, cache):
    """Fit B and D on the calibration dates; calibrate the A and C thresholds per camera (+ pooled '*')."""
    feats, dscores, dens, labels, cams = [], [], [], [], []
    for camera, frames in calib.items():
        for f in frames:
            a = analyze(imread(f["img_path"]), f["polys"], detector, cache, (f["img_path"], "clean"))
            feats.append(a["feats"])
            dens.append(a["dens"])
            labels.append(f["labels"])
            cams += [camera] * len(f["labels"])
            if a["dscore"] is not None:
                dscores.append(a["dscore"])
    feats, dens, labels, cams = np.vstack(feats), np.concatenate(dens), np.concatenate(labels), np.array(cams)
    dscores = np.vstack(dscores) if dscores else None

    models = {"thr_canny": {}, "thr_det": {}, "rates": {"A": {}, "C": {}}, "hybrid": None,
              "prior_odds": labels.mean() / (1 - labels.mean())}
    for camera in list(calib) + ["*"]:
        m = np.ones(len(labels), bool) if camera == "*" else cams == camera
        models["thr_canny"][camera] = best_threshold(dens[m], labels[m])
        models["rates"]["A"][camera] = error_rates(dens[m] >= models["thr_canny"][camera], labels[m])
        if dscores is not None:
            models["thr_det"][camera] = best_threshold(dscores[m, 0], labels[m])
            models["rates"]["C"][camera] = error_rates(dscores[m, 0] >= models["thr_det"][camera], labels[m])
    models["space"] = make_svm().fit(feats, labels)
    if dscores is not None:
        models["hybrid"] = make_svm().fit(np.hstack([feats, dscores]), labels)

    print(f"  training spaces: {len(labels)} ({100 * labels.mean():.0f}% occupied) "
          f"from {sum(len(f) for f in calib.values())} calibration frames, {FEATURE_LEN} features")
    for camera in list(calib) + ["*"]:
        name = "pooled" if camera == "*" else camera
        det = f", YOLO IoU threshold {models['thr_det'][camera]:.3f}" if dscores is not None else ""
        print(f"  {name:<7} Canny threshold {models['thr_canny'][camera]:.3f}{det}")
    return models


# ================================== tracking ===================================
class OccupancyTracker:
    """
    Two-state (free/occupied) HMM forward filter per space.
    Between two frames a space switches state with probability q = 0.5 * (1 - exp(-dt / tau)):
    for frames seconds apart q ~ 0 (strong smoothing, a passing pedestrian cannot flip a space),
    for snapshots far apart q -> 0.5 (the new observation dominates). Observations enter as
    likelihood ratios, so an uncertain observation is outweighed by the history while a confident
    one flips the state immediately.
    """

    def __init__(self, tau_min=TRACK_MEAN_DWELL_MIN):
        self.tau = tau_min
        self.post = self.state = self.t = None
        self.events = []  # (time, space_id, old_state, new_state)

    def update(self, t, lr):
        lr = np.asarray(lr, dtype=float)
        if self.post is None or len(lr) != len(self.post):
            prior = np.full(len(lr), 0.5)
        else:
            dt = max((t - self.t).total_seconds() / 60.0, 0.0)
            q = 0.5 * (1 - np.exp(-dt / self.tau))
            prior = self.post * (1 - q) + (1 - self.post) * q
        self.post = lr * prior / (lr * prior + (1 - prior))
        new = self.post >= 0.5
        if self.state is not None and len(self.state) == len(new):
            for i in np.flatnonzero(new != self.state):
                self.events.append((t, i + 1, bool(self.state[i]), bool(new[i])))
        self.state, self.t = new, t
        return new.copy()


# ========================= viewpoint compensation (registration) ===============
class Registrar:
    """
    Estimates the homography reference frame -> current frame from SIFT matches (Lowe ratio test +
    RANSAC). SIFT was chosen over ORB because it kept matching across weather and lighting changes
    (UFPR04 rainy December reference vs cloudy January frames). If a frame cannot be registered,
    the last good homography is kept, since a moved camera normally stays where it moved to.
    """

    def __init__(self, ref_img):
        self.sift = cv2.SIFT_create(nfeatures=4000)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        self.shape = ref_img.shape[:2]
        self.kp, self.des = self.sift.detectAndCompute(image_view(ref_img)["gray"], None)
        self.last_H = np.eye(3)

    def estimate(self, img):
        H = self._estimate(img)
        if H is not None:
            self.last_H = H
        return self.last_H

    def _estimate(self, img):
        kp, des = self.sift.detectAndCompute(image_view(img)["gray"], None)
        if des is None or self.des is None or len(kp) < 8:
            return None
        good = [p[0] for p in self.matcher.knnMatch(self.des, des, k=2)
                if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
        if len(good) < 10:
            return None
        src = np.float32([self.kp[m.queryIdx].pt for m in good])
        dst = np.float32([kp[m.trainIdx].pt for m in good])
        H, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
        if H is None or inliers.sum() < 10:
            return None
        # reject implausible solutions (corners moving more than 25% of the frame size)
        h, w = self.shape
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        if np.abs(cv2.perspectiveTransform(corners, H) - corners).max() > 0.25 * max(h, w):
            return None
        return H


# ========================= synthetic robustness perturbations ===================
def add_shadow(img, rng):
    """Soft-edged cast shadow over a random half-plane (30-70% of the frame), 35-55% brightness."""
    h, w = img.shape[:2]
    angle = rng.uniform(0, np.pi)
    proj = np.cos(angle) * np.arange(w)[None, :] + np.sin(angle) * np.arange(h)[:, None]
    lo, hi = proj.min(), proj.max()
    mask = (proj < rng.uniform(lo + 0.3 * (hi - lo), lo + 0.7 * (hi - lo))).astype(np.float32)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=8)
    factor = rng.uniform(0.35, 0.55)
    return np.clip(img.astype(np.float32) * (1 - (1 - factor) * mask[..., None]), 0, 255).astype(np.uint8)


def add_occlusion(img, polys, rng, fraction=0.3):
    """Noisy coloured occluders (pedestrian, pole, branch) over 20-60% of 30% of the spaces."""
    out = img.copy()
    for i in rng.choice(len(polys), max(1, int(round(fraction * len(polys)))), replace=False):
        x0, y0, x1, y1 = crop_box(polys[i], img.shape)
        bw, bh = x1 - x0, y1 - y0
        if bw < 4 or bh < 4:
            continue
        ow, oh = max(2, int(bw * rng.uniform(0.4, 0.7))), max(2, int(bh * rng.uniform(0.5, 0.9)))
        ox, oy = x0 + rng.integers(0, bw - ow + 1), y0 + rng.integers(0, bh - oh + 1)
        patch = rng.integers(20, 235, 3) + rng.normal(0, 18, (oh, ow, 3))
        out[oy:oy + oh, ox:ox + ow] = np.clip(patch, 0, 255).astype(np.uint8)
    return out


def camera_shake(img, rng):
    """Random similarity transform: +-3 deg rotation, +-4% zoom, +-20 px shift."""
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-3, 3), rng.uniform(0.96, 1.04))
    M[:, 2] += rng.uniform(-20, 20, 2)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT101), np.vstack([M, [0, 0, 1]])


IMAGE_KEY = {"clean": "clean", "shadow": "shadow", "occlusion": "occlusion", "shake_fixed": "shake",
             "shake_registered": "shake", "refmap_fixed": "clean", "refmap_registered": "clean"}
KEY_ID = {"clean": 0, "shadow": 1, "occlusion": 2, "shake": 3}


# ============================== sequence processing ============================
def run_sequence(seq, models, detector, cache, scenario="clean", frame_step=1, ref=None):
    """
    Process one sequence in time order with every method and its tracker.
    scenario: clean | shadow | occlusion | shake_fixed | shake_registered | refmap_fixed | refmap_registered
    ref: (reference image, reference map) of the camera for the refmap_* scenarios.
    """
    frames = seq["frames"][::frame_step]
    methods = available_methods(models)
    trackers = {m: OccupancyTracker() for m in methods}
    key = IMAGE_KEY[scenario]
    registrar, reg_polys = None, None
    if scenario == "shake_registered":
        registrar, reg_polys = Registrar(imread(frames[0]["img_path"])), frames[0]["polys"]
    elif scenario.startswith("refmap"):
        reg_polys = ref[1]
        registrar = Registrar(ref[0]) if scenario == "refmap_registered" else None

    res = {"camera": seq["camera"], "condition": seq["condition"], "scenario": scenario,
           "times": [], "img_paths": [], "polys": [], "labels": [], "map_iou": [], "matched": [],
           "lr": {m: [] for m in methods}, "tracked": {m: [] for m in methods}}
    for f in frames:
        img = imread(f["img_path"])
        rng = np.random.default_rng([SEED, KEY_ID[key], zlib.crc32(os.path.basename(f["img_path"]).encode())])
        polys, labels = f["polys"], f["labels"]
        map_iou = matched = np.nan
        if scenario == "shadow":
            img = add_shadow(img, rng)
        elif scenario == "occlusion":
            img = add_occlusion(img, polys, rng)
        elif scenario.startswith("shake"):
            img, M = camera_shake(img, rng)
            true_polys = transform_polys(polys, M)
            if registrar is not None:  # same day -> reference map has the same spaces in the same order
                polys = transform_polys(reg_polys, registrar.estimate(img))
            map_iou = np.mean([shapely_iou(to_shapely(p), to_shapely(q)) for p, q in zip(polys, true_polys)])
        elif scenario.startswith("refmap"):
            H = registrar.estimate(img) if registrar is not None else np.eye(3)
            polys = transform_polys(reg_polys, H)
            idx, ious = match_polys(polys, f["polys"])
            labels = np.where(idx >= 0, f["labels"][np.maximum(idx, 0)], -1)
            matched = float((idx >= 0).mean())
            map_iou = float(ious[idx >= 0].mean()) if (idx >= 0).any() else 0.0

        a = analyze(img, polys, detector, cache, (f["img_path"], key))
        probs = predict(models, a, seq["camera"])
        for m in methods:
            lr = likelihood_ratio(models, m, probs[m], seq["camera"])
            res["lr"][m].append(lr)
            res["tracked"][m].append(trackers[m].update(f["time"], lr))
        res["times"].append(f["time"])
        res["img_paths"].append(f["img_path"])
        res["polys"].append(polys)
        res["labels"].append(labels)
        res["map_iou"].append(map_iou)
        res["matched"].append(matched)

    res["labels"] = np.array(res["labels"])
    for m in methods:
        res["lr"][m] = np.array(res["lr"][m])
        res["tracked"][m] = np.array(res["tracked"][m])
    # single-frame decision = the more likely state (likelihood ratio >= 1, i.e. equal priors)
    res["raw"] = {m: res["lr"][m] >= 1.0 for m in methods}
    res["events"] = {m: trackers[m].events for m in methods}
    return res


# ================================== metrics ====================================
def score(labels, pred):
    labels, pred = np.asarray(labels).ravel(), np.asarray(pred).ravel().astype(int)
    valid = labels >= 0
    labels, pred = labels[valid], pred[valid]
    if len(labels) == 0:
        return {"acc": np.nan, "bal_acc": np.nan, "f1": np.nan}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {"acc": 100 * accuracy_score(labels, pred),
                "bal_acc": 100 * balanced_accuracy_score(labels, pred),
                "f1": 100 * f1_score(labels, pred, zero_division=0)}


def free_counts(labels, states):
    valid = labels >= 0
    return ((labels == 0) & valid).sum(1), ((~states.astype(bool)) & valid).sum(1)


def count_changes(states, valid):
    states = states.astype(bool)
    both = valid[1:] & valid[:-1]
    return int(((states[1:] != states[:-1]) & both).sum())


# ============================ outputs: tables & files ==========================
def fmt(v):
    if isinstance(v, (float, np.floating)):
        return "-" if np.isnan(v) else f"{v:.1f}"
    return str(v)


def print_table(title, header, rows):
    cells = [[fmt(v) for v in r] for r in rows]
    widths = [max([len(h)] + [len(c[i]) for c in cells]) for i, h in enumerate(header)]
    line = "  ".join(f"{h:<{w}}" if i == 0 else f"{h:>{w}}" for i, (h, w) in enumerate(zip(header, widths)))
    print(f"\n{title}\n{line}\n{'-' * len(line)}")
    for c in cells:
        print("  ".join(f"{v:<{w}}" if i == 0 else f"{v:>{w}}" for i, (v, w) in enumerate(zip(c, widths))))


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows([[fmt(v) for v in r] for r in rows])


def draw_states(img, polys, states, labels=None, title=""):
    """Green = free, red = occupied; wrong decisions get a yellow halo; banner with availability."""
    canvas = img.copy()
    for i, (p, s) in enumerate(zip(polys, states)):
        if labels is not None and labels[i] >= 0 and bool(labels[i]) != bool(s):
            cv2.polylines(canvas, [p], True, WARNING_BGR, 4)
        cv2.polylines(canvas, [p], True, CRITICAL_BGR if s else GOOD_BGR, 2)
    n, free = len(states), int((~np.asarray(states, bool)).sum())
    text = f"{title} | Free {free}/{n} ({100 * free / max(n, 1):.0f}%)"
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 26), (20, 20, 20), -1)
    cv2.putText(canvas, text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def time_ticks(ax, times, n=7):
    idx = np.unique(np.linspace(0, len(times) - 1, min(n, len(times))).round().astype(int))
    ax.set_xticks(idx)
    ax.set_xticklabels([times[i].strftime("%H:%M") for i in idx])


def plot_dashboard(path, res, methods, best):
    labels, times = res["labels"], res["times"]
    T, N = labels.shape
    fig = plt.figure(figsize=(15, 7.2), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.35], hspace=0.45, wspace=0.12)
    date = times[0].strftime("%Y-%m-%d")
    fig.suptitle(f"{res['camera']} / {res['condition']} - {date}: parking availability",
                 x=0.01, ha="left", fontsize=13, color=INK)

    mid = T // 2
    frame = draw_states(imread(res["img_paths"][mid]), res["polys"][mid], res["tracked"][best][mid],
                        labels[mid], times[mid].strftime("%H:%M"))
    ax0 = fig.add_subplot(gs[:, 0])
    ax0.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    ax0.set_title(f"{METHODS[best]} (tracked) - green free, red occupied, yellow halo = error",
                  fontsize=9, color=INK_2)
    ax0.axis("off")

    ax1 = fig.add_subplot(gs[0, 1])
    style_axes(ax1)
    gt_free, _ = free_counts(labels, res["tracked"][best])
    ax1.plot(range(T), gt_free, color=INK, lw=2, marker="o", ms=4, label="Ground truth", zorder=5)
    for m in methods:
        ax1.plot(range(T), free_counts(labels, res["tracked"][m])[1], color=SERIES[m], lw=2,
                 label=f"{m}: {METHODS[m]}")
    ax1.set_ylim(0, N)
    ax1.set_ylabel("Free spaces")
    ax1.set_title("Free spaces over time (tracked states)", loc="left", color=INK)
    time_ticks(ax1, times)
    ax1.legend(fontsize=8, frameon=False, ncol=3, loc="upper left", bbox_to_anchor=(0, -0.12))

    ax2 = fig.add_subplot(gs[1, 1])
    gt, pr = labels == 1, res["tracked"][best]
    codes = np.where(pr == gt, gt.astype(int), np.where(pr, 2, 3))
    ax2.imshow(codes.T, aspect="auto", interpolation="nearest", cmap=ListedColormap(HEAT_COLORS),
               vmin=-0.5, vmax=3.5)
    ax2.set_ylabel("Space id")
    ax2.set_yticks(np.unique(np.linspace(0, N - 1, 6).round().astype(int)))
    ax2.set_yticklabels([str(i + 1) for i in ax2.get_yticks()])
    time_ticks(ax2, times)
    ax2.tick_params(colors=INK_2, labelsize=8)
    for side in ax2.spines.values():
        side.set_color(AXIS)
    ax2.set_title(f"Per-space states vs ground truth - {METHODS[best]} (tracked)", loc="left", color=INK)
    ax2.legend(handles=[Patch(color=c, label=n) for c, n in zip(HEAT_COLORS, HEAT_NAMES)],
               fontsize=8, frameon=False, ncol=4, loc="upper left", bbox_to_anchor=(0, -0.1))
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def plot_grouped_bars(path, title, groups, values, ylabel="Balanced accuracy (%)"):
    """values: {method: [value per group]} - one bar per method inside every group."""
    methods = list(values)
    x = np.arange(len(groups))
    width = 0.8 / len(methods)
    fig, ax = plt.subplots(figsize=(max(7.5, 1.35 * len(groups) + 2.5), 4.4), facecolor=SURFACE)
    style_axes(ax)
    for k, m in enumerate(methods):
        ax.bar(x - 0.4 + width * (k + 0.5), np.nan_to_num(values[m]), width, color=SERIES[m],
               edgecolor=SURFACE, linewidth=2, label=f"{m}: {METHODS[m]}")
    ax.axhline(50, color=MUTED, lw=1, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=8)
    ax.set_ylim(0, 100)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", color=INK)
    handles = ax.get_legend_handles_labels()[0] + [Line2D([0], [0], color=MUTED, lw=1, ls="--", label="Chance (50%)")]
    ax.legend(handles=handles, fontsize=8, frameon=False, ncol=len(handles), loc="upper left",
              bbox_to_anchor=(0, -0.12))
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def report_sequence(res, methods, out_dir):
    """Metrics table, availability statistics, event log, per-space stats and figures for one sequence."""
    os.makedirs(out_dir, exist_ok=True)
    labels, times = res["labels"], res["times"]
    valid = labels >= 0
    T, N = labels.shape
    best = "D" if "D" in methods else "B"
    gt_changes = count_changes(labels == 1, valid)

    rows, table = [], []
    for m in methods:
        raw, trk = score(labels, res["raw"][m]), score(labels, res["tracked"][m])
        gt_free, raw_free = free_counts(labels, res["raw"][m])
        _, trk_free = free_counts(labels, res["tracked"][m])
        row = {"camera": res["camera"], "condition": res["condition"], "method": m, "frames": T,
               "spaces": int(valid.sum()), "occ_pct": 100 * (labels == 1).sum() / valid.sum(),
               "acc": raw["acc"], "bal_acc": raw["bal_acc"], "f1": raw["f1"], "bal_acc_tracked": trk["bal_acc"],
               "free_mae_raw": np.abs(gt_free - raw_free).mean(), "free_mae_tracked": np.abs(gt_free - trk_free).mean(),
               "changes_gt": gt_changes, "changes_raw": count_changes(res["raw"][m], valid),
               "changes_tracked": count_changes(res["tracked"][m], valid)}
        rows.append(row)
        table.append([f"{m}: {METHODS[m]}", row["acc"], row["bal_acc"], row["f1"], row["bal_acc_tracked"],
                      row["free_mae_raw"], row["free_mae_tracked"],
                      f"{gt_changes}/{row['changes_raw']}/{row['changes_tracked']}"])
    date = times[0].strftime("%Y-%m-%d")
    print_table(f"=== {res['camera']} / {res['condition']} ({date}, {T} frames, {N} spaces, "
                f"{rows[0]['occ_pct']:.0f}% occupied) ===",
                ["method", "Acc", "BalAcc", "F1", "BalAcc trk", "FreeMAE raw", "FreeMAE trk",
                 "changes GT/raw/trk"], table)

    # availability statistics of the best method (tracked) next to the ground truth
    gt_free, pred_free = free_counts(labels, res["tracked"][best])
    hours = np.array([t.hour for t in times])
    hour_free = {h: pred_free[hours == h].mean() for h in np.unique(hours)}
    busiest = min(hour_free, key=hour_free.get)
    i_min, i_max = int(np.argmin(pred_free)), int(np.argmax(pred_free))
    print(f"  Availability ({METHODS[best]}, tracked): mean free {pred_free.mean():.1f}/{N} "
          f"(GT {gt_free.mean():.1f}); fewest free {pred_free[i_min]} at {times[i_min]:%H:%M} "
          f"(GT {gt_free[i_min]}); most free {pred_free[i_max]} at {times[i_max]:%H:%M} "
          f"(GT {gt_free[i_max]}); busiest hour {busiest:02d}:00-{busiest + 1:02d}:00")

    # files: timeline, events, per-space statistics, figures
    header = ["time", "total_spaces", "gt_free"] + [f"{m}_free_{k}" for m in methods for k in ("raw", "tracked")]
    timeline = []
    for t in range(T):
        r = [times[t].strftime("%H:%M:%S"), N, int(gt_free[t])]
        for m in methods:
            r += [int(free_counts(labels[t:t + 1], res["raw"][m][t:t + 1])[1][0]),
                  int(free_counts(labels[t:t + 1], res["tracked"][m][t:t + 1])[1][0])]
        timeline.append(r)
    write_csv(os.path.join(out_dir, "timeline.csv"), header, timeline)

    events = [[t.strftime("%H:%M:%S"), sid, "occupied" if old else "free", "occupied" if new else "free"]
              for t, sid, old, new in res["events"][best]]
    write_csv(os.path.join(out_dir, f"events_{best}.csv"), ["time", "space_id", "from", "to"], events)

    minutes = np.diff([t.timestamp() / 60 for t in times]) if T > 1 else np.array([0.0])
    dt = np.append(minutes, np.median(minutes) if T > 1 else 0.0)
    pred = res["tracked"][best]
    space_rows = [[i + 1, 100 * (labels[:, i] == 1).mean(), 100 * pred[:, i].mean(),
                   count_changes(labels[:, i:i + 1] == 1, valid[:, i:i + 1]),
                   count_changes(pred[:, i:i + 1], valid[:, i:i + 1]),
                   float(((labels[:, i] == 1) * dt).sum()), float((pred[:, i] * dt).sum())] for i in range(N)]
    write_csv(os.path.join(out_dir, "space_stats.csv"),
              ["space_id", "gt_occupancy_pct", "pred_occupancy_pct", "gt_changes", "pred_changes",
               "gt_occupied_minutes", "pred_occupied_minutes"], space_rows)
    busiest_spaces = sorted(space_rows, key=lambda r: -r[4])[:3]
    print(f"  {len(events)} change events logged (GT changes {gt_changes}); highest turnover spaces: "
          + ", ".join(f"#{r[0]} ({r[4]} changes)" for r in busiest_spaces))

    plot_dashboard(os.path.join(out_dir, "dashboard.png"), res, methods, best)
    for t in sorted({0, T // 2, T - 1}):
        img = draw_states(imread(res["img_paths"][t]), res["polys"][t], res["tracked"][best][t],
                          labels[t], f"{res['camera']} {times[t]:%H:%M} {best}")
        imwrite(os.path.join(out_dir, f"frame_{times[t]:%H%M}.jpg"), img)
    return rows


# ========================= space identification: discovery =====================
def discover_spaces(det_list, min_support=0.3, iou_thr=0.5):
    """Cluster vehicle boxes over a day; clusters present in >= min_support of frames = parking spaces."""
    clusters = []  # [sum of boxes, count, last frame index]
    for fi, dets in enumerate(det_list):
        for d in dets:
            b = d[:4].astype(float)
            if clusters:
                means = np.array([c[0] / c[1] for c in clusters])
                ix0, iy0 = np.maximum(means[:, 0], b[0]), np.maximum(means[:, 1], b[1])
                ix1, iy1 = np.minimum(means[:, 2], b[2]), np.minimum(means[:, 3], b[3])
                inter = np.clip(ix1 - ix0, 0, None) * np.clip(iy1 - iy0, 0, None)
                area = lambda x: (x[..., 2] - x[..., 0]) * (x[..., 3] - x[..., 1])
                iou = inter / (area(means) + area(b) - inter + 1e-9)
                k = int(np.argmax(iou))
                if iou[k] >= iou_thr:
                    if clusters[k][2] != fi:
                        clusters[k][0] += b
                        clusters[k][1] += 1
                        clusters[k][2] = fi
                    continue
            clusters.append([b.copy(), 1, fi])
    keep = [c[0] / c[1] for c in clusters if c[1] >= min_support * len(det_list)]
    return np.array(keep).reshape(-1, 4)


# ==================================== main =====================================
def main():
    t_start = time.time()
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    sequences = load_sequences(DATASET_ROOT)
    if not sequences:
        print(f"No sequences under {DATASET_ROOT}/<Camera>/<Weather>/. Run prepare_dataset.py first.")
        return
    cameras = sorted({s["camera"] for s in sequences})
    calib = {c: load_frames(os.path.join(CALIB_ROOT, c)) for c in cameras}
    calib = {c: f for c, f in calib.items() if f}
    unseen = load_sequences(UNSEEN_ROOT, default_condition="Unseen camera") if os.path.isdir(UNSEEN_ROOT) else []
    if not calib:
        print(f"No calibration data under {CALIB_ROOT}/<Camera>/. Run prepare_dataset.py first.")
        return
    print(f"Test sequences: {len(sequences)} ({sum(len(s['frames']) for s in sequences)} frames) | "
          f"calibration cameras: {', '.join(calib)} | unseen cameras: "
          f"{', '.join(s['camera'] for s in unseen) or '-'}")

    detector = VehicleDetector(YOLO_WEIGHTS)
    cache = {}

    # 1) training and threshold calibration (dates disjoint from the test sequences)
    print("\n[1] Training / calibration")
    models = train_models(calib, detector, cache)
    methods = available_methods(models)

    # 2) occupancy + tracking + availability statistics for every test sequence
    print("\n[2] Occupancy classification, tracking and availability statistics")
    summary, clean_results = [], []
    for seq in sequences:
        res = run_sequence(seq, models, detector, cache, "clean")
        clean_results.append(res)
        summary += report_sequence(res, methods, os.path.join(OUTPUT_ROOT, f"{seq['camera']}_{seq['condition']}"))

    # 3) parking-space identification from detections
    if detector.available:
        disc_rows = []
        for res in clean_results:
            found = discover_spaces([cache[(p, "clean")] for p in res["img_paths"]])
            gt_polys, labels = res["polys"][0], res["labels"]
            idx, _ = match_polys(gt_polys, [box_to_poly(b) for b in found])
            busy = (labels == 1).mean(0) >= 0.3
            hit = idx >= 0
            disc_rows.append([f"{res['camera']}/{res['condition']}", len(gt_polys), len(found),
                              100 * hit.mean(), 100 * hit[busy].mean() if busy.any() else np.nan,
                              100 * hit.sum() / max(len(found), 1)])
            img = imread(res["img_paths"][len(res["img_paths"]) // 2])
            for p in gt_polys:
                cv2.polylines(img, [p], True, GOOD_BGR, 1)
            for b in found:
                cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), BLUE_BGR, 2)
            imwrite(os.path.join(OUTPUT_ROOT, f"{res['camera']}_{res['condition']}", "space_discovery.jpg"), img)
        header = ["sequence", "map spaces", "discovered", "recall %", "recall busy %", "match % of discovered"]
        print_table("[3] Parking-space discovery from persistent vehicle detections "
                    "(busy = occupied >= 30% of the day; unmatched discoveries are often real but "
                    "unannotated spaces)", header, disc_rows)
        write_csv(os.path.join(OUTPUT_ROOT, "space_discovery.csv"), header, disc_rows)

    # 4) robustness: synthetic shadows, occlusion and camera shake on every 2nd frame
    print("\n[4] Robustness experiments (every "
          f"{ROBUSTNESS_FRAME_STEP}nd frame of each sequence)")
    scenarios = ["clean", "shadow", "occlusion", "shake_fixed", "shake_registered"]
    rob_rows, rob_plot = [], {m: [] for m in methods}
    for sc in scenarios:
        results = [run_sequence(s, models, detector, cache, sc, ROBUSTNESS_FRAME_STEP) for s in sequences]
        labels = np.concatenate([r["labels"].ravel() for r in results])
        map_iou = np.nanmean(np.concatenate([r["map_iou"] for r in results])) if sc.startswith("shake") else np.nan
        for m in methods:
            raw = score(labels, np.concatenate([r["raw"][m].ravel() for r in results]))
            trk = score(labels, np.concatenate([r["tracked"][m].ravel() for r in results]))
            rob_rows.append([sc, f"{m}: {METHODS[m]}", raw["acc"], raw["bal_acc"], trk["bal_acc"], 100 * map_iou])
            rob_plot[m].append(raw["bal_acc"])
        print(f"  {sc} done ({time.time() - t_start:.0f}s)")
    header = ["scenario", "method", "Acc", "BalAcc", "BalAcc tracked", "map IoU %"]
    print_table("Robustness (map IoU = overlap of the used space map with the true space positions)", header, rob_rows)
    write_csv(os.path.join(OUTPUT_ROOT, "robustness.csv"), header, rob_rows)
    plot_grouped_bars(os.path.join(OUTPUT_ROOT, "robustness_balacc.png"),
                      "Robustness: balanced accuracy per perturbation", scenarios, rob_plot)

    # 5) real viewpoint change: camera map annotated on the first day, camera moved later
    move_rows = []
    for camera in cameras:
        seqs = sorted([r for r in sequences if r["camera"] == camera], key=lambda s: s["frames"][0]["time"])
        ref_frame = seqs[0]["frames"][0]
        ref = (imread(ref_frame["img_path"]), ref_frame["polys"])
        for seq in seqs[1:]:
            _, ious = match_polys(ref[1], seq["frames"][0]["polys"])
            if ious.mean() >= 0.8:  # same map -> the camera did not move for this sequence
                continue
            for sc in ("refmap_fixed", "refmap_registered"):
                r = run_sequence(seq, models, detector, cache, sc, ref=ref)
                for m in methods:
                    s = score(r["labels"], r["raw"][m])
                    move_rows.append([f"{camera}/{seq['condition']}", "registered" if sc.endswith("registered") else "fixed map",
                                      f"{m}: {METHODS[m]}", 100 * np.nanmean(r["matched"]),
                                      100 * np.nanmean(r["map_iou"]), s["acc"], s["bal_acc"]])
    if move_rows:
        header = ["sequence", "map", "method", "matched %", "map IoU %", "Acc", "BalAcc"]
        print_table(f"[5] Real camera move: map from the first day ({ref_frame['time']:%Y-%m-%d}) "
                    "applied to later days", header, move_rows)
        write_csv(os.path.join(OUTPUT_ROOT, "camera_move.csv"), header, move_rows)

    # 6) unseen camera: models trained on the other cameras, pooled thresholds
    groups = [f"{r['camera']}\n{r['condition']}" for r in clean_results]
    bars = {m: [row["bal_acc"] for row in summary if row["method"] == m] for m in methods}
    for seq in unseen:
        r = run_sequence(seq, models, detector, cache, "clean")
        rows = []
        for m in methods:
            s = score(r["labels"], r["raw"][m])
            rows.append([f"{m}: {METHODS[m]}", s["acc"], s["bal_acc"], s["f1"]])
            bars[m].append(s["bal_acc"])
            summary.append({"camera": seq["camera"], "condition": seq["condition"], "method": m,
                            "frames": len(seq["frames"]), "spaces": int((r["labels"] >= 0).sum()),
                            "occ_pct": 100 * (r["labels"] == 1).mean(), "acc": s["acc"],
                            "bal_acc": s["bal_acc"], "f1": s["f1"], "bal_acc_tracked": np.nan,
                            "free_mae_raw": np.nan, "free_mae_tracked": np.nan, "changes_gt": "",
                            "changes_raw": "", "changes_tracked": ""})
        groups.append(f"{seq['camera']}\n(unseen)")
        print_table(f"[6] Unseen camera {seq['camera']} ({len(seq['frames'])} frames from different days; "
                    "trained only on " + ", ".join(calib) + ")", ["method", "Acc", "BalAcc", "F1"], rows)

    keys = ["camera", "condition", "method", "frames", "spaces", "occ_pct", "acc", "bal_acc", "f1",
            "bal_acc_tracked", "free_mae_raw", "free_mae_tracked", "changes_gt", "changes_raw", "changes_tracked"]
    write_csv(os.path.join(OUTPUT_ROOT, "summary_methods.csv"), keys, [[r[k] for k in keys] for r in summary])
    plot_grouped_bars(os.path.join(OUTPUT_ROOT, "summary_balacc.png"),
                      "Balanced accuracy per camera and weather", groups, bars)
    print(f"\nDone in {time.time() - t_start:.0f}s. Outputs: {os.path.abspath(OUTPUT_ROOT)}")


if __name__ == "__main__":
    main()
