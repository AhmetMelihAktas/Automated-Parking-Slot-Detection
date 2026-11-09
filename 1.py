import os
import glob
import xml.etree.ElementTree as ET
import cv2
import numpy as np
from shapely.geometry import Polygon, box
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_score,
                             recall_score, f1_score)
from ultralytics import YOLO


def parse_pklot_xml(xml_path):
    """
    Parse a PKLot XML annotation file to extract parking space polygons and ground-truth occupancy.
    Supports both <contour> points and <rotatedRect> definitions.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    spaces = []

    for space in root.findall("space"):
        space_id = int(space.attrib.get("id", 0))
        # Skip spaces without a valid 'occupied' label (sometimes unlabeled in raw data)
        if "occupied" not in space.attrib:
            continue
        occupied = int(space.attrib["occupied"])

        points = []
        contour = space.find("contour")
        if contour is not None:
            for pt in contour.findall("point"):
                x = int(float(pt.attrib["x"]))
                y = int(float(pt.attrib["y"]))
                points.append([x, y])

        # Fallback to rotatedRect if contour is empty
        if len(points) < 3:
            rot_rect = space.find("rotatedRect")
            if rot_rect is not None:
                cx = float(rot_rect.find("center").attrib["x"])
                cy = float(rot_rect.find("center").attrib["y"])
                w = float(rot_rect.find("size").attrib["w"])
                h = float(rot_rect.find("size").attrib["h"])
                angle = float(rot_rect.find("angle").attrib["d"])
                rect_pts = cv2.boxPoints(((cx, cy), (w, h), angle))
                points = np.int32(rect_pts).tolist()

        if len(points) >= 3:
            spaces.append({
                "id": space_id,
                "occupied": occupied,
                "polygon": np.array(points, dtype=np.int32)
            })

    return spaces


def preprocess_image(image_bgr):
    """
    Apply illumination normalization using CLAHE on the L-channel in LAB color space,
    followed by Gaussian smoothing to reduce high-frequency surface noise.
    """
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    # Apply CLAHE to normalize shadows and illumination variations
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_equalized = clahe.apply(l_channel)

    merged_lab = cv2.merge((l_equalized, a_channel, b_channel))
    enhanced_bgr = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)

    # Apply slight Gaussian blur for noise reduction before edge/feature extraction
    blurred_bgr = cv2.GaussianBlur(enhanced_bgr, (3, 3), 0)
    return enhanced_bgr, blurred_bgr


def compute_edge_densities(preprocessed_bgr, spaces):
    """
    Computes the Canny edge pixel density (edge pixels / polygon area) inside each parking polygon ROI.
    Returns (densities, edge_map).
    """
    gray = cv2.cvtColor(preprocessed_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)

    densities = []
    for sp in spaces:
        poly = sp["polygon"]
        x, y, w, h = cv2.boundingRect(poly)

        # Crop bounding box around the polygon for fast processing
        roi_edges = edges[y:y+h, x:x+w]
        mask = np.zeros((h, w), dtype=np.uint8)
        shifted_poly = poly - [x, y]
        cv2.fillPoly(mask, [shifted_poly], 255)

        # Count edge pixels strictly inside the polygon mask
        masked_edges = cv2.bitwise_and(roi_edges, roi_edges, mask=mask)
        poly_area = np.count_nonzero(mask)
        edge_pixels = np.count_nonzero(masked_edges)

        edge_density = (edge_pixels / poly_area) if poly_area > 0 else 0.0
        densities.append(edge_density)

    return densities, edges


def predict_space_based_canny(preprocessed_bgr, spaces, edge_threshold=0.08):
    """
    Geometric Space-Based Baseline:
    Computes Canny edge pixel density inside each parking polygon ROI.
    If edge density exceeds edge_threshold, the spot is classified as occupied (1).
    """
    densities, edges = compute_edge_densities(preprocessed_bgr, spaces)
    predictions = [1 if d >= edge_threshold else 0 for d in densities]
    return predictions, edges


def calibrate_edge_threshold(calib_dir, max_images=30):
    """
    Choose the Canny edge-density threshold on a separate calibration set (dates not used for testing).
    The threshold that maximizes balanced accuracy = (TPR + TNR) / 2 is selected, so the class
    imbalance (most spots are occupied) cannot push the threshold towards "everything occupied".
    Returns (threshold, balanced_accuracy_in_percent), or (None, None) if no data.
    """
    xml_files = sorted(glob.glob(os.path.join(calib_dir, "**", "*.xml"), recursive=True))[:max_images]
    densities, labels = [], []
    for xml_path in xml_files:
        img_path = os.path.splitext(xml_path)[0] + ".jpg"
        if not os.path.exists(img_path):
            continue
        spaces = parse_pklot_xml(xml_path)
        if not spaces:
            continue
        _, preprocessed_bgr = preprocess_image(cv2.imread(img_path))
        d, _ = compute_edge_densities(preprocessed_bgr, spaces)
        densities.extend(d)
        labels.extend(sp["occupied"] for sp in spaces)

    densities, labels = np.array(densities), np.array(labels)
    if len(labels) == 0 or labels.min() == labels.max():
        return None, None

    candidates = np.linspace(0.0, densities.max(), 500)
    preds = densities[None, :] >= candidates[:, None]           # (n_thresholds, n_spots)
    tpr = preds[:, labels == 1].mean(axis=1)
    tnr = (~preds[:, labels == 0]).mean(axis=1)
    bal_acc = (tpr + tnr) / 2
    best = int(np.argmax(bal_acc))
    return float(candidates[best]), float(bal_acc[best] * 100)


def predict_vehicle_detection_yolo(image_bgr, spaces, yolo_model, iou_threshold=0.25):
    """
    Vehicle-Detection Baseline:
    Detects vehicles (car, motorcycle, bus, truck) using YOLOv8 and calculates
    the intersection ratio (IoU) between detected bounding boxes and each parking polygon.
    """
    # COCO classes for vehicles: 2=car, 3=motorcycle, 5=bus, 7=truck
    results = yolo_model.predict(image_bgr, classes=[2, 3, 5, 7], conf=0.25, verbose=False)[0]
    detected_boxes = []

    if results.boxes is not None:
        for b in results.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = b
            detected_boxes.append(box(x1, y1, x2, y2))

    predictions = []
    for sp in spaces:
        poly_shapely = Polygon(sp["polygon"])
        if not poly_shapely.is_valid:
            poly_shapely = poly_shapely.buffer(0)

        max_overlap = 0.0
        for det_box in detected_boxes:
            if poly_shapely.intersects(det_box):
                intersection_area = poly_shapely.intersection(det_box).area
                # Overlap ratio relative to the parking space polygon area
                overlap_ratio = intersection_area / poly_shapely.area if poly_shapely.area > 0 else 0.0
                if overlap_ratio > max_overlap:
                    max_overlap = overlap_ratio

        pred_label = 1 if max_overlap >= iou_threshold else 0
        predictions.append(pred_label)

    return predictions, detected_boxes


def draw_parking_results(image_bgr, spaces, predictions, title_text=""):
    """
    Draw parking polygons colored by predicted occupancy:
    Green = Available (0), Red = Occupied (1).
    """
    canvas = image_bgr.copy()
    occupied_count = sum(predictions)
    total_count = len(predictions)

    for sp, pred in zip(spaces, predictions):
        color = (0, 0, 255) if pred == 1 else (0, 255, 0)
        cv2.polylines(canvas, [sp["polygon"]], isClosed=True, color=color, thickness=2)

    # Draw availability statistics banner at the top-left corner
    avail_count = total_count - occupied_count
    stats_str = f"{title_text} | Available: {avail_count}/{total_count} ({100.0 * avail_count / max(total_count, 1):.1f}%)"
    cv2.rectangle(canvas, (10, 10), (620, 50), (0, 0, 0), -1)
    cv2.putText(canvas, stats_str, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return canvas


def compute_metrics(y_true, y_pred):
    """Return accuracy / balanced accuracy / precision / recall / F1 (in %) for binary occupancy labels."""
    return {
        "acc": accuracy_score(y_true, y_pred) * 100,
        "bal_acc": balanced_accuracy_score(y_true, y_pred) * 100,
        "prec": precision_score(y_true, y_pred, zero_division=0) * 100,
        "rec": recall_score(y_true, y_pred, zero_division=0) * 100,
        "f1": f1_score(y_true, y_pred, zero_division=0) * 100,
    }


def evaluate_dataset_subset(data_dir, output_dir="progress_outputs", max_images=50, yolo_model=None,
                            edge_threshold=0.08):
    """
    Run both baselines on a subset of PKLot images, compute metrics, and save visual figures.
    Returns {"n_images", "n_spots", "space": metrics, "yolo": metrics}, or None if no data.
    """
    os.makedirs(output_dir, exist_ok=True)
    xml_files = sorted(glob.glob(os.path.join(data_dir, "**", "*.xml"), recursive=True))[:max_images]

    if not xml_files:
        print(f"No XML files found in {data_dir}. Please check the dataset path.")
        return None

    if yolo_model is None:
        yolo_model = YOLO("yolov8n.pt")

    y_true_all = []
    y_pred_space_all = []
    y_pred_yolo_all = []
    n_images = 0
    figures_saved = False

    for xml_path in xml_files:
        img_path = os.path.splitext(xml_path)[0] + ".jpg"
        if not os.path.exists(img_path):
            continue

        image_bgr = cv2.imread(img_path)
        spaces = parse_pklot_xml(xml_path)
        if not spaces:
            continue

        enhanced_bgr, preprocessed_bgr = preprocess_image(image_bgr)
        preds_space, edges = predict_space_based_canny(preprocessed_bgr, spaces, edge_threshold=edge_threshold)
        preds_yolo, _ = predict_vehicle_detection_yolo(enhanced_bgr, spaces, yolo_model, iou_threshold=0.25)

        y_true = [sp["occupied"] for sp in spaces]
        y_true_all.extend(y_true)
        y_pred_space_all.extend(preds_space)
        y_pred_yolo_all.extend(preds_yolo)
        n_images += 1

        # Save visual figures for the first sample image to include in the Progress Report
        if not figures_saved:
            figures_saved = True
            vis_space = draw_parking_results(image_bgr, spaces, preds_space, "Space-Based (Canny)")
            vis_yolo = draw_parking_results(image_bgr, spaces, preds_yolo, "Vehicle-Detection (YOLOv8)")

            cv2.imwrite(os.path.join(output_dir, "1_original.jpg"), image_bgr)
            cv2.imwrite(os.path.join(output_dir, "2_preprocessed_clahe.jpg"), enhanced_bgr)
            cv2.imwrite(os.path.join(output_dir, "3_canny_edges.jpg"), edges)
            cv2.imwrite(os.path.join(output_dir, "4_space_based_result.jpg"), vis_space)
            cv2.imwrite(os.path.join(output_dir, "5_yolo_based_result.jpg"), vis_yolo)

    if not y_true_all:
        print(f"No valid annotated spaces found in {data_dir}.")
        return None

    # Compute and print quantitative metrics for this subset
    results = {"n_images": n_images, "n_spots": len(y_true_all),
               "occ_pct": 100.0 * sum(y_true_all) / len(y_true_all)}
    for key, name, preds in [("space", "Space-Based (Canny Baseline)", y_pred_space_all),
                             ("yolo", "Vehicle-Detection (YOLOv8 Baseline)", y_pred_yolo_all)]:
        m = compute_metrics(y_true_all, preds)
        results[key] = m
        print(f"=== {name} ===")
        print(f"Total Evaluated Spots: {len(y_true_all)}")
        print(f"Accuracy : {m['acc']:.2f}% | Balanced Acc: {m['bal_acc']:.2f}% | Precision: {m['prec']:.2f}% | "
              f"Recall: {m['rec']:.2f}% | F1-Score: {m['f1']:.2f}%\n")
    return results


def print_summary_table(rows):
    """Print one line per (camera, weather) condition with both baselines side by side."""
    header = (f"{'Camera':<8}{'Weather':<8}{'Imgs':>5}{'Spots':>7}{'Occ%':>6} | "
              f"{'Thr':>6}{'Acc':>7}{'BalAcc':>8}{'F1':>7} | {'Acc':>7}{'BalAcc':>8}{'F1':>7}")
    group = f"{'':<34} | {'Canny (space-based)':^28} | {'YOLOv8 (vehicle)':^22}"
    print("=" * len(header))
    print(group)
    print(header)
    print("-" * len(header))
    for camera, weather, thr, r in rows:
        print(f"{camera:<8}{weather:<8}{r['n_images']:>5}{r['n_spots']:>7}{r['occ_pct']:>6.1f} | "
              f"{thr:>6.3f}{r['space']['acc']:>7.2f}{r['space']['bal_acc']:>8.2f}{r['space']['f1']:>7.2f} | "
              f"{r['yolo']['acc']:>7.2f}{r['yolo']['bal_acc']:>8.2f}{r['yolo']['f1']:>7.2f}")
    print("=" * len(header))
    print("Occ% = share of occupied spots (an 'always occupied' guess reaches this accuracy).")
    print("BalAcc = (recall on occupied + recall on empty) / 2; 50% means no better than guessing.")


if __name__ == "__main__":
    # Expected layout: PKLot/<Camera>/<Weather>/<date>/*.jpg + *.xml
    # (created by prepare_dataset.py). Each Camera/Weather folder is evaluated separately.
    # Canny thresholds are calibrated per camera on PKLot_calibration/<Camera>/ (other dates
    # than the test folders), so test images never influence the threshold.
    DATASET_ROOT = "./PKLot"
    CALIBRATION_ROOT = "./PKLot_calibration"
    OUTPUT_ROOT = "progress_outputs"
    MAX_IMAGES_PER_CONDITION = 25
    DEFAULT_EDGE_THRESHOLD = 0.08

    # 1) Calibrate the Canny edge-density threshold for each camera
    edge_thresholds = {}
    for camera_dir in sorted(glob.glob(os.path.join(DATASET_ROOT, "*"))):
        if not os.path.isdir(camera_dir):
            continue
        camera = os.path.basename(camera_dir)
        thr, bal = calibrate_edge_threshold(os.path.join(CALIBRATION_ROOT, camera))
        if thr is None:
            thr = DEFAULT_EDGE_THRESHOLD
            print(f"[calibration] {camera}: no calibration data, using default threshold {thr:.3f}")
        else:
            print(f"[calibration] {camera}: edge_threshold = {thr:.3f} "
                  f"(balanced accuracy on calibration set: {bal:.2f}%)")
        edge_thresholds[camera] = thr

    # 2) Evaluate every Camera/Weather condition separately
    yolo_model = YOLO("yolov8n.pt")
    summary_rows = []

    for condition_dir in sorted(glob.glob(os.path.join(DATASET_ROOT, "*", "*"))):
        if not os.path.isdir(condition_dir):
            continue
        camera = os.path.basename(os.path.dirname(condition_dir))
        weather = os.path.basename(condition_dir)
        thr = edge_thresholds.get(camera, DEFAULT_EDGE_THRESHOLD)
        print(f"\n##### {camera} / {weather} (Canny threshold {thr:.3f}) #####")
        res = evaluate_dataset_subset(condition_dir,
                                      output_dir=os.path.join(OUTPUT_ROOT, f"{camera}_{weather}"),
                                      max_images=MAX_IMAGES_PER_CONDITION,
                                      yolo_model=yolo_model,
                                      edge_threshold=thr)
        if res is not None:
            summary_rows.append((camera, weather, thr, res))

    if summary_rows:
        print_summary_table(summary_rows)
    else:
        print(f"No Camera/Weather folders with XML files found under {DATASET_ROOT}.")