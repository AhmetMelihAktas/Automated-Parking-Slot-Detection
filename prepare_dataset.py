"""
archive/ (Roboflow PKLot export: train/valid/test + _annotations.coco.json)
    -> PKLot/<Camera>/<Weather>/<date>/<timestamp>.jpg + .xml  (PKLot-style XML)

Roboflow export has no camera/weather folders, so:
  * Camera  : detected from the number of annotated spaces per image
              (PUCPR = 100, UFPR04 = 28, UFPR05 = 40).
  * Weather : chosen by date after visual inspection of the frames
              (Sunny = hard shadows, Cloudy = diffuse light + dry asphalt,
               Rainy = dark wet asphalt / reflections / drops on the lens).
  * XML     : generated from COCO bounding boxes (axis-aligned, in 640x640 coords),
              written in the same format that parse_pklot_xml() in 1.py reads.

Also writes a separate calibration set
    -> PKLot_calibration/<Camera>/<date>/...
taken only from dates that are NOT used in the test folders. 1.py uses it to choose
the Canny edge-density threshold per camera, so the test images never influence the threshold.

Usage (from the Project folder):
    python prepare_dataset.py
    python prepare_dataset.py --archive ../archive --out ./PKLot --n 25 --n-calib 30
"""
import argparse
import collections
import json
import os
import shutil
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))

CAMERA_SPOTS = {"PUCPR": 100, "UFPR04": 28, "UFPR05": 40}

# camera -> weather -> (date, start HH:MM, end HH:MM)
SELECTION = {
    "UFPR04": {
        "Sunny":  ("2012-12-21", "09:00", "15:30"),
        "Cloudy": ("2013-01-22", "10:00", "18:30"),
        "Rainy":  ("2012-12-07", "16:00", "19:30"),
    },
    "PUCPR": {
        "Sunny":  ("2012-09-20", "09:00", "15:00"),
        "Cloudy": ("2012-10-16", "08:30", "17:00"),
        "Rainy":  ("2012-09-19", "08:00", "17:00"),
    },
}

# Calibration images: evenly spread over all other dates of the camera, daytime only
CALIBRATION_WINDOW = ("09:00", "17:00")

# Camera never used for training/calibration (tests viewpoint generalisation in 2.py):
# camera -> daytime window; images are spread evenly over all of its dates
UNSEEN = {"UFPR05": ("09:00", "17:00")}

# COCO category ids in the Roboflow export
CAT_EMPTY, CAT_OCCUPIED = 1, 2


def load_archive(archive_dir):
    """Return {clean_name: record} for every image in train/valid/test."""
    records = {}
    for split in ("train", "valid", "test"):
        json_path = os.path.join(archive_dir, split, "_annotations.coco.json")
        with open(json_path, encoding="utf-8") as f:
            coco = json.load(f)
        anns = collections.defaultdict(list)
        for a in coco["annotations"]:
            anns[a["image_id"]].append(a)
        for im in coco["images"]:
            fn = im["file_name"]                      # 2012-12-07_16_42_25_jpg.rf.<hash>.jpg
            clean = fn.split("_jpg.rf.")[0]           # 2012-12-07_16_42_25
            records[clean] = {
                "src": os.path.join(archive_dir, split, fn),
                "name": clean,
                "date": clean[:10],
                "time": clean[11:16].replace("_", ":"),
                "anns": anns[im["id"]],
            }
    return records


def write_pklot_xml(record, xml_path, camera_id):
    root = ET.Element("parking", id=camera_id.lower())
    space_id = 0
    # sort spaces top-to-bottom, left-to-right so ids are stable between images
    for a in sorted(record["anns"], key=lambda a: (round(a["bbox"][1] / 20), a["bbox"][0])):
        if a["category_id"] not in (CAT_EMPTY, CAT_OCCUPIED):
            continue
        space_id += 1
        x, y, w, h = a["bbox"]
        sp = ET.SubElement(root, "space", id=str(space_id),
                           occupied="1" if a["category_id"] == CAT_OCCUPIED else "0")
        rr = ET.SubElement(sp, "rotatedRect")
        ET.SubElement(rr, "center", x=f"{x + w / 2:.1f}", y=f"{y + h / 2:.1f}")
        ET.SubElement(rr, "size", w=f"{w:.1f}", h=f"{h:.1f}")
        ET.SubElement(rr, "angle", d="0")
        ct = ET.SubElement(sp, "contour")
        for px, py in ((x, y), (x + w, y), (x + w, y + h), (x, y + h)):
            ET.SubElement(ct, "point", x=str(int(round(px))), y=str(int(round(py))))
    ET.indent(root)
    ET.ElementTree(root).write(xml_path, encoding="utf-8", xml_declaration=True)
    return space_id


def evenly_spaced(items, n):
    if len(items) <= n:
        return items
    return [items[round(i * (len(items) - 1) / (n - 1))] for i in range(n)]


def export(records, out_dir_fn, camera):
    """Copy images + write XMLs. out_dir_fn(record) -> target folder. Returns (spaces, occupied)."""
    occ = total = 0
    for r in records:
        out_dir = out_dir_fn(r)
        os.makedirs(out_dir, exist_ok=True)
        shutil.copy2(r["src"], os.path.join(out_dir, r["name"] + ".jpg"))
        total += write_pklot_xml(r, os.path.join(out_dir, r["name"] + ".xml"), camera)
        occ += sum(a["category_id"] == CAT_OCCUPIED for a in r["anns"])
    return total, occ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default=os.path.join(HERE, "..", "archive"))
    ap.add_argument("--out", default=os.path.join(HERE, "PKLot"))
    ap.add_argument("--calib-out", default=os.path.join(HERE, "PKLot_calibration"))
    ap.add_argument("--n", type=int, default=25, help="images per weather folder")
    ap.add_argument("--n-calib", type=int, default=30, help="calibration images per camera")
    ap.add_argument("--unseen-out", default=os.path.join(HERE, "PKLot_unseen"))
    ap.add_argument("--n-unseen", type=int, default=25, help="images of the unseen camera")
    args = ap.parse_args()

    records = load_archive(args.archive)
    print(f"{len(records)} images in archive")

    for camera, weathers in SELECTION.items():
        n_spots = CAMERA_SPOTS[camera]
        cam_records = [r for r in records.values() if len(r["anns"]) == n_spots]
        print(f"\n{camera}: {len(cam_records)} images")

        for weather, (date, t0, t1) in weathers.items():
            pool = sorted((r for r in cam_records if r["date"] == date and t0 <= r["time"] <= t1),
                          key=lambda r: r["name"])
            chosen = evenly_spaced(pool, args.n)
            out_dir = os.path.join(args.out, camera, weather, date)
            total, occ = export(chosen, lambda r: out_dir, camera)
            print(f"  {weather:6s} {date}: {len(chosen)}/{len(pool)} images "
                  f"({chosen[0]['time']}-{chosen[-1]['time']}), "
                  f"{total} spaces, {100 * occ / max(total, 1):.0f}% occupied")

        # calibration set: other dates only, so test images never tune the threshold
        test_dates = {date for date, _, _ in weathers.values()}
        t0, t1 = CALIBRATION_WINDOW
        pool = sorted((r for r in cam_records
                       if r["date"] not in test_dates and t0 <= r["time"] <= t1),
                      key=lambda r: r["name"])
        chosen = evenly_spaced(pool, args.n_calib)
        total, occ = export(chosen, lambda r: os.path.join(args.calib_out, camera, r["date"]), camera)
        print(f"  Calib  {len(chosen)} images from {len({r['date'] for r in chosen})} other dates, "
              f"{total} spaces, {100 * occ / max(total, 1):.0f}% occupied")

    for camera, (t0, t1) in UNSEEN.items():
        n_spots = CAMERA_SPOTS[camera]
        pool = sorted((r for r in records.values()
                       if len(r["anns"]) == n_spots and t0 <= r["time"] <= t1),
                      key=lambda r: r["name"])
        chosen = evenly_spaced(pool, args.n_unseen)
        total, occ = export(chosen, lambda r: os.path.join(args.unseen_out, camera, r["date"]), camera)
        print(f"\n{camera} (unseen camera): {len(chosen)} images from "
              f"{len({r['date'] for r in chosen})} dates, {total} spaces, "
              f"{100 * occ / max(total, 1):.0f}% occupied")


if __name__ == "__main__":
    main()
