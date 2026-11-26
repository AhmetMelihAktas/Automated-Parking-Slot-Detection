# Smart Parking Occupancy and Parking Analytics

CENG 391 Introduction to Image Understanding, term project of group G13.

A fixed-camera system that finds the parking spaces in the image, decides for every space whether it is free or
occupied, tracks occupancy changes over time and reports availability statistics. It compares geometric
space-based classification with vehicle detection, and it is tested for robustness to shadows, weather, partial
occlusion and camera viewpoint changes. All experiments use the PKLot dataset.

Four occupancy methods are compared:

| Method | Idea |
|---|---|
| A | Canny edge density inside the space, threshold calibrated per camera (geometric baseline) |
| B | 123 hand-crafted features per space (texture, gradients, edges, frequency, corners, colour, HOG) + RBF-SVM |
| C | YOLOv8 vehicle detection (full frame + 2x2 tiles), one-to-one box-space assignment |
| D | Hybrid: B's features and C's detection scores in one SVM |

A per-space hidden Markov model filter turns the frame decisions into stable states and change events, and a
SIFT + RANSAC homography re-projects the space map when the camera moves.

## Repository contents

| Path | Content |
|---|---|
| `Smart_Parking_Occupancy_and_Parking_Analytics.py` | Main program: training, evaluation, tracking, statistics, robustness experiments, figures |
| `prepare_dataset.py` | Builds the data folders below from the Roboflow PKLot export |
| `space_map_tool.py` | Review tool for the complete space maps (see below) |
| `PKLot/` | Test sequences: 2 cameras x 3 weather conditions, one day and 25 frames each |
| `PKLot_calibration/` | Training / threshold calibration frames, 30 per camera, from other days |
| `PKLot_unseen/` | 25 frames of a third camera (UFPR05) that is never used for training |
| `space_maps/` | Space maps per camera position: automatic proposals (`*_proposed.xml`) and hand-reviewed maps |
| `yolov8n.pt` | YOLOv8n weights pretrained on COCO (Ultralytics) |
| `outputs/` | Results of the last run (CSV tables and figures) |
| `CENG391_Project_Report_Template.pdf`, `literature.bib`, `chicago.bst`, `iyte_logo.png` | Course report template files |

## Setup

```
python -m venv .venv
.venv\Scripts\activate          # Windows   (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
```

## Running

Paths are relative to the scripts, so they can be started from any folder.

**1. Prepare the data (optional).** The prepared folders `PKLot/`, `PKLot_calibration/` and `PKLot_unseen/` are
already in the repository. To rebuild them, download the PKLot export from Roboflow in COCO format, unpack it next
to this folder as `../archive` (with `train/`, `valid/`, `test/` and their `_annotations.coco.json`), and run

```
python prepare_dataset.py
python prepare_dataset.py --archive ../archive --n 25 --n-calib 30 --n-unseen 25    # same, with the defaults written out
```

The export has no camera or weather folders, so the camera is recognised from the number of annotated spaces
(PUCPR 100, UFPR04 28, UFPR05 40) and one day per weather condition is chosen in the script. Calibration frames are
taken only from days that are not used for testing.

**2. Run the system.**

```
python Smart_Parking_Occupancy_and_Parking_Analytics.py
```

It takes about 10 minutes on a CPU. It writes new space-map proposals to `space_maps/` (a reviewed map
`space_maps/<position>.xml` is used if it exists) and all results to `outputs/`:

| Output | Content |
|---|---|
| `summary_methods.csv`, `summary_balacc.png` | Accuracy, balanced accuracy and F1 of every method per sequence and on the unseen camera |
| `robustness.csv`, `robustness_balacc.png` | Synthetic shadows, occlusion and camera shake (with and without registration) |
| `camera_move.csv` | Real UFPR04 camera move, fixed map vs. SIFT + RANSAC registration |
| `space_discovery.csv` | Spaces discovered from persistent detections over one day |
| `full_lot.csv`, `space_map_eval.csv` | Full-lot availability and the evaluation of the discovery against the reviewed maps |
| `<Camera>_<Weather>/` | Per sequence: timeline, change events, per-space statistics, dashboard, example frames and full-lot images |

**3. Review the space maps (optional).**

```
python space_map_tool.py                      # every camera position, one after the other
python space_map_tool.py UFPR04_2012-12-07    # one camera position
```

| Input | Action |
|---|---|
| left-drag | add a space |
| right-click in a box | delete an added or proposed space (PKLot spaces cannot be changed) |
| `f` / `b` / `m` | next / previous real frame, back to the median image |
| `h` | hide / show the boxes |
| `z` | undo |
| `s` | save |
| `n` | save and go to the next camera position |
| `q` or `Esc` | save and quit |

The reviewed map is saved as `space_maps/<position>.xml` and used by the next run of the main program.

## Complete space maps

PKLot labels only part of each lot (on PUCPR, 100 of the 281 spaces in the reviewed map). The main program
therefore proposes the missing spaces from YOLOv8 detections pooled over every day of a camera position, and the
proposals are corrected by hand with `space_map_tool.py`. The added spaces are not PKLot labels and have no occupancy
ground truth: they are counted in the availability statistics of the whole lot, but never in any accuracy figure.
The original PKLot annotation files are not modified.

## Data and license

PKLot: P. R. L. de Almeida, L. S. Oliveira, A. S. Britto Jr., E. J. Silva Jr. and A. L. Koerich, "PKLot - A robust
dataset for parking lot classification", Expert Systems with Applications 42(11), 2015. The frames in this
repository come from the Roboflow export of PKLot (https://public.roboflow.com/object-detection/pklot, CC BY 4.0),
which stores the frames as 640x640 images with axis-aligned space boxes.
