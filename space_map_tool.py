"""
CENG 391 - review tool for the complete parking-space maps (space_map_tool.py)

PKLot annotates only part of each parking lot. The main script proposes the missing spaces from vehicle
detections pooled over all days of a camera position and writes them to space_maps/<position>_proposed.xml.
This tool shows the proposals on the median image of that camera position (parked cars of many days leave
"ghosts" where the spaces are) so that they can be corrected by hand. The result is saved as
space_maps/<position>.xml; the next run of the main script uses it as the reference layout of the whole lot.

Run (paths are relative to this script, so it can be started from any folder):
    python space_map_tool.py                      every camera position, one after the other
    python space_map_tool.py UFPR04_2012-12-07    one camera position

Mouse  left-drag             add a space (draw a box around one parking space)
       right-click in a box  delete that added / proposed space (yellow PKLot spaces cannot be changed)
Keys   f / b                 next / previous real frame of this camera position (m = back to the median image)
       h                     hide / show the boxes
       z                     undo
       s                     save
       n                     save and go to the next camera position
       q or Esc              save and quit
"""
import glob
import os
import sys

import cv2
import numpy as np

import Smart_Parking_Occupancy_and_Parking_Analytics as spa

VIEW_SIZE = (1280, 720)          # PKLot frames are 1280x720; the Roboflow export stores them as 640x640
WINDOW = "space map review"
PKLOT_BGR, EXTRA_BGR, DRAW_BGR = (0, 215, 255), (255, 140, 0), (255, 0, 255)
HEADER = 44                      # text strip above the image (keeps the top row of spaces visible)


def position_frames(camera, map_polys):
    """All annotated frames (test and calibration days) on which the camera stood in this position."""
    frames = []
    for root in (spa.DATASET_ROOT, spa.CALIB_ROOT, spa.UNSEEN_ROOT):
        if os.path.isdir(os.path.join(root, camera)):
            frames += spa.load_frames(os.path.join(root, camera))
    return [f for f in sorted(frames, key=lambda f: f["time"]) if spa.same_position(map_polys, f["polys"])]


def median_image(frames, n=40):
    """Median of up to n frames spread over all days: cars move away, the parking layout stays."""
    picked = frames[::max(1, len(frames) // n)]
    return np.median(np.stack([spa.imread(f["img_path"]) for f in picked]), axis=0).astype(np.uint8)


class Editor:
    def __init__(self, name, camera, pklot, extra, frames):
        self.name, self.camera, self.pklot, self.extra, self.frames = name, camera, pklot, list(extra), frames
        self.median = median_image(frames)
        h, w = self.median.shape[:2]
        self.scale = np.array([VIEW_SIZE[0] / w, VIEW_SIZE[1] / h])
        self.view = -1                 # -1 = median image, otherwise index of a real frame
        self.show_boxes = True
        self.undo = []                 # ("add", index) or ("del", index, polygon)
        self.drag = None               # (start, current) in view coordinates while drawing
        self.dirty = False

    # ---- coordinates: polygons are stored in image (640x640) pixels, the window shows 1280x720
    def to_view(self, poly):
        return np.round(np.asarray(poly, float) * self.scale).astype(np.int32)

    def to_image(self, x, y):
        return np.array([x, y], float) / self.scale

    # ---- editing
    def add_box(self, p0, p1):
        (x0, y0), (x1, y1) = np.minimum(p0, p1), np.maximum(p0, p1)
        if x1 - x0 < 3 or y1 - y0 < 3:     # a click, not a box
            return False
        self.extra.append(spa.box_to_poly(np.round([x0, y0, x1, y1])))
        self.undo.append(("add", len(self.extra) - 1))
        self.dirty = True
        return True

    def delete_at(self, point):
        """Delete the smallest added box that contains point (image coordinates)."""
        hits = [i for i, p in enumerate(self.extra)
                if cv2.pointPolygonTest(p.reshape(-1, 1, 2).astype(np.float32), tuple(map(float, point)), False) >= 0]
        if not hits:
            return False
        i = min(hits, key=lambda k: cv2.contourArea(self.extra[k].astype(np.float32)))
        self.undo.append(("del", i, self.extra.pop(i)))
        self.dirty = True
        return True

    def undo_last(self):
        if not self.undo:
            return
        action = self.undo.pop()
        if action[0] == "add":
            self.extra.pop(action[1])
        else:
            self.extra.insert(action[1], action[2])
        self.dirty = True

    def on_mouse(self, event, x, y, flags, param=None):
        y = min(max(y - HEADER, 0), VIEW_SIZE[1] - 1)   # window -> image view coordinates
        x = min(max(x, 0), VIEW_SIZE[0] - 1)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag = ((x, y), (x, y))
        elif event == cv2.EVENT_MOUSEMOVE and self.drag is not None:
            self.drag = (self.drag[0], (x, y))
        elif event == cv2.EVENT_LBUTTONUP and self.drag is not None:
            self.add_box(self.to_image(*self.drag[0]), self.to_image(x, y))
            self.drag = None
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.delete_at(self.to_image(x, y))

    # ---- drawing
    def background(self):
        img = self.median if self.view < 0 else spa.imread(self.frames[self.view]["img_path"])
        return cv2.resize(img, VIEW_SIZE, interpolation=cv2.INTER_CUBIC)

    def render(self, bg=None):
        canvas = (self.background() if bg is None else bg).copy()
        if self.show_boxes:
            for p in self.pklot:
                cv2.polylines(canvas, [self.to_view(p)], True, PKLOT_BGR, 1)
            for p in self.extra:
                cv2.polylines(canvas, [self.to_view(p)], True, EXTRA_BGR, 2)
        if self.drag is not None:
            cv2.rectangle(canvas, self.drag[0], self.drag[1], DRAW_BGR, 2)
        view = "median of %d frames" % len(self.frames) if self.view < 0 else \
            "frame %d/%d  %s" % (self.view + 1, len(self.frames), self.frames[self.view]["time"].strftime("%Y-%m-%d %H:%M"))
        text = "%s | PKLot %d (yellow) + added %d (blue) | %s%s" % (
            self.name, len(self.pklot), len(self.extra), view, " | unsaved" if self.dirty else "")
        hint = "drag = add, right-click = delete, f/b/m = frames, h = hide, z = undo, s = save, n = next, q = quit"
        header = np.full((HEADER, canvas.shape[1], 3), 20, np.uint8)
        cv2.putText(header, text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(header, hint, (8, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (190, 190, 190), 1, cv2.LINE_AA)
        return np.vstack([header, canvas])

    def save(self):
        path = os.path.join(spa.SPACE_MAP_ROOT, self.name + ".xml")
        spa.save_space_map(path, self.camera, self.pklot, self.extra, "added")
        self.dirty = False
        print("  saved %s (%d PKLot + %d added spaces)" % (path, len(self.pklot), len(self.extra)))

    def run(self):
        """Returns False if the user wants to quit."""
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, VIEW_SIZE[0], VIEW_SIZE[1] + HEADER)
        cv2.setMouseCallback(WINDOW, self.on_mouse)
        bg, bg_view = self.background(), self.view
        while True:
            if bg_view != self.view:
                bg, bg_view = self.background(), self.view
            cv2.imshow(WINDOW, self.render(bg))
            key = cv2.waitKey(20) & 0xFF
            if key == ord("f"):
                self.view = (self.view + 1) % len(self.frames)
            elif key == ord("b"):
                self.view = (self.view - 1) % len(self.frames) if self.view >= 0 else len(self.frames) - 1
            elif key == ord("m"):
                self.view = -1
            elif key == ord("h"):
                self.show_boxes = not self.show_boxes
            elif key == ord("z"):
                self.undo_last()
            elif key == ord("s"):
                self.save()
            elif key in (ord("n"), ord("q"), 27):
                self.save()
                return key == ord("n")
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:  # window closed with the mouse
                self.save()
                return False


def load_position(name):
    """Reviewed map if it exists (continue editing), otherwise the automatic proposals."""
    reviewed = os.path.join(spa.SPACE_MAP_ROOT, name + ".xml")
    proposed = os.path.join(spa.SPACE_MAP_ROOT, name + "_proposed.xml")
    pklot, extra = spa.load_space_map(reviewed if os.path.exists(reviewed) else proposed)
    camera = name.rsplit("_", 1)[0]
    return camera, pklot, extra


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))  # space_maps/ and PKLot*/ are next to this script
    names = sorted(os.path.basename(p)[:-len("_proposed.xml")]
                   for p in glob.glob(os.path.join(spa.SPACE_MAP_ROOT, "*_proposed.xml")))
    if not names:
        print("No proposals in %s/. Run Smart_Parking_Occupancy_and_Parking_Analytics.py first." % spa.SPACE_MAP_ROOT)
        return
    if len(sys.argv) > 1:
        names = [n for n in names if n in sys.argv[1:]]
    for name in names:
        camera, pklot, extra = load_position(name)
        frames = position_frames(camera, pklot)
        print("%s: %d PKLot + %d proposed/added spaces, %d frames" % (name, len(pklot), len(extra), len(frames)))
        if not frames:
            print("  no frames of this camera position found, skipped")
            continue
        if not Editor(name, camera, pklot, extra, frames).run():
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
