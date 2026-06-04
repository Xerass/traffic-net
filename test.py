import cv2
import numpy as np
from collections import defaultdict, deque
from vidgear.gears import CamGear
from ultralytics import YOLO

# --- Config ---
DENSITY_THRESHOLD = 8
VELOCITY_THRESHOLD = 5.0
TRAIL_LENGTH = 30
STATUS_INTERVAL = 5.0  # seconds between traffic status updates
CALIBRATE = False

# Intersection line (normalized 0-1, calibrated from live feed)
LINE_START = (0.3961, 0.1847)
LINE_END = (0.1289, 0.3847)

model_pth = "runs/trafficnet_cloud/weights/best.pt"
model = YOLO(model_pth)

yt_url = "https://www.youtube.com/watch?v=6dp-bvQ7RWo"
options = {"STREAM_RESOLUTION": "720p"}
stream = CamGear(source=yt_url, stream_mode=True, logging=True, **options).start()

# --- Calibration mode ---
if CALIBRATE:
    calib_points = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(calib_points) < 2:
            calib_points.append((x, y))
            print(f"  Point {len(calib_points)}: pixel ({x}, {y})")

    frame = stream.read()
    if frame is None:
        raise RuntimeError("Could not read first frame")
    h, w = frame.shape[:2]

    print("\n=== CALIBRATION MODE ===")
    print("Click 2 points on the frame to define the intersection line.")
    print("Press 'r' to reset, 'q' to confirm.\n")

    cv2.imshow("TrafficNet - Calibrate", frame)
    cv2.setMouseCallback("TrafficNet - Calibrate", on_mouse)

    while True:
        display = frame.copy()
        for i, pt in enumerate(calib_points):
            cv2.circle(display, pt, 6, (0, 255, 255), -1)
            cv2.putText(display, f"P{i+1}", (pt[0]+10, pt[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        if len(calib_points) == 2:
            cv2.line(display, calib_points[0], calib_points[1], (0, 255, 255), 2)

        cv2.imshow("TrafficNet - Calibrate", display)
        key = cv2.waitKey(30) & 0xFF

        if key == ord("r"):
            calib_points.clear()
            print("  Reset points.")
        elif key == ord("q") and len(calib_points) == 2:
            break

    LINE_START = (round(calib_points[0][0] / w, 4), round(calib_points[0][1] / h, 4))
    LINE_END = (round(calib_points[1][0] / w, 4), round(calib_points[1][1] / h, 4))

    print(f"\n>>> Paste these into your config:")
    print(f'LINE_START = {LINE_START}')
    print(f'LINE_END   = {LINE_END}')
    print(f"=========================\n")

    cv2.destroyWindow("TrafficNet - Calibrate")

# --- Tracking state ---
import time

track_history: dict[int, deque] = defaultdict(lambda: deque(maxlen=TRAIL_LENGTH))
crossed_ids: set[int] = set()
crossing_count = 0
prev_side: dict[int, float] = {}

# --- Smoothed traffic status ---
status_label = "CLEAR"
status_color = (0, 255, 0)
status_last_update = time.time()
density_samples: list[int] = []
velocity_samples: list[float] = []


def cross_product_sign(a, b, p):
    """Which side of line AB is point P on? Positive = left, negative = right."""
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def classify_traffic(density, avg_velocity):
    if density >= DENSITY_THRESHOLD and avg_velocity <= VELOCITY_THRESHOLD:
        return "HEAVY TRAFFIC", (0, 0, 255)
    elif density >= DENSITY_THRESHOLD:
        return "DENSE / MOVING", (0, 165, 255)
    elif avg_velocity <= VELOCITY_THRESHOLD and density > 0:
        return "SLOW / SPARSE", (0, 255, 255)
    else:
        return "CLEAR", (0, 255, 0)


while True:
    frame = stream.read()
    if frame is None:
        break

    h, w = frame.shape[:2]
    line_pt1 = (int(LINE_START[0] * w), int(LINE_START[1] * h))
    line_pt2 = (int(LINE_END[0] * w), int(LINE_END[1] * h))

    # ByteTrack via ultralytics — replaces manual centroid matching
    results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)

    for r in results:
        annotated = r.plot()
        boxes = r.boxes

        if boxes.id is not None:
            ids = boxes.id.int().cpu().tolist()
            xywh = boxes.xywh.cpu().numpy()

            velocities = []

            for track_id, box in zip(ids, xywh):
                cx, cy = float(box[0]), float(box[1])
                track = track_history[track_id]
                track.append((cx, cy))

                # Velocity: displacement from previous position
                if len(track) >= 2:
                    dx = track[-1][0] - track[-2][0]
                    dy = track[-1][1] - track[-2][1]
                    velocities.append(np.sqrt(dx * dx + dy * dy))

                # --- Intersection crossing detection ---
                side = cross_product_sign(line_pt1, line_pt2, (cx, cy))
                if track_id in prev_side:
                    if prev_side[track_id] * side < 0 and track_id not in crossed_ids:
                        crossed_ids.add(track_id)
                        crossing_count += 1
                prev_side[track_id] = side

                # Draw trail
                points = np.array(track, dtype=np.int32).reshape(-1, 1, 2)
                if len(points) > 1:
                    cv2.polylines(annotated, [points], False, (230, 160, 0), 2)

            density = len(ids)
            avg_velocity = float(np.mean(velocities)) if velocities else 0.0
        else:
            density = 0
            avg_velocity = 0.0

        density_samples.append(density)
        velocity_samples.append(avg_velocity)

        now = time.time()
        if now - status_last_update >= STATUS_INTERVAL:
            avg_d = float(np.mean(density_samples))
            avg_v = float(np.mean(velocity_samples))
            status_label, status_color = classify_traffic(avg_d, avg_v)
            density_samples.clear()
            velocity_samples.clear()
            status_last_update = now

        label, color = status_label, status_color

        # Draw intersection line
        cv2.line(annotated, line_pt1, line_pt2, (0, 255, 255), 2)
        cv2.putText(annotated, "INTERSECTION", (line_pt1[0] + 8, line_pt1[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # HUD
        cv2.rectangle(annotated, (0, h - 100), (380, h), (30, 30, 30), -1)
        cv2.putText(annotated, f"Vehicles: {density}  Avg vel: {avg_velocity:.1f} px/f",
                    (8, h - 72), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
        cv2.putText(annotated, f"Crossed intersection: {crossing_count}",
                    (8, h - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
        cv2.putText(annotated, label,
                    (8, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)

        cv2.imshow("TrafficNet", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

stream.stop()
cv2.destroyAllWindows()
