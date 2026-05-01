#needed for video input
import cv2
import numpy as np

#streamlines the yt feed -> input pipeline
from vidgear.gears import CamGear

#for the model
from ultralytics import YOLO

# --- Traffic analysis thresholds (baseline, not trained) ---
DENSITY_THRESHOLD = 8       # min cars in frame to be considered "dense"
VELOCITY_THRESHOLD = 5.0    # max avg px/frame movement to be considered "slow"

model_pth = "runs/trafficnet_cloud/weights/best.pt"
model = YOLO(model_pth)

yt_url = "https://www.youtube.com/watch?v=6dp-bvQ7RWo"

options = {"STREAM_RESOLUTION": "720p"}
stream = CamGear(source=yt_url, stream_mode=True, logging=True, **options).start()

# prev_centroids: list of (cx, cy) from the last frame
prev_centroids = []


def get_centroids(boxes):
    """Return (cx, cy) for each bounding box."""
    centroids = []
    for box in boxes.xyxy.cpu().numpy():
        cx = (box[0] + box[2]) / 2
        cy = (box[1] + box[3]) / 2
        centroids.append((cx, cy))
    return centroids


def match_and_compute_velocities(prev, curr):
    """
    Greedy nearest-centroid matching between prev and curr centroids.
    Returns list of pixel displacements for matched pairs.
    """
    if not prev or not curr:
        return []

    prev = np.array(prev)
    curr = np.array(curr)
    velocities = []
    used = set()

    for p in prev:
        dists = np.linalg.norm(curr - p, axis=1)
        idx = int(np.argmin(dists))
        if idx not in used:
            velocities.append(float(dists[idx]))
            used.add(idx)

    return velocities


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

    results = model(frame, stream=True)

    for r in results:
        annotated_frame = r.plot()

        curr_centroids = get_centroids(r.boxes)
        density = len(curr_centroids)

        velocities = match_and_compute_velocities(prev_centroids, curr_centroids)
        avg_velocity = float(np.mean(velocities)) if velocities else 0.0

        prev_centroids = curr_centroids

        label, color = classify_traffic(density, avg_velocity)

        # HUD overlay
        h, w = annotated_frame.shape[:2]
        cv2.rectangle(annotated_frame, (0, h - 80), (340, h), (30, 30, 30), -1)
        cv2.putText(annotated_frame, f"Cars: {density}  Avg vel: {avg_velocity:.1f} px/f",
                    (8, h - 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
        cv2.putText(annotated_frame, label,
                    (8, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)

        cv2.imshow("TrafficNet", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

stream.stop()
cv2.destroyAllWindows()
