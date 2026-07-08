"""Face-aware 9:16 crop positioning.

Samples a handful of frames from a clip's time range, detects faces
with OpenCV's YuNet detector (model file vendored in the repo), and
returns the horizontal offset the 9:16 crop window should use so it
centers on the dominant face instead of the frame middle.

When two faces sit far apart (side-by-side podcast layout), no 9:16
window can contain both — we pick the cluster with the most face area
(usually the closest/largest person) rather than splitting the
difference and cropping the wall between them. Returns None when no
face is found; the caller falls back to a plain center crop.
"""

import os

import cv2

try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except AttributeError:
    pass

MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_detection_yunet_2023mar.onnx")
SAMPLE_FRAMES = 8
SCORE_THRESHOLD = 0.6
# Detections whose x-centers differ by more than this fraction of the
# frame width are treated as different people, not jitter of one face.
CLUSTER_GAP_FRAC = 0.2


def compute_crop_x(video_path: str, start: float, end: float) -> tuple[int, int, int] | None:
    """Returns (crop_x, crop_w, src_h) in source pixels, or None if no
    face was detected (or the source is already 9:16 or narrower)."""
    if not os.path.exists(MODEL_PATH):
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    crop_w = int(src_h * 9 / 16)
    if crop_w <= 0 or crop_w >= src_w or src_h <= 0:
        cap.release()
        return None

    detector = cv2.FaceDetectorYN_create(
        MODEL_PATH, "", (src_w, src_h), score_threshold=SCORE_THRESHOLD
    )

    detections = []  # (x_center, face_area)
    for i in range(SAMPLE_FRAMES):
        t = start + (end - start) * (i + 0.5) / SAMPLE_FRAMES
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            continue
        _, faces = detector.detect(frame)
        if faces is None:
            continue
        for f in faces:
            x, y, fw, fh = f[0], f[1], f[2], f[3]
            detections.append((x + fw / 2.0, float(fw) * fh))
    cap.release()

    if not detections:
        return None

    # Cluster by x-center, then crop toward the cluster with the most
    # accumulated face area across the sampled frames.
    detections.sort(key=lambda d: d[0])
    clusters = [[detections[0]]]
    for det in detections[1:]:
        if det[0] - clusters[-1][-1][0] > src_w * CLUSTER_GAP_FRAC:
            clusters.append([det])
        else:
            clusters[-1].append(det)
    best = max(clusters, key=lambda c: sum(area for _, area in c))
    xs = sorted(x for x, _ in best)
    face_x = xs[len(xs) // 2]

    crop_x = int(min(max(face_x - crop_w / 2, 0), src_w - crop_w))
    return crop_x, crop_w, src_h
