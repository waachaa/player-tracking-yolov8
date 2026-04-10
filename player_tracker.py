"""
player_tracker.py
=================
Sports Player Detection & Tracking
Uses YOLOv8 (pretrained) + ByteTrack to detect and track players in any sports video.
Outputs an annotated video with bounding boxes, player IDs, and movement trails.
 
Requirements:
    pip install ultralytics opencv-python numpy
 
Usage:
    python player_tracker.py --input your_video.mp4 --output tracked_output.mp4
 
Optional flags:
    --conf 0.4          Detection confidence threshold (default: 0.4)
    --show              Display live preview window while processing
    --no-trails         Disable movement trail visualization
    --heatmap           Save a heatmap image at the end (saved as heatmap.png)
"""
 
import argparse
import collections
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
 
 
# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
TRAIL_LENGTH   = 40      # how many past positions to draw per player
TRAIL_FADE     = True    # fade older trail points
PERSON_CLASS   = 0       # COCO class index for "person"
COLORS = [               # palette for player IDs (cycles)
    (255,  80,  80), ( 80, 200, 255), ( 80, 255, 130), (255, 200,  50),
    (200,  80, 255), (255, 140,  50), ( 50, 230, 230), (255,  80, 180),
    (160, 255,  80), (140, 140, 255),
]
 
 
def get_color(track_id: int):
    return COLORS[track_id % len(COLORS)]
 
 
def draw_trail(frame, trail, color, fade=True):
    pts = list(trail)
    for i in range(1, len(pts)):
        if fade:
            alpha = i / len(pts)
            c = tuple(int(v * alpha) for v in color)
        else:
            c = color
        cv2.line(frame, pts[i - 1], pts[i], c, 2, cv2.LINE_AA)
 
 
def process_video(input_path: str, output_path: str, conf: float,
                  show: bool, trails: bool, heatmap: bool):
 
    print(f"\n🔍  Loading YOLOv8 model...")
    model = YOLO("yolov8n.pt")          # nano = fastest; swap for yolov8s.pt for better accuracy
 
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {input_path}")
 
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
 
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
 
    # Storage
    trail_history: dict[int, collections.deque] = {}
    heatmap_acc   = np.zeros((height, width), dtype=np.float32) if heatmap else None
    player_distances: dict[int, float] = {}
    frame_idx = 0
 
    print(f"📹  Processing '{input_path}'  ({width}×{height} @ {fps:.1f}fps, {total} frames)\n")
 
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
 
        # ── Run YOLOv8 tracking (ByteTrack built-in) ──────────────────────
        results = model.track(
            frame,
            persist=True,           # keep track IDs across frames
            conf=conf,
            classes=[PERSON_CLASS],
            tracker="bytetrack.yaml",
            verbose=False,
        )
 
        annotated = frame.copy()
 
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            ids   = results[0].boxes.id.cpu().numpy().astype(int)
 
            for box, tid in zip(boxes, ids):
                x1, y1, x2, y2 = box
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2   # foot-point midpoint
                color  = get_color(tid)
 
                # ── Trail ──────────────────────────────────────────────────
                if trails:
                    if tid not in trail_history:
                        trail_history[tid] = collections.deque(maxlen=TRAIL_LENGTH)
                    trail_history[tid].append((cx, y2))     # bottom-center = feet
 
                    if len(trail_history[tid]) > 1:
                        draw_trail(annotated, trail_history[tid], color, TRAIL_FADE)
 
                # ── Distance traveled (pixels) ─────────────────────────────
                if tid not in player_distances:
                    player_distances[tid] = 0.0
                if trails and len(trail_history[tid]) >= 2:
                    prev = trail_history[tid][-2]
                    curr = trail_history[tid][-1]
                    player_distances[tid] += float(np.linalg.norm(
                        np.array(curr) - np.array(prev)
                    ))
 
                # ── Heatmap accumulation ───────────────────────────────────
                if heatmap_acc is not None:
                    cv2.circle(heatmap_acc, (cx, cy), 15, 1.0, -1)
 
                # ── Bounding box ───────────────────────────────────────────
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
 
                # ── Label background ───────────────────────────────────────
                label   = f"P{tid}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
                cv2.putText(annotated, label, (x1 + 3, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
 
        # ── Frame counter overlay ─────────────────────────────────────────
        cv2.putText(annotated, f"Frame {frame_idx}/{total}",
                    (10, height - 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (200, 200, 200), 1, cv2.LINE_AA)
 
        writer.write(annotated)
 
        if show:
            cv2.imshow("Player Tracker  [q = quit]", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
 
        if frame_idx % 50 == 0 or frame_idx == total:
            pct = frame_idx / max(total, 1) * 100
            print(f"  {pct:5.1f}%  frame {frame_idx}/{total}", end="\r")
 
    cap.release()
    writer.release()
    if show:
        cv2.destroyAllWindows()
 
    print(f"\n\n✅  Saved annotated video → {output_path}")
 
    # ── Heatmap export ────────────────────────────────────────────────────
    if heatmap and heatmap_acc is not None:
        hm_path = str(Path(output_path).with_name("heatmap.png"))
        norm = cv2.normalize(heatmap_acc, None, 0, 255, cv2.NORM_MINMAX)
        hm_color = cv2.applyColorMap(norm.astype(np.uint8), cv2.COLORMAP_JET)
        cv2.imwrite(hm_path, hm_color)
        print(f"🌡️   Saved heatmap          → {hm_path}")
 
    # ── Summary stats ─────────────────────────────────────────────────────
    if player_distances:
        print("\n📊  Player Movement Summary (pixels traveled):")
        for tid, dist in sorted(player_distances.items()):
            print(f"    Player {tid:>3}:  {dist:>8.0f} px")
 
 
# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Sports player detection & tracking with YOLOv8 + ByteTrack"
    )
    parser.add_argument("--input",   required=True,  help="Path to input video file")
    parser.add_argument("--output",  default="tracked_output.mp4", help="Output video path")
    parser.add_argument("--conf",    type=float, default=0.4, help="Detection confidence threshold")
    parser.add_argument("--show",    action="store_true", help="Show live preview")
    parser.add_argument("--no-trails", action="store_true", help="Disable movement trails")
    parser.add_argument("--heatmap", action="store_true", help="Export heatmap image")
    args = parser.parse_args()
 
    process_video(
        input_path  = args.input,
        output_path = args.output,
        conf        = args.conf,
        show        = args.show,
        trails      = not args.no_trails,
        heatmap     = args.heatmap,
    )
 
 
if __name__ == "__main__":
    main()