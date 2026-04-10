# sports-player-tracker

Automatic player detection and tracking in sports video using YOLOv8 and ByteTrack. Generates annotated video with bounding boxes, player IDs, movement trails, and activity heatmaps.

---

## Demo

https://github.com/user-attachments/assets/your-demo-video-id

> Replace the link above with your tracked_output.mp4 once uploaded to GitHub.

---

## How It Works

1. **YOLOv8** scans each frame and draws a bounding box around every detected player
2. **ByteTrack** links detections across frames and assigns each player a persistent ID (P1, P2, P3…)
3. A **movement trail** of the last 40 positions is drawn behind each player
4. Optionally, a **heatmap** is exported showing which areas of the field had the most activity

No custom training required — YOLOv8 comes pretrained and already knows what a person looks like.

---

## Requirements

```bash
pip install ultralytics opencv-python numpy
```

YOLOv8 weights (`yolov8n.pt`) download automatically on first run.

---

## Usage

```bash
python player_tracker.py --input your_video.mp4
```

### All options

| Flag | Default | Description |
|---|---|---|
| `--input` | required | Path to input video |
| `--output` | `tracked_output.mp4` | Output video filename |
| `--conf` | `0.4` | Detection confidence threshold (0.0–1.0) |
| `--show` | off | Show live preview window while processing |
| `--no-trails` | off | Disable movement trails |
| `--heatmap` | off | Export a `heatmap.png` alongside the video |

### Example with all options

```bash
python player_tracker.py --input match.mp4 --output result.mp4 --conf 0.4 --show --heatmap
```

---

## Output

- **Annotated video** — bounding boxes, player IDs, and movement trails overlaid on original footage
- **heatmap.png** — color map of player activity across the clip (red = high activity, blue = low)
- **Terminal summary** — pixel distance traveled per player ID

---

## Limitations

- Does not distinguish between teams, referees, or other people on the pitch
- Distance is measured in pixels, not physical meters (no camera calibration)
- Players that leave and re-enter the frame may receive a new ID
- Best results on a GPU; CPU processing may run slightly below real-time on 30fps footage

---

## Built With

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [ByteTrack](https://github.com/ifzhang/ByteTrack) (via Ultralytics)
- [OpenCV](https://opencv.org/)
