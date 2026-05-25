# License Plate Detection System

Detect license plates from a webcam, image, or video using OpenCV and the included Haar cascade.

## Setup

```bash
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run

Use webcam:

```bash
.\.venv\Scripts\python.exe Number_plate_detection.py
```

Use an image:

```bash
.\.venv\Scripts\python.exe Number_plate_detection.py --image path\to\car.jpg
```

Use a video:

```bash
.\.venv\Scripts\python.exe Number_plate_detection.py --video path\to\traffic.mp4
```

Press `S` to save the latest detected plate crop into `detected_plates/`.
Press `Esc` or `Q` to exit.
