import csv
import json
import shutil
import uuid
from pathlib import Path

import cv2
from flask import Flask, jsonify, render_template, request, send_from_directory

from Number_plate_detection import DEFAULT_CASCADE, detect_plates, load_cascade, save_plate


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
DEFAULT_VIDEO = Path(r"C:\Users\HP\Downloads\video.mp4")
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".mp4", ".avi", ".mov", ".mkv"}

app = Flask(__name__)
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


def is_allowed_file(filename):
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def relative_url(path):
    return f"/outputs/{path.relative_to(OUTPUT_DIR).as_posix()}"


def write_reports(run_dir, detections):
    csv_path = run_dir / "detections.csv"
    json_path = run_dir / "detections.json"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["index", "frame", "x", "y", "width", "height", "crop_file", "annotated_frame"],
        )
        writer.writeheader()
        writer.writerows(detections)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(detections, f, indent=2)

    return csv_path, json_path


def process_image(input_path, run_dir, min_area):
    cascade = load_cascade(DEFAULT_CASCADE)
    image = cv2.imread(str(input_path))
    if image is None:
        raise FileNotFoundError(f"Unable to open image: {input_path}")

    annotated = image.copy()
    plates = detect_plates(annotated, cascade, min_area)
    frames_dir = run_dir / "annotated_frames"
    crops_dir = run_dir / "plate_crops"
    frames_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)

    detections = []
    annotated_path = frames_dir / "image_result.jpg"
    cv2.imwrite(str(annotated_path), annotated)

    for index, plate in enumerate(plates):
        crop_path = save_plate(image, plate, crops_dir, index)
        x, y, w, h = [int(v) for v in plate]
        detections.append({
            "index": index,
            "frame": 1,
            "x": x,
            "y": y,
            "width": w,
            "height": h,
            "crop_file": relative_url(crop_path),
            "annotated_frame": relative_url(annotated_path),
        })

    csv_path, json_path = write_reports(run_dir, detections)
    return detections, csv_path, json_path


def similar_box(box, previous_boxes):
    x, y, w, h = box
    cx, cy = x + w / 2, y + h / 2
    for px, py, pw, ph in previous_boxes:
        pcx, pcy = px + pw / 2, py + ph / 2
        if abs(cx - pcx) < 28 and abs(cy - pcy) < 28 and abs(w - pw) < 40 and abs(h - ph) < 30:
            return True
    return False


def process_video(input_path, run_dir, min_area, frame_step=5, max_detections=20):
    cascade = load_cascade(DEFAULT_CASCADE)
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {input_path}")

    frames_dir = run_dir / "annotated_frames"
    crops_dir = run_dir / "plate_crops"
    frames_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)

    detections = []
    seen_boxes = []
    frame_no = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_no += 1
        if frame_no % frame_step != 0:
            continue

        annotated = frame.copy()
        plates = detect_plates(annotated, cascade, min_area)
        for plate in plates:
            if similar_box(plate, seen_boxes):
                continue

            seen_boxes.append(plate)
            index = len(detections)
            annotated_path = frames_dir / f"frame_{frame_no:04d}.jpg"
            crop_path = save_plate(frame, plate, crops_dir, index)
            cv2.imwrite(str(annotated_path), annotated)

            x, y, w, h = [int(v) for v in plate]
            detections.append({
                "index": index,
                "frame": frame_no,
                "x": x,
                "y": y,
                "width": w,
                "height": h,
                "crop_file": relative_url(crop_path),
                "annotated_frame": relative_url(annotated_path),
            })

            if len(detections) >= max_detections:
                cap.release()
                csv_path, json_path = write_reports(run_dir, detections)
                return detections, csv_path, json_path

    cap.release()
    csv_path, json_path = write_reports(run_dir, detections)
    return detections, csv_path, json_path


@app.route("/")
def index():
    return render_template("index.html", default_video_exists=DEFAULT_VIDEO.exists())


@app.route("/api/process", methods=["POST"])
def process_upload():
    try:
        run_id = uuid.uuid4().hex[:10]
        run_dir = OUTPUT_DIR / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        min_area = int(request.form.get("min_area", 120))
        use_default = request.form.get("use_default") == "true"
        file = request.files.get("file")

        if file and file.filename:
            if not is_allowed_file(file.filename):
                return jsonify({"status": "error", "message": "Unsupported file type."}), 400
            input_path = UPLOAD_DIR / f"{run_id}_{Path(file.filename).name}"
            file.save(input_path)
        elif use_default and DEFAULT_VIDEO.exists():
            input_path = run_dir / DEFAULT_VIDEO.name
            shutil.copy2(DEFAULT_VIDEO, input_path)
        else:
            return jsonify({"status": "error", "message": "Upload a file or use the default video."}), 400

        suffix = input_path.suffix.lower()
        if suffix in {".jpg", ".jpeg", ".png", ".bmp"}:
            detections, csv_path, json_path = process_image(input_path, run_dir, min_area)
        else:
            detections, csv_path, json_path = process_video(input_path, run_dir, min_area)

        return jsonify({
            "status": "success",
            "run_id": run_id,
            "input_name": input_path.name,
            "detections": detections,
            "count": len(detections),
            "csv_url": relative_url(csv_path),
            "json_url": relative_url(json_path),
            "output_folder": str(run_dir),
        })
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route("/outputs/<path:filename>")
def outputs(filename):
    return send_from_directory(OUTPUT_DIR, filename)


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=5050, use_reloader=False)
