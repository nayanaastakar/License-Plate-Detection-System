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


def plate_score(plate, frame_shape):
    x, y, w, h = plate
    frame_height = frame_shape[0]
    aspect_ratio = w / h if h else 0
    area = w * h
    aspect_score = 1.0 - min(abs(aspect_ratio - 3.5) / 3.5, 1.0)
    center_y_ratio = (y + h / 2.0) / frame_height if frame_height else 0.0
    vertical_score = 0.25 + center_y_ratio
    if center_y_ratio < 0.3:
        vertical_score *= 0.15
    size_penalty = 0.45 if area > 9000 else 1.0
    return area * (0.55 + 0.45 * aspect_score) * vertical_score * size_penalty


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

    best_plate = max(plates, key=lambda plate: plate_score(plate, image.shape)) if plates else None
    if best_plate is not None:
        crop_path = save_plate(image, best_plate, crops_dir, 0)
        final_crop = run_dir / "final_number_plate.jpg"
        shutil.copy2(crop_path, final_crop)
        x, y, w, h = [int(v) for v in best_plate]
        detections.append({
            "index": 0,
            "frame": 1,
            "x": x,
            "y": y,
            "width": w,
            "height": h,
            "crop_file": relative_url(final_crop),
            "annotated_frame": relative_url(annotated_path),
        })

    csv_path, json_path = write_reports(run_dir, detections)
    return detections, csv_path, json_path


def process_video(input_path, run_dir, min_area, frame_step=5):
    cascade = load_cascade(DEFAULT_CASCADE)
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {input_path}")

    frames_dir = run_dir / "annotated_frames"
    crops_dir = run_dir / "plate_crops"
    frames_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)

    best = None
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
            score = plate_score(plate, frame.shape)
            if best is None or score > best["score"]:
                best = {
                    "score": score,
                    "frame": frame.copy(),
                    "annotated": annotated.copy(),
                    "plate": plate,
                    "frame_no": frame_no,
                }

    cap.release()

    detections = []
    if best is not None:
        annotated_path = frames_dir / f"final_frame_{best['frame_no']:04d}.jpg"
        crop_path = save_plate(best["frame"], best["plate"], crops_dir, 0)
        final_crop = run_dir / "final_number_plate.jpg"
        shutil.copy2(crop_path, final_crop)
        cv2.imwrite(str(annotated_path), best["annotated"])

        x, y, w, h = [int(v) for v in best["plate"]]
        detections.append({
            "index": 0,
            "frame": int(best["frame_no"]),
            "x": x,
            "y": y,
            "width": w,
            "height": h,
            "crop_file": relative_url(final_crop),
            "annotated_frame": relative_url(annotated_path),
        })

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
