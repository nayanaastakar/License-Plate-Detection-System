import shutil
import uuid
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request, send_from_directory

from Number_plate_detection import DEFAULT_CASCADE, load_cascade, save_plate


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
REALTIME_DIR = OUTPUT_DIR / "realtime"
DEFAULT_VIDEO = Path(r"C:\Users\HP\Downloads\video.mp4")
DEFAULT_IMAGE = Path(r"C:\Users\HP\Downloads\img car.jpg")
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".mp4", ".avi", ".mov", ".mkv"}
INDIAN_STATE_CODES = {
    "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "GA", "GJ", "HR", "HP",
    "JH", "JK", "KA", "KL", "LA", "LD", "MH", "ML", "MN", "MP", "MZ", "NL",
    "OD", "PB", "PY", "RJ", "SK", "TN", "TS", "TR", "UK", "UP", "WB",
}
OCR_CONFUSION_COSTS = {
    ("X", "M"): 0.35,
    ("B", "H"): 0.35,
    ("8", "B"): 0.35,
    ("0", "O"): 0.35,
    ("1", "I"): 0.35,
    ("5", "S"): 0.35,
    ("2", "Z"): 0.35,
}

app = Flask(__name__)
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
REALTIME_DIR.mkdir(exist_ok=True)
OCR_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
OCR_FONTS = [
    cv2.FONT_HERSHEY_SIMPLEX,
    cv2.FONT_HERSHEY_DUPLEX,
    cv2.FONT_HERSHEY_COMPLEX,
    cv2.FONT_HERSHEY_TRIPLEX,
]
OCR_TEMPLATES = None
LATEST_CAMERA_DETECTION = {}


def is_allowed_file(filename):
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def relative_url(path):
    return f"/outputs/{path.relative_to(OUTPUT_DIR).as_posix()}"


def write_plate_text(run_dir, plate_text):
    text_path = run_dir / "plate_text.txt"
    text_path.write_text(plate_text or "Text not recognized", encoding="utf-8")
    return text_path


def is_plate_candidate(plate, frame_shape):
    x, y, w, h = [int(v) for v in plate]
    frame_height, frame_width = frame_shape[:2]
    if not frame_width or not frame_height or not h:
        return False

    aspect_ratio = w / h
    area_ratio = (w * h) / (frame_width * frame_height)
    width_ratio = w / frame_width
    height_ratio = h / frame_height

    return (
        2.0 <= aspect_ratio <= 6.2
        and 0.0005 <= area_ratio <= 0.055
        and 0.04 <= width_ratio <= 0.48
        and 0.018 <= height_ratio <= 0.16
    )


def candidate_key(plate):
    x, y, w, h = [int(v) for v in plate]
    return (x // 3, y // 3, w // 3, h // 3)


def draw_plate_marker(frame, plate):
    x, y, w, h = [int(v) for v in plate]
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)


def contour_plate_candidates(frame, min_area):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    filtered = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(filtered, 30, 200)
    contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    frame_height, frame_width = frame.shape[:2]
    candidates = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area < min_area:
            continue

        aspect_ratio = w / h if h else 0
        area_ratio = area / (frame_width * frame_height)
        center_y_ratio = (y + h / 2.0) / frame_height
        if (
            2.0 <= aspect_ratio <= 6.5
            and 0.001 <= area_ratio <= 0.08
            and 0.35 <= center_y_ratio <= 0.92
            and w <= frame_width * 0.62
            and h <= frame_height * 0.25
        ):
            candidates.append((x, y, w, h))
    return candidates


def haar_plate_candidates(frame, cascade, min_area):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detections = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
    return [
        (int(x), int(y), int(w), int(h))
        for x, y, w, h in detections
        if int(w) * int(h) >= min_area
    ]


def collect_plate_candidates(frame, annotated, cascade, min_area):
    plates = haar_plate_candidates(frame, cascade, min_area)
    plates.extend(contour_plate_candidates(frame, min_area))

    unique = []
    seen = set()
    for plate in plates:
        key = candidate_key(plate)
        if key in seen:
            continue
        seen.add(key)
        unique.append(plate)
    return unique


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


def normalize_character(image):
    ys, xs = np.where(image > 0)
    if len(xs) == 0:
        return np.zeros((48, 32), dtype=np.uint8)

    x1, x2 = xs.min(), xs.max() + 1
    y1, y2 = ys.min(), ys.max() + 1
    crop = image[y1:y2, x1:x2]
    height, width = crop.shape
    size = max(height, width)
    canvas = np.zeros((size, size), dtype=np.uint8)
    y_offset = (size - height) // 2
    x_offset = (size - width) // 2
    canvas[y_offset:y_offset + height, x_offset:x_offset + width] = crop
    return cv2.resize(canvas, (32, 48), interpolation=cv2.INTER_AREA)


def build_ocr_templates():
    templates = []
    for char in OCR_CHARS:
        for font in OCR_FONTS:
            for scale in (1.2, 1.4, 1.6, 1.8):
                for thickness in (2, 3, 4):
                    canvas = np.zeros((80, 60), dtype=np.uint8)
                    (text_width, text_height), _ = cv2.getTextSize(char, font, scale, thickness)
                    x = (60 - text_width) // 2
                    y = (80 + text_height) // 2
                    cv2.putText(canvas, char, (x, y), font, scale, 255, thickness, cv2.LINE_AA)
                    _, binary = cv2.threshold(canvas, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    templates.append((char, normalize_character(binary)))
    return templates


def allowed_chars_for_position(index, length):
    if length >= 9:
        if index in (0, 1, 4, 5):
            return "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        if index in (2, 3) or index >= 6:
            return "0123456789"
    return OCR_CHARS


def classify_character(character_image, index, length):
    global OCR_TEMPLATES
    if OCR_TEMPLATES is None:
        OCR_TEMPLATES = build_ocr_templates()

    normalized = normalize_character(character_image)
    allowed = allowed_chars_for_position(index, length)
    best_score = -1.0
    best_char = ""
    for char, template in OCR_TEMPLATES:
        if char not in allowed:
            continue
        score = cv2.matchTemplate(normalized, template, cv2.TM_CCOEFF_NORMED)[0][0]
        if score > best_score:
            best_score = score
            best_char = char
    return best_char


def confusion_cost(source, target):
    if source == target:
        return 0.0
    return OCR_CONFUSION_COSTS.get((source, target), OCR_CONFUSION_COSTS.get((target, source), 1.0))


def correct_indian_plate_text(text):
    if len(text) < 4:
        return text

    prefix = text[:2]
    if prefix in INDIAN_STATE_CODES:
        return text

    best_code = None
    best_cost = 3.0
    for code in INDIAN_STATE_CODES:
        cost = confusion_cost(prefix[0], code[0]) + confusion_cost(prefix[1], code[1])
        if cost < best_cost:
            best_code = code
            best_cost = cost

    if best_code and best_cost <= 0.9:
        return f"{best_code}{text[2:]}"
    return text


def read_plate_text(plate_image):
    if plate_image is None or plate_image.size == 0:
        return ""

    upscaled = cv2.resize(plate_image, None, fx=6, fy=6, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    _, _, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    boxes = []
    for x, y, width, height, area in stats[1:]:
        if area < 100 or height < 30 or width < 8:
            continue
        if height > upscaled.shape[0] * 0.9 or width > upscaled.shape[1] * 0.9:
            continue
        if y < upscaled.shape[0] * 0.2 or y > upscaled.shape[0] * 0.78:
            continue
        boxes.append((int(x), int(y), int(width), int(height)))

    boxes = sorted(boxes, key=lambda box: box[0])
    if not 6 <= len(boxes) <= 12:
        return ""

    characters = []
    for index, (x, y, width, height) in enumerate(boxes):
        character_image = binary[y:y + height, x:x + width]
        characters.append(classify_character(character_image, index, len(boxes)))
    return correct_indian_plate_text("".join(characters))


def detection_from_candidate(frame, annotated, plate, frame_no, run_dir, frames_dir, crops_dir):
    x, y, w, h = [int(v) for v in plate]
    plate_image = frame[y:y + h, x:x + w]
    plate_text = read_plate_text(plate_image)
    if not plate_text:
        return None

    draw_plate_marker(annotated, plate)
    annotated_path = frames_dir / f"final_frame_{frame_no:04d}.jpg"
    crop_path = save_plate(frame, plate, crops_dir, 0)
    final_crop = run_dir / "final_number_plate.jpg"
    text_path = write_plate_text(run_dir, plate_text)
    shutil.copy2(crop_path, final_crop)
    cv2.imwrite(str(annotated_path), annotated)

    return {
        "index": 0,
        "frame": int(frame_no),
        "plate_text": plate_text,
        "x": x,
        "y": y,
        "width": w,
        "height": h,
        "crop_file": relative_url(final_crop),
        "annotated_frame": relative_url(annotated_path),
        "text_file": relative_url(text_path),
    }


def detect_readable_plate(frame, cascade, min_area):
    annotated = frame.copy()
    candidates = [
        plate
        for plate in collect_plate_candidates(frame, annotated, cascade, min_area)
        if is_plate_candidate(plate, frame.shape)
    ]

    for plate in sorted(candidates, key=lambda p: plate_score(p, frame.shape), reverse=True):
        x, y, w, h = [int(v) for v in plate]
        plate_image = frame[y:y + h, x:x + w]
        plate_text = read_plate_text(plate_image)
        if plate_text:
            return plate, plate_text
    return None, ""


def save_realtime_detection(frame, plate, plate_text):
    frames_dir = REALTIME_DIR / "annotated_frames"
    crops_dir = REALTIME_DIR / "plate_crops"
    frames_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)

    annotated = frame.copy()
    draw_plate_marker(annotated, plate)
    x, y, w, h = [int(v) for v in plate]
    cv2.putText(
        annotated,
        plate_text,
        (x, min(frame.shape[0] - 10, y + h + 24)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (15, 118, 110),
        2,
        cv2.LINE_AA,
    )

    annotated_path = frames_dir / "latest_frame.jpg"
    crop_path = REALTIME_DIR / "final_number_plate.jpg"
    text_path = write_plate_text(REALTIME_DIR, plate_text)
    cv2.imwrite(str(annotated_path), annotated)
    cv2.imwrite(str(crop_path), frame[y:y + h, x:x + w])

    LATEST_CAMERA_DETECTION.clear()
    LATEST_CAMERA_DETECTION.update({
        "plate_text": plate_text,
        "x": x,
        "y": y,
        "width": w,
        "height": h,
        "crop_file": relative_url(crop_path),
        "annotated_frame": relative_url(annotated_path),
        "text_file": relative_url(text_path),
        "output_folder": str(REALTIME_DIR),
    })


def camera_frame_generator(camera_index, min_area):
    cascade = load_cascade(DEFAULT_CASCADE)
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        frame = np.full((360, 640, 3), 245, dtype=np.uint8)
        cv2.putText(frame, "Camera not available", (90, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 180), 2)
        ok, encoded = cv2.imencode(".jpg", frame)
        if ok:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + encoded.tobytes() + b"\r\n"
        return

    last_saved_text = ""
    frame_no = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_no += 1
            annotated = frame.copy()
            if frame_no % 3 == 0:
                plate, plate_text = detect_readable_plate(frame, cascade, min_area)
                if plate is not None:
                    draw_plate_marker(annotated, plate)
                    x, y, w, h = [int(v) for v in plate]
                    cv2.putText(
                        annotated,
                        plate_text,
                        (x, min(frame.shape[0] - 10, y + h + 24)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (15, 118, 110),
                        2,
                        cv2.LINE_AA,
                    )
                    if plate_text != last_saved_text:
                        save_realtime_detection(frame, plate, plate_text)
                        last_saved_text = plate_text

            ok, encoded = cv2.imencode(".jpg", annotated)
            if not ok:
                continue
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + encoded.tobytes() + b"\r\n"
    finally:
        cap.release()


def process_image(input_path, run_dir, min_area):
    cascade = load_cascade(DEFAULT_CASCADE)
    image = cv2.imread(str(input_path))
    if image is None:
        raise FileNotFoundError(f"Unable to open image: {input_path}")

    annotated = image.copy()
    plates = collect_plate_candidates(image, annotated, cascade, min_area)
    frames_dir = run_dir / "annotated_frames"
    crops_dir = run_dir / "plate_crops"
    frames_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)

    detections = []
    annotated_path = frames_dir / "image_result.jpg"
    cv2.imwrite(str(annotated_path), annotated)

    candidates = [plate for plate in plates if is_plate_candidate(plate, image.shape)]
    for plate in sorted(candidates, key=lambda p: plate_score(p, image.shape), reverse=True):
        detection = detection_from_candidate(image, annotated, plate, 1, run_dir, frames_dir, crops_dir)
        if detection:
            detections.append(detection)
            break

    return detections


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
        plates = collect_plate_candidates(frame, annotated, cascade, min_area)
        for plate in plates:
            if not is_plate_candidate(plate, frame.shape):
                continue
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
        detection = detection_from_candidate(
            best["frame"],
            best["annotated"],
            best["plate"],
            best["frame_no"],
            run_dir,
            frames_dir,
            crops_dir,
        )
        if detection:
            detections.append(detection)

    return detections


@app.route("/")
def index():
    return render_template(
        "index.html",
        default_video_exists=DEFAULT_VIDEO.exists(),
        default_image_exists=DEFAULT_IMAGE.exists(),
    )


@app.route("/api/process", methods=["POST"])
def process_upload():
    try:
        run_id = uuid.uuid4().hex[:10]
        run_dir = OUTPUT_DIR / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        min_area = int(request.form.get("min_area", 120))
        use_default = request.form.get("use_default") == "true"
        use_default_image = request.form.get("use_default_image") == "true"
        file = request.files.get("file")

        if file and file.filename:
            if not is_allowed_file(file.filename):
                return jsonify({"status": "error", "message": "Unsupported file type."}), 400
            input_path = UPLOAD_DIR / f"{run_id}_{Path(file.filename).name}"
            file.save(input_path)
        elif use_default and DEFAULT_VIDEO.exists():
            input_path = run_dir / DEFAULT_VIDEO.name
            shutil.copy2(DEFAULT_VIDEO, input_path)
        elif use_default_image and DEFAULT_IMAGE.exists():
            input_path = run_dir / DEFAULT_IMAGE.name
            shutil.copy2(DEFAULT_IMAGE, input_path)
        else:
            return jsonify({"status": "error", "message": "Upload a file, use the default video, or use the default image."}), 400

        suffix = input_path.suffix.lower()
        if suffix in {".jpg", ".jpeg", ".png", ".bmp"}:
            detections = process_image(input_path, run_dir, min_area)
        else:
            detections = process_video(input_path, run_dir, min_area)

        return jsonify({
            "status": "success",
            "run_id": run_id,
            "input_name": input_path.name,
            "detections": detections,
            "count": len(detections),
            "text_url": detections[0]["text_file"] if detections else "",
            "output_folder": str(run_dir),
        })
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route("/camera_feed")
def camera_feed():
    camera_index = int(request.args.get("camera", 0))
    min_area = int(request.args.get("min_area", 120))
    return Response(
        camera_frame_generator(camera_index, min_area),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/api/camera/latest")
def latest_camera_detection():
    return jsonify({
        "status": "success",
        "detection": LATEST_CAMERA_DETECTION,
        "count": 1 if LATEST_CAMERA_DETECTION else 0,
    })


@app.route("/outputs/<path:filename>")
def outputs(filename):
    return send_from_directory(OUTPUT_DIR, filename)


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=5050, use_reloader=False)
