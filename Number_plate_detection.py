import argparse
from pathlib import Path

import cv2


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CASCADE = BASE_DIR / "haarcascade_russian_plate_number.xml"
EXIT_KEYS = {27, ord("q"), ord("Q")}


def parse_args():
    parser = argparse.ArgumentParser(description="Detect license plates from camera, image, or video.")
    parser.add_argument("--image", type=str, help="Path to an input image.")
    parser.add_argument("--video", type=str, help="Path to an input video.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index to use when image/video is not provided.")
    parser.add_argument("--cascade", type=str, default=str(DEFAULT_CASCADE), help="Path to Haar cascade XML file.")
    parser.add_argument("--output-dir", type=str, default="detected_plates", help="Folder for saved plate crops.")
    parser.add_argument("--min-area", type=int, default=500, help="Minimum detected plate area.")
    return parser.parse_args()


def load_cascade(cascade_path):
    path = Path(cascade_path)
    if not path.exists():
        raise FileNotFoundError(f"Cascade file not found: {path}")

    cascade = cv2.CascadeClassifier(str(path))
    if cascade.empty():
        raise RuntimeError(f"Unable to load cascade file: {path}")
    return cascade


def detect_plates(frame, cascade, min_area):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detections = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
    plates = []

    for x, y, w, h in detections:
        if w * h < min_area:
            continue
        plates.append((x, y, w, h))
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, "Number Plate", (x, max(30, y - 8)), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 2)

    return plates


def save_plate(frame, plate, output_dir, count):
    x, y, w, h = plate
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"plate_{count:03d}.jpg"
    cv2.imwrite(str(output_path), frame[y:y + h, x:x + w])
    return output_path


def show_status(frame, text):
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 42), (0, 128, 0), cv2.FILLED)
    cv2.putText(frame, text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


def should_exit(delay=1):
    return cv2.waitKey(delay) & 0xFF in EXIT_KEYS


def process_image(image_path, cascade, output_dir, min_area):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Unable to open image: {image_path}")

    plates = detect_plates(image, cascade, min_area)
    if plates:
        output_path = save_plate(image, plates[0], output_dir, 0)
        show_status(image, f"Saved: {output_path}")
    else:
        show_status(image, "No plate detected")

    cv2.imshow("License Plate Detection", image)
    while not should_exit(50):
        pass


def process_stream(source, cascade, output_dir, min_area):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open source: {source}")

    count = 0
    last_plate = None

    while True:
        success, frame = cap.read()
        if not success:
            break

        plates = detect_plates(frame, cascade, min_area)
        if plates:
            last_plate = plates[0]

        cv2.putText(
            frame,
            "Press S to save plate, Esc/Q to exit",
            (12, frame.shape[0] - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        cv2.imshow("License Plate Detection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in EXIT_KEYS:
            break
        if key in {ord("s"), ord("S")}:
            if last_plate is None:
                show_status(frame, "No plate detected to save")
            else:
                output_path = save_plate(frame, last_plate, output_dir, count)
                count += 1
                show_status(frame, f"Saved: {output_path}")
            cv2.imshow("License Plate Detection", frame)
            cv2.waitKey(500)

    cap.release()


def main():
    args = parse_args()
    selected_inputs = [bool(args.image), bool(args.video)]
    if sum(selected_inputs) > 1:
        raise ValueError("Use only one input source: --image, --video, or --camera.")

    cascade = load_cascade(args.cascade)
    output_dir = Path(args.output_dir)

    if args.image:
        process_image(args.image, cascade, output_dir, args.min_area)
    elif args.video:
        process_stream(args.video, cascade, output_dir, args.min_area)
    else:
        process_stream(args.camera, cascade, output_dir, args.min_area)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
