import os
import uuid
from typing import List, Optional, Tuple

import cv2
import numpy as np
from flask import Flask, flash, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = os.path.join("static", "results")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tiff", "tif", "heic", "heif"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-secret-key")


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def resize_for_processing(image: np.ndarray, max_dim: int = 1600) -> Tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    scale = 1.0
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return image, scale


def auto_canny_thresholds(gray: np.ndarray) -> Tuple[int, int]:
    median = np.median(gray)
    lower = int(max(0, (1.0 - 0.33) * median))
    upper = int(min(255, (1.0 + 0.33) * median))
    return lower, upper


def preprocess_edges(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    lower, upper = auto_canny_thresholds(gray)
    edges = cv2.Canny(gray, lower, upper)
    edges = cv2.dilate(edges, None, iterations=1)
    edges = cv2.erode(edges, None, iterations=1)
    return gray, edges


def order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def find_calibration_square(edges: np.ndarray, gray: np.ndarray) -> Optional[Tuple[np.ndarray, float, float]]:
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    img_area = gray.shape[0] * gray.shape[1]
    best_quad = None
    best_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 0.01 * img_area:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) != 4:
            continue

        rect = approx.reshape(4, 2)
        (x, y, w, h) = cv2.boundingRect(rect)
        aspect_ratio = w / float(h) if h > 0 else 0
        if aspect_ratio < 0.9 or aspect_ratio > 1.1:
            continue

        if area > best_area:
            best_area = area
            best_quad = rect

    if best_quad is None:
        return None

    ordered = order_points(best_quad)
    side_lengths = [
        np.linalg.norm(ordered[i] - ordered[(i + 1) % 4])
        for i in range(4)
    ]
    side_length_px = float(np.mean(side_lengths))
    if side_length_px <= 0:
        return None

    pixels_per_mm = side_length_px / 100.0
    return ordered, side_length_px, pixels_per_mm


def bbox_from_points(pts: np.ndarray) -> Tuple[int, int, int, int]:
    x_coords = pts[:, 0]
    y_coords = pts[:, 1]
    return int(np.min(x_coords)), int(np.min(y_coords)), int(np.max(x_coords)), int(np.max(y_coords))


def iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter_area
    return inter_area / float(union) if union > 0 else 0.0


def find_bag_rect(edges: np.ndarray, gray: np.ndarray, calibration_quad: np.ndarray) -> Optional[Tuple[np.ndarray, float, float]]:
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    img_area = gray.shape[0] * gray.shape[1]
    calib_box = bbox_from_points(calibration_quad)

    candidates: List[Tuple[np.ndarray, float, float, float]] = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 0.01 * img_area:
            continue

        rect = cv2.minAreaRect(cnt)
        (center_x, center_y), (w, h), angle = rect
        if w <= 0 or h <= 0:
            continue

        box = cv2.boxPoints(rect)
        box = order_points(box)
        box_int = np.int32(box)

        candidate_box = bbox_from_points(box_int)
        overlap = iou(candidate_box, calib_box)
        if overlap > 0.2:
            continue

        rect_area = w * h
        candidates.append((box_int, float(w), float(h), rect_area))

    if not candidates:
        return None

    candidates.sort(key=lambda c: c[3], reverse=True)
    box_int, w, h, _ = candidates[0]
    return box_int, w, h


def annotate_image(image: np.ndarray, calibration_quad: np.ndarray, bag_rect: np.ndarray, width_mm: float, height_mm: float, width_in: float, height_in: float) -> np.ndarray:
    output = image.copy()
    cv2.polylines(output, [np.int32(calibration_quad)], True, (0, 165, 255), 3)
    cv2.polylines(output, [np.int32(bag_rect)], True, (0, 255, 0), 3)

    text = f"W: {width_mm:.1f} mm ({width_in:.2f} in), H: {height_mm:.1f} mm ({height_in:.2f} in)"
    text_org = (20, 40)

    overlay = output.copy()
    cv2.rectangle(overlay, (10, 10), (output.shape[1] - 10, 60), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.4, output, 0.6, 0, output)

    cv2.putText(output, text, text_org, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return output


def process_image(image_path: str, original_filename: str) -> dict:
    bgr = cv2.imread(image_path)
    if bgr is None:
        return {"success": False, "error": "Could not read the uploaded image."}

    resized, scale = resize_for_processing(bgr)
    gray, edges = preprocess_edges(resized)

    calib_result = find_calibration_square(edges, gray)
    if calib_result is None:
        return {
            "success": False,
            "error": "Calibration square not detected. Please ensure the printed 100 mm x 100 mm grid is fully visible, flat, and well-lit in the image.",
        }

    calibration_quad, side_length_px, pixels_per_mm = calib_result
    if pixels_per_mm <= 0:
        return {"success": False, "error": "Invalid calibration data detected."}

    bag_result = find_bag_rect(edges, gray, calibration_quad)
    if bag_result is None:
        return {
            "success": False,
            "error": "Could not detect a rectangular object (bag). Please make sure the bag is clearly visible and not merged with the background.",
        }

    bag_rect, width_px, height_px = bag_result
    width_px, height_px = sorted([width_px, height_px], reverse=True)

    width_mm = width_px / pixels_per_mm
    height_mm = height_px / pixels_per_mm
    width_in = width_mm / 25.4
    height_in = height_mm / 25.4

    annotated = annotate_image(resized, calibration_quad, bag_rect, width_mm, height_mm, width_in, height_in)
    result_filename = f"annotated_{uuid.uuid4().hex}.jpg"
    result_path = os.path.join(RESULT_FOLDER, result_filename)
    cv2.imwrite(result_path, annotated)

    return {
        "success": True,
        "original_filename": original_filename,
        "width_mm": width_mm,
        "height_mm": height_mm,
        "width_in": width_in,
        "height_in": height_in,
        "annotated_image": result_filename,
    }


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/measure", methods=["POST"])
def measure():
    if "file" not in request.files:
        flash("No file part in the request.")
        return redirect(url_for("index"))

    file = request.files["file"]
    if file.filename == "":
        flash("No file selected.")
        return redirect(url_for("index"))

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        unique_name = f"{uuid.uuid4().hex}_{filename}"
        upload_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
        file.save(upload_path)

        result = process_image(upload_path, filename)
        if result.get("success"):
            annotated_url = url_for("static", filename=f"results/{result['annotated_image']}")
            return render_template(
                "result.html",
                filename=filename,
                width_mm=result["width_mm"],
                height_mm=result["height_mm"],
                width_in=result["width_in"],
                height_in=result["height_in"],
                annotated_image=annotated_url,
                error_message=None,
            )
        return render_template("result.html", filename=filename, error_message=result.get("error"))

    flash("Unsupported file type. Please upload an image file.")
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
