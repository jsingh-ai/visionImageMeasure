import os
import uuid
from typing import List, Optional, Tuple

import cv2
import numpy as np
from flask import Flask, flash, redirect, render_template, request, url_for
from markupsafe import Markup
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
    if lower == upper:
        upper = min(255, lower + 50)
    return lower, upper


def preprocess_edges(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    lower, upper = auto_canny_thresholds(blurred)
    canny_edges = cv2.Canny(blurred, lower, upper)

    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5
    )
    combined = cv2.bitwise_or(canny_edges, thresh)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opened = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)
    return gray, opened


def order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


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


def warp_square_and_scale(gray: np.ndarray, quad: np.ndarray) -> Optional[Tuple[np.ndarray, float, float]]:
    rect = order_points(quad.astype("float32"))
    (tl, tr, br, bl) = rect
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_w = int(max(width_a, width_b))
    max_h = int(max(height_a, height_b))
    if max_w < 20 or max_h < 20:
        return None
    dst = np.array([[0, 0], [max_w - 1, 0], [max_w - 1, max_h - 1], [0, max_h - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(gray, M, (max_w, max_h))
    side_length_px = float((max_w + max_h) / 2.0)
    pixels_per_mm = side_length_px / 100.0
    return rect, side_length_px, pixels_per_mm


def find_calibration_square(edges: np.ndarray, gray: np.ndarray) -> Optional[Tuple[np.ndarray, float, float]]:
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    img_area = gray.shape[0] * gray.shape[1]
    best = None
    best_area = 0.0
    fiducials: List[np.ndarray] = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 0.005 * img_area:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) != 4:
            continue

        quad = approx.reshape(4, 2)
        rect = cv2.boundingRect(quad)
        _, _, w, h = rect
        if w <= 0 or h <= 0:
            continue

        if area < 0.02 * img_area:
            fiducials.append(quad)
            continue

        warped = warp_square_and_scale(gray, quad)
        if warped is None:
            continue
        _, side_length_px, pixels_per_mm = warped
        if side_length_px <= 0 or pixels_per_mm <= 0:
            continue

        if area > best_area:
            best_area = area
            best = (quad, side_length_px, pixels_per_mm)

    if best is None:
        return None

    quad, side_length_px, pixels_per_mm = best
    ordered = order_points(quad.astype("float32"))

    if len(fiducials) >= 2:
        fid_points = np.concatenate(fiducials, axis=0).astype("float32")
        if fid_points.shape[0] >= 4:
            ordered = order_points(fid_points)

    return ordered, side_length_px, pixels_per_mm


def find_bag_rect(edges: np.ndarray, gray: np.ndarray, calibration_quad: np.ndarray) -> Optional[Tuple[np.ndarray, float, float]]:
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    img_area = gray.shape[0] * gray.shape[1]
    calib_box = bbox_from_points(calibration_quad)
    candidates: List[Tuple[np.ndarray, float, float, float]] = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 0.005 * img_area:
            continue

        rect = cv2.minAreaRect(cnt)
        (cx, cy), (w, h), _ = rect
        if w <= 0 or h <= 0:
            continue
        min_side = min(w, h)
        max_side = max(w, h)
        if min_side < 20 or (min_side / max_side) < 0.2:
            continue

        box = cv2.boxPoints(rect)
        box = order_points(box)
        box_int = np.int32(box)

        overlap = iou(bbox_from_points(box_int), calib_box)
        if overlap > 0.2:
            continue

        rect_area = w * h
        candidates.append((box_int, float(w), float(h), rect_area))

    if not candidates:
        return None

    candidates.sort(key=lambda c: c[3], reverse=True)
    box_int, w, h, _ = candidates[0]
    return box_int, w, h


def annotate_image(
    image: np.ndarray,
    calibration_quad: Optional[np.ndarray],
    bag_rect: Optional[np.ndarray],
    width_mm: Optional[float],
    height_mm: Optional[float],
    width_in: Optional[float],
    height_in: Optional[float],
    status_text: Optional[str] = None,
) -> np.ndarray:
    output = image.copy()
    if calibration_quad is not None:
        cv2.polylines(output, [np.int32(calibration_quad)], True, (0, 165, 255), 3)
    if bag_rect is not None:
        cv2.polylines(output, [np.int32(bag_rect)], True, (0, 255, 0), 3)
        if width_mm is not None and height_mm is not None:
            center = tuple(np.mean(bag_rect, axis=0).astype(int))
            cv2.putText(
                output,
                f"W {width_mm:.1f} mm  H {height_mm:.1f} mm",
                (center[0] - 100, center[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

    overlay = output.copy()
    cv2.rectangle(overlay, (10, 10), (output.shape[1] - 10, 70), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.4, output, 0.6, 0, output)

    if width_mm is not None and height_mm is not None and width_in is not None and height_in is not None:
        text = f"W: {width_mm:.1f} mm ({width_in:.2f} in)  |  H: {height_mm:.1f} mm ({height_in:.2f} in)"
    else:
        text = status_text or "Processing"
    cv2.putText(output, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return output


def save_display_image(image: np.ndarray, filename_prefix: str, text: str) -> str:
    annotated = annotate_image(image, None, None, None, None, None, None, status_text=text)
    result_filename = f"{filename_prefix}_{uuid.uuid4().hex}.jpg"
    result_path = os.path.join(RESULT_FOLDER, result_filename)
    cv2.imwrite(result_path, annotated)
    return result_filename


def process_image(image_path: str, original_filename: str) -> dict:
    bgr = cv2.imread(image_path)
    if bgr is None:
        return {"success": False, "error": "Could not read the uploaded image.", "display_image": None}

    resized, scale = resize_for_processing(bgr)
    gray, edges = preprocess_edges(resized)

    calib_result = find_calibration_square(edges, gray)
    if calib_result is None:
        display = save_display_image(resized, "error_calibration", "Calibration square not detected")
        return {
            "success": False,
            "error": "Calibration square not detected. Please ensure the printed 100 mm x 100 mm grid is fully visible, flat, and well-lit in the image.",
            "display_image": display,
        }

    calibration_quad, side_length_px, pixels_per_mm = calib_result
    if pixels_per_mm <= 0:
        display = save_display_image(resized, "error_calibration", "Calibration invalid")
        return {"success": False, "error": "Invalid calibration data detected.", "display_image": display}

    bag_result = find_bag_rect(edges, gray, calibration_quad)
    if bag_result is None:
        display = save_display_image(resized, "error_bag", "Bag not detected")
        return {
            "success": False,
            "error": "Could not detect a rectangular object (bag). Please make sure the bag is clearly visible and not merged with the background.",
            "display_image": display,
        }

    bag_rect, width_px, height_px = bag_result
    width_px, height_px = sorted([width_px, height_px], reverse=True)

    width_px_orig = width_px / scale
    height_px_orig = height_px / scale
    pixels_per_mm_orig = pixels_per_mm / scale

    width_mm = width_px_orig / pixels_per_mm_orig
    height_mm = height_px_orig / pixels_per_mm_orig
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

        display_image = result.get("display_image")
        error_message = result.get("error", "Processing failed.")
        if display_image:
            display_url = url_for("static", filename=f"results/{display_image}")
            error_message = Markup(
                f'{error_message}<br><img src="{display_url}" alt="Uploaded image" style="max-width:100%;margin-top:10px;"/>'
            )

        return render_template("result.html", filename=filename, error_message=error_message)

    flash("Unsupported file type. Please upload an image file.")
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
