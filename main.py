import cv2
import numpy as np
import mediapipe as mp
from collections import deque

# ------------------- Settings -------------------
REFERENCE_REGION = (100, 100, 200, 200)
LEFT_EYE = [7, 33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163]
RIGHT_EYE = [382, 362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381]
LEFT_IRIS = [469, 470, 471, 472]
RIGHT_IRIS = [474, 475, 476, 477]
ROLLING_WINDOW = 10

# Reference color (hex #e9edee)
REF_BRIGHTNESS = 238.0
REF_REDNESS = 0.0
REF_YELLOWNESS = 0.0

# ------------------- Analysis Buffer -------------------
analysis_buffer = deque(maxlen=ROLLING_WINDOW)

# ------------------- Helper Functions -------------------
def get_white_patch_average(frame, region):
    x1, y1, x2, y2 = region
    patch = frame[y1:y2, x1:x2]
    avg_color = np.mean(patch, axis=(0, 1))  # BGR average
    return avg_color

def apply_manual_white_balance(frame, scale):
    corrected = frame.astype(np.float32)
    corrected[:, :, 0] *= scale[0]
    corrected[:, :, 1] *= scale[1]
    corrected[:, :, 2] *= scale[2]
    corrected = np.clip(corrected, 0, 255).astype(np.uint8)
    return corrected

def extract_sclera(frame, face_landmarks, h, w):
    def get_eye_poly(indices):
        return np.array([
            (int(face_landmarks.landmark[idx].x * w), int(face_landmarks.landmark[idx].y * h))
            for idx in indices
        ], dtype=np.int32)

    left_eye_pts = get_eye_poly(LEFT_EYE)
    right_eye_pts = get_eye_poly(RIGHT_EYE)

    mask_eye = np.zeros((h, w), dtype=np.uint8)
    mask_iris = np.zeros((h, w), dtype=np.uint8)

    cv2.fillConvexPoly(mask_eye, left_eye_pts, 255)
    cv2.fillConvexPoly(mask_eye, right_eye_pts, 255)

    cx_l = int(face_landmarks.landmark[468].x * w)
    cy_l = int(face_landmarks.landmark[468].y * h)
    cx_r = int(face_landmarks.landmark[473].x * w)
    cy_r = int(face_landmarks.landmark[473].y * h)

    radius_l = int(np.mean([
        np.linalg.norm((face_landmarks.landmark[i].x * w - cx_l,
                        face_landmarks.landmark[i].y * h - cy_l))
        for i in LEFT_IRIS
    ]))
    radius_r = int(np.mean([
        np.linalg.norm((face_landmarks.landmark[i].x * w - cx_r,
                        face_landmarks.landmark[i].y * h - cy_r))
        for i in RIGHT_IRIS
    ]))

    cv2.circle(mask_iris, (cx_l, cy_l), radius_l, 255, -1)
    cv2.circle(mask_iris, (cx_r, cy_r), radius_r, 255, -1)

    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # mask_iris = cv2.dilate(mask_iris, kernel, iterations=1)

    mask_sclera = cv2.subtract(mask_eye, mask_iris)
    mask_sclera = cv2.GaussianBlur(mask_sclera, (5, 5), 0)

    sclera_only = cv2.bitwise_and(frame, frame, mask=mask_sclera)
    return sclera_only, mask_sclera

def analyze_sclera(sclera_img, sclera_mask):
    hsv = cv2.cvtColor(sclera_img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    mask = sclera_mask > 15
    hue_vals = h[mask]
    sat_vals = s[mask]
    val_vals = v[mask]

    avg_val = np.mean(val_vals)
    red_pixels = np.sum((hue_vals < 15) | (hue_vals > 160))
    yellow_pixels = np.sum((hue_vals >= 15) & (hue_vals < 45))
    total = hue_vals.shape[0]

    redness_ratio = red_pixels / total
    yellowness_ratio = yellow_pixels / total

    # Deviation calculation
    dev_brightness = abs(avg_val - REF_BRIGHTNESS) / REF_BRIGHTNESS * 100
    dev_redness = abs(redness_ratio - REF_REDNESS) * 100
    dev_yellowness = abs(yellowness_ratio - REF_YELLOWNESS) * 100

    return {
        "Brightness Δ%": dev_brightness,
        "Redness Δ%": dev_redness,
        "Yellowness Δ%": dev_yellowness
    }

def average_metrics(buffer):
    if not buffer:
        return None
    keys = buffer[0].keys()
    avg = {k: np.mean([entry[k] for entry in buffer]) for k in keys}
    return avg

def categorize_brightness(dev):
    if dev <= 20:
        return "Normal"
    elif dev <= 45:
        return "Slightly Fatigued"
    else:
        return "Seriously Fatigued"

def categorize_yellowness(dev):
    if dev <= 10:
        return "Normal"
    elif dev <= 35:
        return "Slightly Jaundiced"
    elif dev <= 30:
        return "Jaundiced"
    else:
        return "Severely Jaundiced"

def categorize_redness(dev):
    if dev <= 20:
        return "Normal"
    elif dev <= 35:
        return "Reddened Eye"
    else:
        return "Bloodshot Eye"

def overlay_metrics(img, metrics, x=10, y=30):
    if not metrics:
        return
    spacing = 35

    brightness_label = categorize_brightness(metrics["Brightness Δ%"])
    yellowness_label = categorize_yellowness(metrics["Yellowness Δ%"])
    redness_label = categorize_redness(metrics["Redness Δ%"])

    labels = [
        f"Brightness: {brightness_label}",
        f"Yellowness: {yellowness_label}",
        f"Redness: {redness_label}"
    ]

    for i, label in enumerate(labels):
        # White outline
        cv2.putText(img, label, (x, y + i * spacing),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 4)
        # Black fill
        cv2.putText(img, label, (x, y + i * spacing),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)

def draw_eye_landmarks(img, face_landmarks, w, h):
    def draw_poly(indices, color):
        pts = np.array([
            (int(face_landmarks.landmark[idx].x * w), int(face_landmarks.landmark[idx].y * h))
            for idx in indices
        ], dtype=np.int32)
        cv2.polylines(img, [pts], isClosed=True, color=color, thickness=1)

    draw_poly(LEFT_EYE, (0, 255, 0))       # Green for eye outline
    draw_poly(RIGHT_EYE, (0, 255, 0))
    draw_poly(LEFT_IRIS, (255, 0, 0))      # Blue for iris
    draw_poly(RIGHT_IRIS, (255, 0, 0))

# ------------------- Setup -------------------
cap = cv2.VideoCapture(0)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

calibrated = False
show_sclera = False
show_analysis = False
scale_factors = None
show_sclera_mask = False

print("📸 'c' to calibrate using white paper")
print("🧿 'a' to activate sclera extraction")
print("📊 'd' to toggle display of sclera metrics")
print("❌ ESC to exit")

# ------------------- Main Loop -------------------
while True:
    success, frame = cap.read()
    if not success:
        break

    h, w, _ = frame.shape
    x1, y1, x2, y2 = REFERENCE_REGION
    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        ref_bgr = get_white_patch_average(frame, REFERENCE_REGION)
        scale_factors = 255.0 / ref_bgr
        calibrated = True
        print("✅ Calibrated.")

    if key == ord('a') and calibrated:
        show_sclera = True
        show_sclera_mask = False
        print("🧿 Showing eye landmarks on full image")

    if key == ord('s') and calibrated:
        show_sclera = True
        show_sclera_mask = True
        print("👁️ Showing sclera-only view")

    if key == ord('d') and calibrated and show_sclera:
        show_analysis = not show_analysis
        print(f"📊 Display {'ON' if show_analysis else 'OFF'}")

    if calibrated:
        frame = apply_manual_white_balance(frame, scale_factors)

    output = frame.copy()

    if not show_sclera:
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)

    if calibrated and show_sclera:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                sclera_only, sclera_mask = extract_sclera(frame, face_landmarks, h, w)
                metrics = analyze_sclera(sclera_only, sclera_mask)
                if metrics:
                    analysis_buffer.append(metrics)

                if show_sclera_mask:
                    output = sclera_only
                else:
                    output = frame.copy()
                    draw_eye_landmarks(output, face_landmarks, w, h)

                if show_analysis:
                    avg_metrics = average_metrics(analysis_buffer)
                    overlay_metrics(output, avg_metrics)


    cv2.imshow("Eye Analyzer", output)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
