# 👁️ Real-Time Sclera Analysis using OpenCV & MediaPipe

This project is a **real-time eye analysis tool** built with **Python**, **OpenCV**, and **MediaPipe** that detects and evaluates **scleral discoloration** to assess potential health indicators like **fatigue**, **jaundice**, and **irritation**.

It extracts the **sclera (white part of the eye)**, analyzes its color, and compares it against reference values for a healthy eye — accounting for camera white balance using a calibration step. The system provides **live feedback** with a stable rolling average over recent frames.

---

## 🧠 What It Detects

- **Fatigue**: Dullness and low brightness in the sclera
- **Jaundice**: Yellowish tint from potential bilirubin buildup
- **Redness**: Irritation or inflammation of the eye

---

## 🧰 Tech Stack

| Tool/Library    | Purpose                                      |
|-----------------|----------------------------------------------|
| `OpenCV`        | Video capture, image processing, UI          |
| `MediaPipe`     | Facial landmark detection (eyes, pupils)     |
| `NumPy`         | Array math and color analysis                |
| `deque` (Python) | Queue for smoothing results across frames   |

---

## 🧪 How It Works

1. **Facial Landmark Detection**  
   - Uses **MediaPipe Face Mesh** to locate key eye and iris points

2. **Sclera Segmentation**  
   - Pupil location helps isolate the surrounding **sclera** using geometric filtering with NumPy

3. **White Balance Calibration**  
   - Press `C` and hold a **white sheet** in front of the camera  
   - Ensures consistent color detection despite camera auto-correction

4. **Real-Time Color Analysis**  
   - Brightness, **yellowness**, and **redness** metrics are calculated
   - Results are **averaged over the last 10 frames** using a queue for stability

5. **Visual Feedback**  
   - Eye landmarks are drawn for accuracy checking  
   - Mask view isolates only the sclera for debugging or display

---

## 🎮 Controls

| Key         | Function                                                                 |
|-------------|--------------------------------------------------------------------------|
| `C`         | Calibrate camera white balance with a white paper                       |
| `A`         | Start real-time sclera analysis                                          |
| `S`         | Show sclera-only masked view                                             |
| `D`         | Display on-screen analysis results (brightness, redness, yellowness)     |
| `ESC`       | Exit the application                                                     |

---


---

## 🧑‍💻 Installation

Make sure you have Python 3.7+ and the required libraries:

```bash
pip install opencv-python mediapipe numpy
