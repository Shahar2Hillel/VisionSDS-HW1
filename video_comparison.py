import sys
import cv2
import numpy as np
from ultralytics import YOLO

# === Load Models ===
models = {
    "Baseline": YOLO("baseline.pt").to("cuda"),
    "Improved 1": YOLO("improved_1.pt").to("cuda"),
    "Improved 2": YOLO("improved_2.pt").to("cuda")
}

video_path = "/datashare/HW1/ood_video_data/surg_1.mp4"
output_path = "video_comparison.avi"

# === Open input video ===
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Failed to open video: {video_path}")
    sys.exit(1)

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# === Resize settings ===
scale = 0.5
title_height = 40
scaled_width = int(width * scale)
scaled_height = int(height * scale)

# Final frame dimensions
frame_width = scaled_width * len(models)
frame_height = scaled_height + title_height

# === Set up video writer ===
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

font = cv2.FONT_HERSHEY_SIMPLEX

print(f"Processing video: {video_path}")
print(f"Saving output to: {output_path}")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    annotated_frames = []

    for name, model in models.items():
        results = model.predict(source=frame, conf=0.25, verbose=False)
        annotated = results[0].plot()

        # Resize
        annotated = cv2.resize(annotated, (scaled_width, scaled_height))

        # Add black space for title
        annotated = cv2.copyMakeBorder(annotated, title_height, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        cv2.putText(annotated, name, (10, 30), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        annotated_frames.append(annotated)

    combined = np.hstack(annotated_frames)
    out.write(combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print("Finished.")