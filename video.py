import sys
import cv2
from ultralytics import YOLO
model = YOLO("best.pt")
video_path = "/datashare/HW1/ood_video_data/surg_1.mp4"
output_path = "video_output.avi"
# --- Open video ---
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Failed to open video: {video_path}")
    sys.exit(1)

# --- Get video properties ---
fps = int(cap.get(cv2.CAP_PROP_FPS))
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# --- Prepare video writer ---
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

print(f"Processing video: {video_path}")
print(f"Saving output to: {output_path}")

# --- Frame-by-frame prediction ---
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Predict and annotate frame
    results = model.predict(source=frame, conf=0.25, verbose=False)
    annotated_frame = results[0].plot()  # draw boxes

    # Show and save

    out.write(annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Cleanup ---
cap.release()
out.release()
cv2.destroyAllWindows()
print("Finished.")