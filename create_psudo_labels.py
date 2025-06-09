import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from ultralytics import YOLO

MODEL_PATH = 'models'
os.makedirs(MODEL_PATH, exist_ok=True) # Create the directory if it doesn't exist
SKIP_FRAMES = 5 # For speed skip some frames
PSUDO_PATH = f"{MODEL_PATH}/pseudo_labels"
FRAME_SAVE_DIR = f"{PSUDO_PATH}/images"
LABEL_SAVE_DIR = f"{PSUDO_PATH}/labels"

os.makedirs(FRAME_SAVE_DIR, exist_ok=True)
os.makedirs(LABEL_SAVE_DIR, exist_ok=True)

def filter_bboxes(results, conf_threshold):
    high_conf_boxes = []
    high_conf_classes = []
    high_conf_scores = []
    for box in results.boxes:
      conf = float(box.conf.cpu().numpy())
      if conf > conf_threshold:
        x1, y1, x2, y2 = map(float, box.xyxy.cpu().numpy().reshape(-1).tolist())
        cls_idx = int(box.cls.cpu().numpy())
        high_conf_boxes.append((x1, y1, x2, y2))
        high_conf_classes.append(cls_idx)
        high_conf_scores.append(conf)
    if len(high_conf_boxes) < len(results):
      # If no boxes pass the threshold, return empty lists
      return [], [], []
    return high_conf_boxes, high_conf_classes, high_conf_scores


def save_frame(frame, frame_idx, frame_save_dir, names_prefix):
    img_filename = f"{names_prefix}_{frame_idx:06d}.jpg"  # e.g. 000123.jpg
    img_path = os.path.join(frame_save_dir, img_filename)
    cv2.imwrite(img_path, frame)
    return img_filename

def save_label(frame_idx,
               high_conf_boxes,
               high_conf_classes,
               high_conf_scores,
               W,
               H,
               label_save_dir,
               names_prefix):
    label_filename = f"{names_prefix}_{frame_idx:06d}.txt"
    label_path = os.path.join(label_save_dir, label_filename)
    with open(label_path, "w") as f_label:
      for (x1, y1, x2, y2), cls_idx in zip(high_conf_boxes, high_conf_classes):
        # Convert [x1,y1,x2,y2] to normalized [x_center, y_center, w, h]
        box_w = x2 - x1
        box_h = y2 - y1
        x_center = x1 + box_w / 2
        y_center = y1 + box_h / 2

        # Normalize by image width/height
        x_c_norm = x_center / W
        y_c_norm = y_center / H
        w_norm = box_w / W
        h_norm = box_h / H

        # Write line: "class_idx x_center y_center w h"
        f_label.write(f"{cls_idx} {x_c_norm:.6f} {y_c_norm:.6f} "
                      f"{w_norm:.6f} {h_norm:.6f}\n")

    return label_filename

def cluster_feature_extraction(frame):
  small = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
  hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
  hist = cv2.calcHist(
      [hsv], [0, 1, 2], None,
      (8, 8, 8),  # e.g. (8,8,8) bins
      [0, 180, 0, 256, 0, 256]
  )
  hist = cv2.normalize(hist, hist).flatten()
  return hist

def create_pseudo_labels(model_path, video_path, conf_threshold, output_path=PSUDO_PATH):
  names_prefix = video_path.split('/')[-1].split('.')[0]
  frame_output_dir = f"{output_path}/images"
  label_output_dir = f"{output_path}/labels"
  os.makedirs(frame_output_dir, exist_ok=True)
  os.makedirs(label_output_dir, exist_ok=True)

  model = YOLO(model_path)
  clustering_featuers = []
  cap = cv2.VideoCapture(video_path)
  total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  print(f"Total frames: {total_frames}")
  with tqdm(total=total_frames // SKIP_FRAMES + 1, desc="Frames processed") as pbar:
    frame_idx = 0
    while True:
      ret, frame = cap.read()
      if not ret:
        break

      if frame_idx % SKIP_FRAMES != 0:
        frame_idx += 1
        continue

      H, W = frame.shape[:2]

      results = model(frame, verbose=False)
      results = results[0]

      high_conf_boxes, high_conf_classes, high_conf_scores = filter_bboxes(results, conf_threshold)

      if len(high_conf_boxes) == 0:
              frame_idx += 1
              pbar.update(1)
              continue

      img_filename = save_frame(frame, frame_idx, frame_output_dir, names_prefix)
      label_filename = save_label(frame_idx,
                                  high_conf_boxes,
                                  high_conf_classes,
                                  high_conf_scores,
                                  W,
                                  H,
                                  label_output_dir,
                                  names_prefix)


      clustering_featuers.append((img_filename,
                                  label_filename,
                                  cluster_feature_extraction(frame)))

      frame_idx += 1
      pbar.update(1)

  cap.release()
  return clustering_featuers




model_path = f"fine_tune_best.pt" # Path to the fine-tuned YOLO model on labels
video_path1 = f"/datashare/HW1/id_video_data/20_2_24_1.mp4" # In Distribution
video_path2 = f"/datashare/HW1/id_video_data/4_2_24_B_2.mp4" # In Distribution
video_path3 = f"/datashare/HW1/ood_video_data/4_2_24_A_1.mp4" # Out of Distribution

conf_thresholds = [0.5, 0.6, 0.7, 0.8]
# video_paths = [video_path1, video_path2]  # for In Distribution
video_paths = [video_path3]  # for (also) Out of Distribution


# create psudo dataset for each threshold:
for i, conf_threshold in enumerate(conf_thresholds):
  print(f"conf_threshold: {conf_threshold}")
  name = f"run_{i}_conf_threshold_{conf_threshold}"
  output_path = f"{PSUDO_PATH}/{name}"
  for video_path in video_paths:
    clustering_featuers = create_pseudo_labels(model_path,
                                               video_path,
                                               conf_threshold,
                                               output_path)
