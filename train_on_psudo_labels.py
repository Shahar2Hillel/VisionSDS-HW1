import os
import cv2
import yaml
import glob
import shutil
import optuna
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

DATA_ROOT = "/datashare/HW1/labeled_image_data"
classes_path = os.path.join(DATA_ROOT, "classes.txt")
with open(classes_path, "r") as f:
    class_names = [line.strip() for line in f if line.strip()]

def fine_tune_on_psudo_labels(model_path, psudo_labels_path, name):
  model = YOLO(model_path)
  psudo_data_dict = {
    # "train": f"{psudo_labels_path}/images",
    "train": f"images",
    "val": os.path.join("../../../models/pseudo_labels/validation_labeled/images"),
    "nc": len(class_names),
    "names": class_names,
    }

  psudo_data_yaml_path = os.path.join(psudo_labels_path, "data.yaml")
  with open(psudo_data_yaml_path, "w") as f:
      yaml.safe_dump(psudo_data_dict, f, sort_keys=False)

  results = model.train(
          data=psudo_data_yaml_path,    # path to data.yaml
          epochs=100,              # number of epochs
          imgsz=640,              # training image size (pixels)
          batch=16,               # batch size (adjust to your GPU)
          lr0=1e-3,               # initial learning rate
          project=f"{MODEL_PATH}/runs/psudo",# where to save runs
          name=name,# run name
          verbose=False,
          pretrained=True)
  return results



  # fine tune model on psudo labeled data
results_history = []
conf_thresholds = [0.5, 0.6, 0.7, 0.8]

for i, conf_threshold in enumerate(conf_thresholds):
  print(f"conf_threshold: {conf_threshold}")
  name = f"run_{i}_conf_threshold_{conf_threshold}"
  output_path = f"{PSUDO_PATH}/{name}"
  model_path = f"{MODEL_PATH}/runs/psudo/{name}/weights/best.pt"
  best_model_path = model_path if os.path.exists(model_path) else "fine_tune_best.pt"
  results = fine_tune_on_psudo_labels(best_model_path, output_path, name)
  results_history.append((name, results))

# # this is for evaluation of the models but is given in the train process
# results_history = []
# for i, conf_threshold in enumerate(conf_thresholds):
#   print(f"conf_threshold: {conf_threshold}")
#   name = f"run_{i}_conf_threshold_{conf_threshold}"
#   output_path = f"{PSUDO_PATH}/{name}"
#   model = YOLO(f"{MODEL_PATH}/runs/psudo/{name}/weights/best.pt")
#   data_yaml_path = f'{output_path}/data.yaml'
#   results = model.val(data=data_yaml_path, imgsz=640, verbose=True)
#   results_history.append((name, results))


# aggregate results - evalutation
names = list(map(lambda x: x[0], results_history))
aps = list(map(lambda x: x[1].box.map50, results_history))
run_num = list(map(lambda x: float(x.split('_')[1]), names))
conf_thresholds = list(map(lambda x: float(x.split('_')[-1]), names))
mrs = list(map(lambda x: x[1].box.mr, results_history))


# Create the plot
plt.figure(figsize=(10, 6))

# Plot mAP50
plt.plot(run_num, aps, label='mAP50', marker='o')
for i, txt in enumerate(conf_thresholds):
    plt.annotate(f'conf={txt}', (run_num[i], aps[i]), textcoords="offset points", xytext=(0,10), ha='center')

# Plot Mean Recall
plt.plot(run_num, mrs, label='Mean Recall', marker='s')
for i, txt in enumerate(conf_thresholds):
    plt.annotate(f'conf={txt}', (run_num[i], mrs[i]), textcoords="offset points", xytext=(0,-15), ha='center')

# Labels and title
plt.xlabel('Run Number')
plt.ylabel('Score')
plt.title('mAP50 and Mean Recall vs Run Number')
plt.legend()
plt.grid(True)

# Save and show
plt.savefig('mean_recall_map50_vs_run_num.png')
plt.show()

  