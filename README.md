# VisionSDS-HW1
## Instructions
Some paths in the scripts are **specific to the original development machine** and must be updated before running the code on your own system.

Please **review and modify the following paths** in each script before use.

---

###  `video.py`

```python
from ultralytics import YOLO
model = YOLO("best.pt") # 🔧 Update this path
video_path = "/datashare/HW1/ood_video_data/surg_1.mp4"  # 🔧 Update this path
output_path = "video_output.avi"                         # Optional: custom output path
---
###  `predict.py`
```python
from ultralytics import YOLO
DATA_YAML_PATH = "/tmp/pycharm_project_665/data.yaml" # 🔧 Update this path
SAVE_RESULTS_PATH = "/tmp/pycharm_project_665" # 🔧 Update this path
model = YOLO("model.pt") # 🔧 Update this path
---
You can download the trained model weights from the link below (also included in this repository):
🔗 [Download best.pt from Google Drive](https://drive.google.com/file/d/1SCTXQkINXoXBkb2sX41pOyea4_-IrTP4/view?usp=drive_link)

<img src="https://github.com/user-attachments/assets/19cdc65a-777c-4400-8b7f-d6b8fd133b13" alt="Prediction Example" height="250"/>


