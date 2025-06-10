# VisionSDS-HW1
## Project Structure
├── requirments.txt # all requirments needed to run
├── data.yaml - configure to run on train or validation datasets
├── video.py - runs predictions on a video using OpenCV
├── predict.py - runs predictions on an image
├── Final Report.pdf

extras/ # additional code we generated for project
├── create_psudo_labels.py
├── eda (1).ipynb - our code for inital eda (part 1 in report)
├── train_on_psudo_labels.py
├── video_comparison.py - like video.py but can compare models results
├── yolo_fine_tuning.ipynb 

val_results/ # Output of predict.py example, results on validation set

## Models
You can download the trained model weights from the link below (also included in this repository):
🔗 [Download best.pt from Google Drive](https://drive.google.com/drive/folders/1ac2NTu83rvMahezbDF0n47aiCpFbXPuq?usp=sharing)

## Video Results
You can view the output video from the link here:
🔗 [Video Output from Google Drive](https://drive.google.com/file/d/1XoHqAdwqKghfIG6xVhkRrhbZWhkb1dNo/view?usp=sharing)

## Installing Instructions
1. Install the packages in requirments.txt
2. Download 'best.pt' from [Models Folder](https://drive.google.com/drive/folders/1ac2NTu83rvMahezbDF0n47aiCpFbXPuq?usp=sharing)
3. Locate the model in the same folder with predict.py and video.py.
4. Some paths in the scripts are **specific to the original development machine**, Please **review and modify the following paths** in each script before use:
---

###  `video.py`

```python
from ultralytics import YOLO
model = YOLO("best.pt") # 🔧 Update this path
video_path = "/datashare/HW1/ood_video_data/surg_1.mp4"  # 🔧 Update this path
output_path = "video_output.avi"                         # Optional: custom output path
```

###  `predict.py`
```python
from ultralytics import YOLO
DATA_YAML_PATH = "/tmp/pycharm_project_665/data.yaml" # 🔧 Update this path
SAVE_RESULTS_PATH = "/tmp/pycharm_project_665" # 🔧 Update this path
model = YOLO("model.pt") # 🔧 Update this path
``` 
---


Video Output Example:

<img src="https://github.com/user-attachments/assets/6c289d9c-c9ad-499f-89fd-512e0c461cbf" alt="Prediction Example" height="250">
