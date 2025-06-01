from ultralytics import YOLO

model = YOLO("best.pt")

metrics = model.val(
    data="/tmp/pycharm_project_665/data.yaml",
    project="/tmp/pycharm_project_665",  # <- saves under this folder
    name="val_results"                   # <- folder will be /tmp/pycharm_project_665/val_results/
)