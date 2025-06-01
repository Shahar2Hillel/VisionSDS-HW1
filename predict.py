from ultralytics import YOLO
DATA_YAML_PATH = "/tmp/pycharm_project_665/data.yaml"
SAVE_RESULTS_PATH = "/tmp/pycharm_project_665"
model = YOLO("model.pt")

metrics = model.val(
    data= DATA_YAML_PATH,
    project=SAVE_RESULTS_PATH,  
    name="val_results"                   
)
