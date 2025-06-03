from ultralytics import YOLO
DATA_YAML_PATH = "data.yaml"
SAVE_RESULTS_PATH = ""
model = YOLO("model.pt").to("cuda")

metrics = model.val(
    data= DATA_YAML_PATH,
    project=SAVE_RESULTS_PATH,  
    name="val_results"                   
)
