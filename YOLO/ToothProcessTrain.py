from ultralytics import YOLO
import os
# Load a model

model = YOLO("../runs/detect/train27/weights/best.pt")  # load a pretrained model (recommended for training)

save_dir = "../runs/detect/process/train1"

# Klasör var mı kontrol et, yoksa oluştur
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
# Train the model
results = model.train(data="../bitirme_projesiyolov8Data/data.yaml",
                      epochs=300,
                      imgsz=(1435,768),  # Check if the image size is correct
                      batch=32,
                      optimizer="Adam")