from ultralytics import YOLO

# Load a model

model = YOLO("../runs/detect/train25/weights/best.pt")  # load a pretrained model (recommended for training)


# Train the model
results = model.train(data="../config.yaml", epochs=20, imgsz=640, batch=32, optimizer="Adam")

