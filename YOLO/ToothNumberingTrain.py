from ultralytics import YOLO

# Load a model

model = YOLO("../runs/detect/train26/weights/best.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data="../config.yaml",
                      epochs=200,
                      imgsz=(1435, 768),
                      batch=32,
                      optimizer="Adam",
                      lr=0.01
                      )
