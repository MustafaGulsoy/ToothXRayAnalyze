import os

import torch
from ultralytics import YOLO


def main():
    # Load a model
    # Ensure the CUDA device is visible
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # or '0,1' for multiple GPUs

    # Check if a GPU is available and set the device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        torch.cuda.set_per_process_memory_fraction(1.0, 0)

    model = YOLO("../runs/detect/train83/weights/best.pt")  # load a pretrained model (recommended for training)

    # Train the model
    results = model.train(data="../bitirme_projesiyolov8Data/data.yaml",
                          epochs=600,

                          imgsz=640,  #   Check if the image size is correct
                          batch=8,
                          optimizer="Adam",
                          lr0=0.11
                          )


if __name__ == '__main__':
    main()
