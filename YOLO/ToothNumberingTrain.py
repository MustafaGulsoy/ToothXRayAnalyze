import os

import torch
from ultralytics import YOLO


def main():
    # Ensure the CUDA device is visible
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # or '0,1' for multiple GPUs

    # Check if a GPU is available and set the device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        torch.cuda.set_per_process_memory_fraction(1.0, 0)
    # # Load the model
    model = YOLO("../runs/detect/train91/weights/best.pt")
    # .to(device))  # load a pretrained model (recommended for trainxing)

    # Train the model
    results = model.train(data="../config.yaml",
                          epochs=60,
                          imgsz=640,
                          batch=3,
                          optimizer="Adam",
                          lr0=0.01,
                          device=device  # Set the device for training
                          )


if __name__ == '__main__':
    main()
