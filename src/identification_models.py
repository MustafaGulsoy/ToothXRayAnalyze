import torch
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo


def train_faster_rcnn(train_data, val_data):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("train_dataset",)
    cfg.DATASETS.TEST = ("val_dataset",)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 300
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Diş sınıflandırması için

    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    return cfg


def train_yolo_v5(train_data, val_data):
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    model.train(data='data.yaml', epochs=300)

    model.save('../models/yolov5s.pt')
    return model
