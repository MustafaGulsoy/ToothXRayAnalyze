from torch.utils.data import Dataset, DataLoader
import os, torch
from glob import glob
from PIL import Image


class CustomDataset(Dataset):

    def __init__(self, root, transformations=None):
        super().__init__()

        self.transformations = transformations
        self.javob_yulaklari = sorted(glob(f"{root}/yolo-labels/*"))
        self.klasslar_soni = 2
        self.data = {}  # key = rasm_yulagi ; value = bounding boxes

        for idx, javob_yulagi in enumerate(self.javob_yulaklari):

            rasm_yulagi = javob_yulagi.replace(".txt", ".png").replace("yolo-labels", "all-images")
            bboxes = open(f"{javob_yulagi}", "r").read().split("\n")[:-1]
            if len(bboxes) < 1: continue
            self.data[rasm_yulagi] = bboxes

    def get_area(self, bboxes):
        return (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])

    def yolo2cv(self, bbox, im_h, im_w):
        return self.get_coordinates(*[float(bb) for bb in bbox.split(" ")[1:]], im_h, im_w)

    def get_coordinates(self, x, y, w, h, im_h, im_w):
        return [int((x - (w / 2)) * im_w), int((y - (h / 2)) * im_h), int((x + (w / 2)) * im_w),
                int((y + (h / 2)) * im_h)]

    def create_target(self, bboxes, labels, is_crowd, rasm_id, area):

        target = {}
        target["boxes"] = bboxes
        target["labels"] = labels
        target["iscrowd"] = is_crowd
        target["image_id"] = rasm_id
        target["area"] = area

        return target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        rasm_yulagi = list(self.data.keys())[idx]
        rasm = Image.open(rasm_yulagi).convert("RGB")  # 4channels -> 3channels
        im_w, im_h = rasm.size[0], rasm.size[1]
        bboxes = []
        for data in self.data[rasm_yulagi]: bboxes.append(self.yolo2cv(data, im_w=im_w, im_h=im_h))
        bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
        labels = torch.ones((len(bboxes),), dtype=torch.int64)
        is_crowd = torch.zeros((len(bboxes),), dtype=torch.int64)
        rasm_id = torch.tensor([idx])
        area = self.get_area(bboxes)

        target = self.create_target(bboxes, labels, is_crowd, rasm_id, area)

        if self.transformations: rasm, target = self.transformations(rasm, target)

        return rasm, target