# Collaborative Deep Learning Model for Tooth Segmentation and Identification

## Proje Açıklaması
Bu proje, panoramik radyografiler kullanarak diş segmentasyonu ve tanımlama için işbirlikçi bir derin öğrenme modeli geliştirmeyi amaçlamaktadır. Model, iki bağımsız modelin (diş segmentasyonu ve diş tanımlama) çıktısını birleştirerek daha iyi sonuçlar elde etmeyi amaçlamaktadır.

## Gerekli Modellerin İndirilmesi
- **Mask R-CNN Modeli:** [Mask R-CNN Model Weights](https://github.com/matterport/Mask_RCNN/releases)
- **U-Net Modeli:** Kendi U-Net modelinizi eğitmek için [bu örnekten](https://github.com/zhixuhao/unet) faydalanabilirsiniz.
- **Faster R-CNN Modeli:** [Detectron2 Model Zoo](https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md)
- **YOLO-v5 Modeli:** [YOLOv5 Model Weights](https://github.com/ultralytics/yolov5/releases)

## Proje Yapısı
project_root/\
│\
├── data/\
│ ├── train/\
│ ├── val/\
│ ├── test/\
│ └── annotations/\
│\
├── models/\
│ ├── mask_rcnn_model.h5\
│ ├── unet_model.h5\
│ ├── faster_rcnn_model.h5\
│ └── yolo_v5_model.h5\
│\
├── src/\
│ ├── data_loader.py\
│ ├── segmentation_models.py\
│ ├── identification_models.py\
│ ├── collaborative_model.py\
│ └── main.py\
│\
└── README.md\

## Kurulum ve Çalıştırma
1. **Veri setini hazırlayın:**
   - `data/` klasörü altına eğitim, doğrulama ve test veri setlerini yerleştirin.

2. **Modelleri indirin ve `models/` klasörüne yerleştirin:**
   - Mask R-CNN: `mask_rcnn_model.h5`
   - U-Net: `unet_model.h5`
   - Faster R-CNN: `faster_rcnn_model.h5`
   - YOLO-v5: `yolo_v5_model.h5`

3. **Python bağımlılıklarını yükleyin:**
   ```bash
   pip install -r requirements.txt