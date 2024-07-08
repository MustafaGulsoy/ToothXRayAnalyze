import cv2
from ultralytics import YOLO

# Load the model
model_path = '../runs/detect/train26/weights/best.pt'
model = YOLO(model_path)

# Read the image
# image_path = '../Tooth Numbering data/train/images/61_jpg.rf.aeeda292aebd31dd9ae28d75d566944d.jpg'
image_path = './download.png'
img = cv2.imread(image_path)
H, W, _ = img.shape

# Predict with the model
results = model(img)

# Iterate over the results and draw bounding boxes
for result in results:
    boxes = result.boxes.xyxy.cpu().numpy()  # Convert to 27numpy array
    confidences = result.boxes.conf.cpu().numpy()  # Convert to numpy array
    labels = result.boxes.cls.cpu().numpy()  # Convert to numpy array

    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i].astype(int)
        confidence = confidences[i]
        label = int(labels[i])

        # # Draw the bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Put label and confidence score {confidence:.2f}"
        text = f"{result.names[int(labels[i])]}"
        cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        # cv2.putText(img, text, (int((x1 + x2) / 2), int((y1 + y2) - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

output_image_path = 'outputYolo.jpg'
cv2.imwrite(output_image_path, img)
# # Show the image
# cv2.imshow('Image with Bounding Boxes', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
