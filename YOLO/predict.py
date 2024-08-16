from collections import defaultdict

import cv2
from ultralytics import YOLO

# Load the model
model_path = "../runs/detect/train92/weights/best.pt"
model = YOLO(model_path)

# Read the image
# image_path = '../Tooth Numbering data/train/images/61_jpg.rf.aeeda292aebd31dd9ae28d75d566944d.jpg'
image_path = './testImages/1.jpg'
img = cv2.imread(image_path)
H, W, _ = img.shape

# Predict with the model
results = model(img)

# Extracting detected classes
# detected_classes = []
# for result in results:
#     for item in result.boxes.data:
#         # YOLOv5 returns detections in the format [x1, y1, x2, y2, confidence, class]
#         class_id = int(item[5])  # class ID
#         class_name = model.names[class_id]  # get the class name from class ID
#         detected_classes.append(class_name)
#
# # Print the detected classes
# print("Detected classes:", detected_classes)
#


# Initialize a dictionary to hold the best results per class
best_results = defaultdict(lambda: (None, -1))  # (result, confidence)

# for result in results:
#     boxes = result.boxes.xyxy.cpu().numpy()  # Convert to 27numpy array
#     confidences = result.boxes.conf.cpu().numpy()  # Convert to numpy array
#     labels = result.boxes.cls.cpu().numpy()  # Convert to numpy array
#
#     for i in range(len(boxes)):
#         x1, y1, x2, y2 = boxes[i].astype(int)
#         confidence = confidences[i]
#         label = int(labels[i])
#
#         # # Draw the bounding box
#         cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
#
#         # Put label and confidence score {confidence:.2f}"
#         text = f"{result.names[int(labels[i])]}"
#         cv2.putText(img, text, (int((x1 + x2) / 2), int((y1 + y2) / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12),
#                     2)
#         # cv2.putText(img, text, (int((x1 + x2) / 2), int((y1 + y2) - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
#
# output_image_path = './output.jpg'
# cv2.imwrite(output_image_path, img)


# # Show the image
# cv2.imshow('Image with Bounding Boxes', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# Iterate through results
for result in results:
    boxes = result.boxes.xyxy.cpu().numpy()  # Convert to numpy array
    confidences = result.boxes.conf.cpu().numpy()  # Convert to numpy array
    labels = result.boxes.cls.cpu().numpy()  # Convert to numpy array

    for i in range(len(boxes)):
        class_id = int(labels[i])
        confidence = confidences[i]
        best_results[i] = (result, confidence, boxes[i])

# Draw bounding boxes for the best results
for class_id, (result, confidence, box) in best_results.items():
    x1, y1, x2, y2 = box.astype(int)

    # Draw the bounding box
    # cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Put label and confidence score
    # text = f"{result.names[ int(result.boxes[0].cls)]} "

    text = f"{result.names[int(result.boxes[class_id].cls)]} "
    cv2.putText(img, text,
                (int((x1 + x2) / 2 -4* int(len(result.names[int(result.boxes[class_id].cls)]))), int((y1 + y2) / 2)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (36, 255, 12), 2,
                lineType=1)

# Save the output image
output_image_path = 'outputYolo2.jpg'
cv2.imwrite(output_image_path, img)

# # Show the image (optional)
# cv2.imshow('Image with Bounding Boxes', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
