from roboflow import Roboflow
rf = Roboflow(api_key="5L2wdzKisZR03KLYysZfs")
project = rf.workspace().project("tooth-detection-and-numbering-tcckb")
model = project.version(3).model

# infer on a local image
print(model.predict("output.png", confidence=40, overlap=30).json())

# visualize your prediction
# model.predict("your_image.jpg", confidence=40, overlap=30).save("prediction.jpg")

# infer on an image hosted elsewhere
# print(model.predict("URL_OF_YOUR_IMAGE", hosted=True, confidence=40, overlap=30).json())