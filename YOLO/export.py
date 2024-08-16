# import argparse
# from torch_utils import select_device
#
# from models import *
# from datasets import *
# from general import *
#
#
# if __name__ == '__main__':
#     torch.cuda.empty_cache()
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--weights', type=str, default='./yolov4.pt', help='weights path')
#     parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='image size')
#     parser.add_argument('--batch-size', type=int, default=1, help='batch size')
#     parser.add_argument('--cfg', type=str, default=1, help='batch size')
#     opt = parser.parse_args()
#     opt.img_size *= 2 if len(opt.img_size) == 1 else 1  # expand
#     cfg, imgsz, weights = \
#         opt.cfg, opt.img_size, opt.weights
#
#     # Input
#     device = select_device('0')
#     img = torch.zeros((opt.batch_size, 3, *opt.img_size), device=device)  # image size(1,3,320,192) iDetection
#     # Load PyTorch model
#     model = Darknet(cfg, imgsz).cuda()
#     model.load_state_dict(torch.load(weights, map_location=device)['model'])
#     # model = TempModel()
#     # model = torch.load_state_dict(torch.load(opt.weights))
#     model.eval()
#     # model.model[-1].export = True  # set Detect() layer export=True
#     y = model(img)  # dry run
#
#     # ONNX export
#     try:
#         import onnx
#
#         print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
#         f = opt.weights.replace('.pt', '.onnx')  # filename
#         model.fuse()  # only for ONNX
#         torch.onnx.export(model, img, f, verbose=False, opset_version=12, input_names=['images'],
#                           output_names=['classes', 'boxes'] if y is None else ['output'])
#
#         # Checks
#         onnx_model = onnx.load(f)  # load onnx model
#         onnx.checker.check_model(onnx_model)  # check onnx model
#         print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model
#         print('ONNX export success, saved as %s' % f)
#     except Exception as e:
#         print('ONNX export failure: %s' % e)
import base64

import requests
headers = {
    'Authorization': f'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJodHRwOi8vc2NoZW1hcy54bWxzb2FwLm9yZy93cy8yMDA1LzA1L2lkZW50aXR5L2NsYWltcy9zaWQiOiIyMTUzNDgzOTMiLCJuYW1laWQiOiIxMTY1MSIsInVuaXF1ZV9uYW1lIjoiUEFDUyIsImdpdmVuX25hbWUiOiJQQUNTIiwiaHR0cDovL2thcmRlbGVueWF6aWxpbS5jb20vd3MvMjAxOS8xMi9pZGVudGl0eS9jbGFpbXMvSW5zdHV0aW9uSWQiOiIxIiwicm9sZSI6WyJ1c2VyIiwic3lzQWRtaW4iXSwibmJmIjoxNzIzMDk2OTY5LCJleHAiOjE3MjMwOTgxNjksImlhdCI6MTcyMzA5Njk2OX0.O0cfYBKp7_yPQQ9fBJdxRdC5OfftObg_gJPonisWSsQ'
}

data = requests.get("https://pacs.konyasm.gov.tr:30028/gateway/pacs/dicom-web/wado?requestType=WADO&studyUID=1.2.840.20240806.114840.059.202408061131208.1&seriesUID=1.2.840.20240806.114840.059.202408061131208.2&objectUID=1.2.840.10008.20240806115031208&contentType=image/jpeg",
                    headers=headers)
base64_encoded = base64.b64encode(data.content)
base64_string = base64_encoded.decode('utf-8')
print(data)
