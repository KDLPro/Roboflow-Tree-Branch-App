from pathlib import Path
from utils.torch_utils import select_device, time_sync

import sys
import os

#Initialization
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

class DetectionApp:
    def __init__(self):
        self.weights=ROOT / 'yolov5s.pt'  # model.pt path(s)
        self.source=ROOT / 'data/images'  # file/dir/URL/glob, 0 for webcam
        self.data=ROOT / 'data/coco128.yaml'  # dataset.yaml path
        self.imgsz=(416, 416)  # inference size (height, width)
        self.conf_thres=0.25  # confidence threshold
        self.iou_thres=0.45  # NMS IOU threshold
        self.max_det=1000  # maximum detections per image
        self.device = select_device('cpu')  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        self.view_img=False  # show results
        self.save_txt=True  # save results to *.txt
        self.save_conf=False  # save confidences in --save-txt labels
        self.save_crop=False  # save cropped prediction boxes
        self.nosave=False  # do not save images/videos
        self.classes=None  # filter by class: --class 0, or --class 0 2 3
        self.agnostic_nms=False  # class-agnostic NMS
        self.augment=False  # augmented inference
        self.visualize=False  # visualize features
        self.update=False  # update all models
        self.project=ROOT / '__pycache__'  # save results to project/name
        self.exist_ok=False  # existing project/name ok, do not increment
        self.line_thickness=1  # bounding box thickness (pixels)
        self.hide_labels=False  # hide labels
        self.hide_conf=False  # hide confidences
        self.half=False  # use FP16 half-precision inference
        self.dnn=False  # use OpenCV DNN for ONNX inference
        self.model=None  # model file

    def search_pt_file(self):
        pt_exist = False
        for i in os.listdir("./"):
            if i.endswith(".pt"):
                pt_exist = True
                self.weights = i
        return pt_exist
