from PySide6.QtWidgets import * 
from PySide6.QtCore import *
from PySide6.QtGui import *
from pathlib import Path
from detectionvars import DetectionApp

import argparse
import os
import torch
import torch.backends.cudnn as cudnn
import cv2, ctypes



from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync


class MainWindow(QMainWindow, DetectionApp):
    def __init__(self):
        QMainWindow.__init__(self)
        DetectionApp.__init__(self)
        
        # Create thread manager
        self.thread_manager_app = QThreadPool()
  
        self.setWindowTitle("Transformer Obstructions App")

        # Set the geometry and alignment of window
        self.setGeometry(0, 0, 1675, 850)
        center = QScreen.availableGeometry(QApplication.primaryScreen()).center()
        geo = self.frameGeometry()
        geo.moveCenter(center)
        self.move(geo.topLeft())

        # Set icons
        self.setWindowIcon(QIcon('logo.png'))
        self.myappid = 'mycompany.myproduct.subproduct.version' # arbitrary string
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(self.myappid)

        # Create menus and submenus
        self.menu = self.menuBar()
        self.menu.setObjectName("MainMenu")
        
        self.file_menu = self.menu.addMenu("&File")
        self.file_menu.setObjectName("FileMenu")

        self.open_img_but = QAction("&Open Image", self)
        self.open_img_but.setObjectName("FileSelection")
        self.file_menu.addAction(self.open_img_but)
        self.open_img_but.triggered.connect(self.open_file)
        self.open_img_but.setShortcut(Qt.CTRL + Qt.Key_O)

        self.save_img_but = QAction("&Save Prediction to Image", self)
        self.save_img_but.setObjectName("FileSelection")
        self.file_menu.addAction(self.save_img_but)
        self.save_img_but.triggered.connect(self.save_file)
        self.save_img_but.setShortcut(Qt.CTRL + Qt.Key_S)
        
        self.model_menu = self.menu.addMenu("&Model")
        self.model_menu.setObjectName("ModelMenu")

        self.load_model_but = QAction("&Load Model", self)
        self.load_model_but.setObjectName("FileSelection")
        self.model_menu.addAction(self.load_model_but)
        self.load_model_but.triggered.connect(self.load_model)
        self.load_model_but.setShortcut(Qt.CTRL + Qt.Key_L)

        self.set_cuda_but = self.model_menu.addMenu("Set CUDA &Device")
        self.set_cuda_but.setObjectName("FileSubSelection")

        self.cpu_but = QAction("CPU only")
        self.cpu_but.setObjectName("FileSubSelection")
        self.cpu_but.setCheckable(True)
        self.cpu_but.setChecked(True)
        self.cpu_but.triggered.connect(self.cpu_selected)
        self.set_cuda_but.addAction(self.cpu_but)

        self.dev_0_but = QAction("Device 0 only")
        self.dev_0_but.setObjectName("FileSubSelection")
        self.dev_0_but.setCheckable(True)
        self.dev_0_but.triggered.connect(self.dev_0_selected)
        self.set_cuda_but.addAction(self.dev_0_but)

        self.dev_1_but = QAction("Device 1 only")
        self.dev_1_but.setObjectName("FileSubSelection")
        self.dev_1_but.setCheckable(True)
        self.dev_1_but.triggered.connect(self.dev_1_selected)
        self.set_cuda_but.addAction(self.dev_1_but)

        self.dev_2_but = QAction("Device 2 only")
        self.dev_2_but.setObjectName("FileSubSelection")
        self.dev_2_but.setCheckable(True)
        self.dev_2_but.triggered.connect(self.dev_2_selected)
        self.set_cuda_but.addAction(self.dev_2_but)

        self.dev_3_but = QAction("Device 3 only")
        self.dev_3_but.setObjectName("FileSubSelection")
        self.dev_3_but.setCheckable(True)
        self.dev_3_but.triggered.connect(self.dev_3_selected)
        self.set_cuda_but.addAction(self.dev_3_but)
        
        self.predict = self.menu.addMenu("Pr&edict")
        self.predict.setObjectName("PredictMenu")

        self.predict_but = QAction("Do P&rediction", self)
        self.predict_but.setObjectName("FileSelection")
        self.predict.addAction(self.predict_but)
        self.predict_but.triggered.connect(self.create_pred)
        self.predict_but.setShortcut(Qt.CTRL + Qt.Key_R)

        # Image cache label
        self.label = QLabel("No image loaded yet...")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setScaledContents(True)
        self.setCentralWidget(self.label)

        # Image display
        self.img_disp = QHBoxLayout(self.label)
        
        self.orig_img = QLabel(self)
        self.orig_img.setAlignment(Qt.AlignCenter)
        self.img_disp.addWidget(self.orig_img)
        
        self.pred_img = QLabel(self)
        self.pred_img.setAlignment(Qt.AlignCenter)
        self.img_disp.addWidget(self.pred_img)


        # Status bar
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)

        self.show()

        try:
            self.dev_0_selected()
            self.show_msg_timed("CUDA Device 0 selected for running predictions!", 50000)
        except:
            self.show_msg_timed("CPU device selected for running predictions!", 50000)
            self.dev_0_but.setChecked(False)
            self.cpu_but.setChecked(True)
        else:
            self.dev_0_but.setChecked(True)
            
    def disp_loading(self):
        self.show_msg("Loading model...")

    def show_msg(self, msg):
        self.statusBar.showMessage(msg)

    def clr_msg(self):
        self.statusBar.clearMessage()

    def show_msg_timed(self, msg, time = 10000):
        self.statusBar.showMessage(msg, time)

    def cpu_selected(self):
        if not self.cpu_but.isChecked():
            self.cpu_but.setChecked(True)

        if self.dev_0_but.isChecked():
            reverse = not self.dev_0_but.isChecked()
            self.dev_0_but.setChecked(reverse)
            
        if self.dev_1_but.isChecked():
            reverse = not self.dev_1_but.isChecked()
            self.dev_1_but.setChecked(reverse)
            
        if self.dev_2_but.isChecked():
            reverse = not self.dev_2_but.isChecked()
            self.dev_2_but.setChecked(reverse)
            
        if self.dev_3_but.isChecked():
            reverse = not self.dev_3_but.isChecked()
            self.dev_3_but.setChecked(reverse)
            
        self.device = select_device('cpu')
        self.show_msg_timed("CPU device selected for running predictions!")

    def dev_0_selected(self):       
        if not self.dev_0_but.isChecked():
            self.dev_0_but.setChecked(True)

        if self.cpu_but.isChecked():
            reverse = not self.cpu_but.isChecked()
            self.cpu_but.setChecked(reverse)

        if self.dev_1_but.isChecked():
            reverse = not self.dev_1_but.isChecked()
            self.dev_1_but.setChecked(reverse)
            
        if self.dev_2_but.isChecked():
            reverse = not self.dev_2_but.isChecked()
            self.dev_2_but.setChecked(reverse)
            
        if self.dev_3_but.isChecked():
            reverse = not self.dev_3_but.isChecked()
            self.dev_3_but.setChecked(reverse)
            
        try:
            self.device = select_device(0)
            self.show_msg_timed("CUDA Device 0 selected for running predictions!", 50000)
        except:
            reverse = not self.cpu_but.isChecked()
            self.cpu_but.setChecked(reverse)
            reverse = not self.dev_0_but.isChecked()
            self.dev_0_but.setChecked(reverse)
            self.device = select_device('cpu')
            self.show_msg_timed("CUDA Device 0 does not exist. Using CPU device instead to run predictions.", 50000)

    def dev_1_selected(self):          
        if not self.dev_1_but.isChecked():
            self.dev_1_but.setChecked(True)
    
        if self.cpu_but.isChecked():
            reverse = not self.cpu_but.isChecked()
            self.cpu_but.setChecked(reverse)
            
        if self.dev_0_but.isChecked():
            reverse = not self.dev_0_but.isChecked()
            self.dev_0_but.setChecked(reverse)
            
        if self.dev_2_but.isChecked():
            reverse = not self.dev_2_but.isChecked()
            self.dev_2_but.setChecked(reverse)
            
        if self.dev_3_but.isChecked():
            reverse = not self.dev_3_but.isChecked()
            self.dev_3_but.setChecked(reverse)
            
        try:
            self.device = select_device(1)
            self.show_msg_timed("CUDA Device 1 selected for running predictions!", 50000)
        except:
            reverse = not self.dev_1_but.isChecked()
            self.dev_1_but.setChecked(reverse)
            try:
                reverse = not self.dev_0_but.isChecked()
                self.dev_0_but.setChecked(reverse)
                self.device = select_device(0)
                self.show_msg_timed("CUDA Device 1 does not exist. Using CUDA Device 0 instead to run predictions.", 50000)
            except:
                reverse = not self.cpu_but.isChecked()
                self.cpu_but.setChecked(reverse)
                reverse = not self.dev_0_but.isChecked()
                self.dev_0_but.setChecked(reverse)
                self.device = select_device('cpu')
                self.show_msg_timed("CUDA Devices 1 and 0 does not exist. Using CPU device instead to run predictions.", 50000)

    def dev_2_selected(self):          
        if not self.dev_2_but.isChecked():
            self.dev_2_but.setChecked(True)
    
        if self.cpu_but.isChecked():
            reverse = not self.cpu_but.isChecked()
            self.cpu_but.setChecked(reverse)
            
        if self.dev_0_but.isChecked():
            reverse = not self.dev_0_but.isChecked()
            self.dev_0_but.setChecked(reverse)
            
        if self.dev_1_but.isChecked():
            reverse = not self.dev_1_but.isChecked()
            self.dev_1_but.setChecked(reverse)
            
        if self.dev_3_but.isChecked():
            reverse = not self.dev_3_but.isChecked()
            self.dev_3_but.setChecked(reverse)
            
        try:
            self.device = select_device(2)
            self.show_msg_timed("CUDA Device 2 selected for running predictions!", 50000)
        except:
            reverse = not self.dev_2_but.isChecked()
            self.dev_2_but.setChecked(reverse)
            try:
                reverse = not self.dev_0_but.isChecked()
                self.dev_0_but.setChecked(reverse)
                self.device = select_device(0)
                self.show_msg_timed("CUDA Device 2 does not exist. Using CUDA Device 0 instead to run predictions.", 50000)
            except:
                reverse = not self.cpu_but.isChecked()
                self.cpu_but.setChecked(reverse)
                reverse = not self.dev_0_but.isChecked()
                self.dev_0_but.setChecked(reverse)
                self.device = select_device('cpu')
                self.show_msg_timed("CUDA Devices 2 and 0 does not exist. Using CPU device instead to run predictions.", 50000)

    def dev_3_selected(self):          
        if not self.dev_3_but.isChecked():
            self.dev_3_but.setChecked(True)
    
        if self.cpu_but.isChecked():
            reverse = not self.cpu_but.isChecked()
            self.cpu_but.setChecked(reverse)
            
        if self.dev_0_but.isChecked():
            reverse = not self.dev_0_but.isChecked()
            self.dev_0_but.setChecked(reverse)
            
        if self.dev_1_but.isChecked():
            reverse = not self.dev_1_but.isChecked()
            self.dev_1_but.setChecked(reverse)
            
        if self.dev_2_but.isChecked():
            reverse = not self.dev_2_but.isChecked()
            self.dev_2_but.setChecked(reverse)
            
        try:
            self.device = select_device(3)
            self.show_msg_timed("CUDA Device 3 selected for running predictions!", 50000)
        except:
            reverse = not self.dev_3_but.isChecked()
            self.dev_3_but.setChecked(reverse)
            try:
                reverse = not self.dev_0_but.isChecked()
                self.dev_0_but.setChecked(reverse)
                self.device = select_device(0)
                self.show_msg_timed("CUDA Device 3 does not exist. Using CUDA Device 0 instead to run predictions.", 50000)
            except:
                reverse = not self.cpu_but.isChecked()
                self.cpu_but.setChecked(reverse)
                reverse = not self.dev_0_but.isChecked()
                self.dev_0_but.setChecked(reverse)
                self.device = select_device('cpu')
                self.show_msg_timed("CUDA Devices 3 and 0 does not exist. Using CPU device instead to run predictions.", 50000)
        
    def disableActions(self):
        # Disable other actions
        self.open_img_but.setEnabled(False)
        self.save_img_but.setEnabled(False)
        self.load_model_but.setEnabled(False)
        self.predict_but.setEnabled(False)
        self.set_cuda_but.setEnabled(False)

    def enableActions(self):    
        # Enable other actions
        self.open_img_but.setEnabled(True)
        self.save_img_but.setEnabled(True)
        self.load_model_but.setEnabled(True)
        self.predict_but.setEnabled(True)
        self.set_cuda_but.setEnabled(True)

    def open_file(self):
        try:
            source = QFileDialog.getOpenFileName(self, 'Load Image', '',
                                            "Image Files (*.png *.jpg *.bmp)")
            self.disableActions()
            temp_source = source[0]
            self.show_msg("Loading " + temp_source + "...")

            # Save a 416x416 copy of original image to cache
            self.image = cv2.imread(temp_source)
            new_width = 416
            new_height = 416
            resized_copy_1 = cv2.resize(self.image, (new_width, new_height), interpolation= cv2.INTER_LINEAR)
            cv2.imwrite("__pycache__/copy1.png", resized_copy_1)
            
            # Save a 832x832 copy of original image to cache
            new_width = 832
            new_height = 832
            resized_copy_2 = cv2.resize(self.image, (new_width, new_height), interpolation= cv2.INTER_LINEAR)
            cv2.imwrite("__pycache__/copy2.png", resized_copy_2)
            
            # Save another 832x832 copy of original image to cache
            new_width = 832
            new_height = 832
            resized_copy_2 = cv2.resize(self.image, (new_width, new_height), interpolation= cv2.INTER_LINEAR)
            cv2.imwrite("__pycache__/disp.png", resized_copy_2)
            
            self.show_msg_timed("Image loaded!")

            self.label.setText('')
            self.left_pixmap = QPixmap("__pycache__/disp.png")
            self.orig_img.setPixmap(self.left_pixmap)
            self.orig_img.setScaledContents(True)
            self.enableActions()

            self.source = temp_source
        except:
            self.enableActions()
            self.show_msg_timed("Canceled loading operation.")


    def load_model(self):
        try:
            self.disableActions()
            temp_model = QFileDialog.getOpenFileName(self, 'Load YOLOv5 Model', '',
                                            "PyTorch Checkpoint (*.pt)")
            self.weights = temp_model[0]
            if self.weights == "":      
                self.enableActions()
                self.show_msg_timed("Canceled loading operation.")
                return
            self.thread_manager_app.start(self.create_model)
        except:
            self.enableActions()
            self.show_msg_timed("Canceled loading operation.")

    def create_model(self):
        try:
            self.thread_manager_app.start(self.disp_loading)
            self.model = DetectMultiBackend(self.weights, device=self.device, dnn=self.dnn,
                                            data=self.data, fp16=self.half)
            self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
            self.imgsz = check_img_size(self.imgsz, s=self.stride)  # check image size
        except:
            self.enableActions()
            self.show_msg_timed("Model can't be loaded. Please try again.")
        else:
            self.enableActions()
            self.show_msg_timed("Model loaded successfully!")

    def create_pred(self):
        try:
            self.disableActions()
            self.thread_manager_app.start(self.start_predictions)        
            self.enableActions()
            
        except:
            self.enableActions()
            self.show_msg_timed("Please load the model and/or image first!")

    def start_predictions(self):
        self.cached_source = "__pycache__/copy1.png"
        self.show_msg("Creating predictions for " + self.source + "...")

        self.model = DetectMultiBackend(self.weights, device=self.device, dnn=self.dnn,
                                        data=self.data, fp16=self.half)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.imgsz = check_img_size(self.imgsz, s=self.stride)  # check image size
        save_dir = Path(self.project)  # increment run

        dataset = LoadImages(self.cached_source, img_size=self.imgsz, stride=self.stride, auto=self.pt)
        bs = 1  # batch_size
        vid_path, vid_writer = [None] * bs, [None] * bs
        self.model.warmup(imgsz=(1 if self.pt else bs, 3, *self.imgsz))  # warmup
        dt, seen = [0.0, 0.0, 0.0], 0
        for path, im, im0s, vid_cap, s in dataset:
            t1 = time_sync()
            im = torch.from_numpy(im).to(self.device)
            im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            t2 = time_sync()
            dt[0] += t2 - t1

            # Inference
            self.visualize = increment_path(save_dir / Path(path).stem) if self.visualize else False
            self.pred = self.model(im, augment=self.augment, visualize=self.visualize)
            t3 = time_sync()
            dt[1] += t3 - t2

            # NMS
            self.pred = non_max_suppression(self.pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)
            dt[2] += time_sync() - t3

            # Process predictions
            for i, det in enumerate(self.pred):  # per image
                seen += 1
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
                
                p = Path("copy1.png")  # to Path
                save_path = str(save_dir / p.name)  # im.png
                p = Path("annot")  # to Path
                txt_path = str(save_dir / p.name)

                s += '%gx%g ' % im.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy() if self.save_crop else im0  # for save_crop
                annotator = Annotator(im0, line_width=self.line_thickness, font_size=1, example=str(self.names))

                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if self.save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if self.save_conf else (cls, *xywh)  # label format
                            with open(f'{txt_path}.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        
                        c = int(cls)  # integer class
                        label = None if self.hide_labels else (self.names[c] if self.hide_conf else f'{self.names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
 
                else:
                    cv2.imwrite(save_path, im0)
                    self.show_msg_timed("Predictions created for " + self.source + "! The model detected nothing.") # Show predictions on right pixmap

                    self.right_pixmap = QPixmap("__pycache__/copy2.png")
                    self.pred_img.setPixmap(self.right_pixmap)
                    self.pred_img.setScaledContents(True)

                    # Save a copy of original image with annotations to cache
                    self.image = cv2.imread(self.source)
                    new_width = self.image.shape[1]
                    new_height = self.image.shape[0]
                    self.predicted_image = cv2.imread(save_path)
                    resized_copy_1 = cv2.resize(self.predicted_image, (new_width, new_height), interpolation= cv2.INTER_LINEAR)
                    cv2.imwrite("__pycache__/copy3.png", resized_copy_1)

                    return
                    
                cv2.imwrite(save_path, im0)

                # Print time (inference-only)
                LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')# Print results
            
            t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
            LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *self.imgsz)}' % t)
            logs = "Predictions created for " + self.source + "! Detected " + f"{n} {self.names[int(c)]}{'s' * (n > 1)}."           

        copied_source = "__pycache__/copy2.png"
        dataset = LoadImages(copied_source, img_size=self.imgsz, stride=self.stride, auto=self.pt)

        for path, im, im0s, vid_cap, s in dataset:
            im = torch.from_numpy(im).to(self.device)
            im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            
            for i, det in enumerate(self.pred):  # per image
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
                p = Path("copy2.png")  # to Path
                save_path = str(save_dir / p.name)  # im.png

                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy() if self.save_crop else im0  # for save_crop
                annotator = Annotator(im0, line_width=self.line_thickness, font_size=1, example=str(self.names))

                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if self.save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if self.save_conf else (cls, *xywh)  # label format
                            with open(f'{txt_path}.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        
                        c = int(cls)  # integer class
                        label = None if self.hide_labels else (self.names[c] if self.hide_conf else f'{self.names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))

                cv2.imwrite(save_path, im0)

                p = Path("copy3.png")  # to Path
                save_path = str(save_dir / p.name)  # im.png

                cv2.imwrite(save_path, im0)

        self.show_msg_timed(logs)

        # Show predictions on right pixmap
        self.right_pixmap = QPixmap("__pycache__/copy2.png")
        self.pred_img.setPixmap(self.right_pixmap)
        self.pred_img.setScaledContents(True)

        # Save a copy of original image with annotations to cache
        self.image = cv2.imread(self.source)
        new_width = self.image.shape[1]
        new_height = self.image.shape[0]
        self.predicted_image = cv2.imread(save_path)
        resized_copy_1 = cv2.resize(self.predicted_image, (new_width, new_height), interpolation= cv2.INTER_LINEAR)
        cv2.imwrite("__pycache__/copy3.png", resized_copy_1)

    def save_file(self):
        self.clr_msg()
        for i in os.listdir("__pycache__/"):
            if i == "copy3.png":
                self.save_predictions()
                return
        self.show_msg_timed("No image saved to cache!")
        

    def save_predictions(self):
        self.save_dialog = QFileDialog()
        self.save_dialog.setFilter(self.save_dialog.filter() | QDir.Hidden)
        self.save_dialog.setDefaultSuffix('png')
        self.save_dialog.setAcceptMode(QFileDialog.AcceptSave)
        save_filters = ["Portable Network Graphics (*.png)", "Graphic Interchange Format (*.gif)",
                        "Joint Photographic Experts Group (*.jpg)", "Joint Photographic Experts Group (*.jpeg)",
                        "Windows Bitmap (*.bmp)"]
        self.save_dialog.setNameFilters(save_filters)
        if self.save_dialog.exec_() == QDialog.Accepted:
            self.target_image = self.save_dialog.selectedFiles()
            self.target_img_loc = self.target_image[0]
        
        self.predicted_image = cv2.imread("__pycache__/copy3.png")
        cv2.imwrite(self.target_img_loc, self.predicted_image)

        self.show_msg_timed("Image saved successfully to " + self.target_img_loc +"!")
                
            
        
        
        

    





