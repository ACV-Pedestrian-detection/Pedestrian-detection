import argparse
import csv
import os
import platform
import sys
from pathlib import Path
import time
from tkinter import Frame, Tk, Label
from PIL import Image, ImageTk
from matplotlib import pyplot as plt
import torch
from IPython.display import display, clear_output
from PIL import Image, ImageFile 
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.utils.plotting import Annotator, colors, save_one_box

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    check_imshow,
    check_requirements,
    colorstr,
    cv2,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    strip_optimizer,
    xyxy2xywh,
)
from utils.torch_utils import select_device, smart_inference_mode


from tkinter import Tk, Label
from PIL import Image, ImageTk
import threading

import cv2
import threading
from tkinter import *
from PIL import Image, ImageTk

import tkinter as tk


class ImageDisplay:
    def __init__(self, source=0, weights="yolov5s.pt", imgsz=640, conf_thres=0.25):
        self.root = tk.Tk()
        self.label = Label(self.root)
        self.label.pack()
        self.source = source
        self.device = select_device()
        self.imgsz = (imgsz, imgsz)
        self.conf_thres = conf_thres
        self.iou_thres = 0.45
        self.classes = None
        self.agnostic_nms = False
        self.augment = False
        self.max_det = 1000
        self.visualize = False
        self.update = False
        self.model = None
        self.weights = weights
        self.root.bind("<Escape>", lambda e: self.root.quit())
        self.root.bind("<space>", lambda e: self.stop())

    def update_image(self, im_rgb):
        im_pil = Image.fromarray(im_rgb)
        tk_image = ImageTk.PhotoImage(im_pil)
        self.label.config(image=tk_image)
        self.label.image = tk_image
        self.root.geometry(f"{im_pil.width}x{im_pil.height}")
        self.root.update()

    def start(self):
        self.root.mainloop()

        if self.model is None:
            self.model = DetectMultiBackend(self.weights, device=self.device)

        stride, names, pt = self.model.stride, self.model.names, self.model.pt
        imgsz = check_img_size(self.imgsz, s=stride)  # check image size
        vid_stride = 1
        project = Path.cwd() / "runs/detect"
        name = "exp"
        exist_ok = False
        save_txt = False
        augment = False
        save_crop = False
        line_thickness = 3
        hide_labels = False  # hide labels
        hide_conf = False  # hide confidences
        save_conf = False  # save confidences in --save-txt labels
        nosave = False  # do not save images/videos
        classes = None  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms = False  # class-agnostic NMS
        augment = False  # augmented inference
        visualize = False  # visualize features
        update = False  # update all models
        save_csv = False  # save a csv file with detections
        view_img = check_imshow()  # check if display is available
        save_img = not nosave and not str(self.source).endswith(".txt")  # save inference images
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        self.dataset = LoadStreams(self.source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(self.dataset)

        # Run inference
        self.model.warmup(imgsz=(1 if pt or self.model.triton else bs, 3, *imgsz))  # warmup
        seen, windows, dt = 0, [], (Profile(device=self.device), Profile(device=self.device), Profile(device=self.device))
        for path, im, im0s, vid_cap, s in self.dataset:
            with dt[0]:
                im = torch.from_numpy(im).to(self.model.device)  # to gpu
                im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim
                if self.model.xml and im.shape[0] > 1:
                    ims = torch.chunk(im, im.shape[0], 0)

            # Inference
            with dt[1]:
                visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
                if self.model.xml and im.shape[0] > 1:
                    pred = None
                    for image in ims:
                        if pred is None:
                            pred = self.model(image, augment=augment, visualize=visualize).unsqueeze(0)
                        else:
                            pred = torch.cat((pred, self.model(image, augment=augment, visualize=visualize).unsqueeze(0)), dim=0)
                    pred = [pred, None]
                else:
                    pred = self.model(im, augment=augment, visualize=visualize)
            # NMS
            with dt[2]:
                pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)

            # Second-stage classifier (optional)
            # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

            # Define the path for the CSV file
            csv_path = save_dir / "predictions.csv"

            # Create or append to the CSV file
            def write_to_csv(image_name, prediction, confidence):
                """Writes prediction data for an image to a CSV file, appending if the file exists."""
                data = {"Image Name": image_name, "Prediction": prediction, "Confidence": confidence}
                with open(csv_path, mode="a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=data.keys())
                    if not csv_path.is_file():
                        writer.writeheader()
                    writer.writerow(data)

            # Process predictions
            for i, det in enumerate(pred):  # per image
                seen += 1

                p, im0, frame = path[i], im0s[i].copy(), self.dataset.count
                s += f"{i}:"

                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # im.jpg
                txt_path = str(save_dir / "labels" / p.stem) + ("" if self.dataset.mode == "image" else f"_{frame}")  # im.txt
                s += "%gx%g " % im.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy() if save_crop else im0  # for save_crop
                annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, 5].unique():
                        n = (det[:, 5] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        c = int(cls)  # integer class
                        label = names[c] if hide_conf else f"{names[c]}"
                        confidence = float(conf)
                        confidence_str = f"{confidence:.2f}"

                        if save_csv:
                            write_to_csv(p.name, label, confidence_str)

                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                            with open(f"{txt_path}.txt", "a") as f:
                                f.write(("%g " * len(line)).rstrip() % line + "\n")

                        if save_img or view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = None if hide_labels else (names[c] if hide_conf else f"{names[c]} {conf:.2f}")
                            annotator.box_label(xyxy, label, color=colors(c, True))
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / "crops" / names[c] / f"{p.stem}.jpg", BGR=True)

                # Stream results
                im0 = annotator.result()
                if view_img:
                    self.update_image(im0)

    def stop(self):
        self.root.destroy()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="yolov5s.pt", help="model path or triton URL")
    parser.add_argument("--source", type=str, default="0", help="file/dir/URL/glob/screen/0(webcam)")
    parser.add_argument("--img", type=int, default=640, help="inference size (height, width)")
    parser.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    display = ImageDisplay(source=args.source, weights=args.weights, imgsz=args.img, conf_thres=args.conf)
    display.start()

