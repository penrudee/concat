import torch
import yaml 
import torch.nn as nn
import os
import cv2
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from utils.general import non_max_suppression
from deep_sort_pytorch.deep_sort import DeepSort
from models.experimental import attempt_load,attempt_download
from models.yolo import Model as OriginalModel
from utils.segment.general import  scale_coords

from utils.torch_utils import select_device


from models.common import Concat
    


class FixedModel(OriginalModel):
    def __init__(self, cfg_path, *args, **kwargs):
        self.yaml = None
        with open(cfg_path, 'r') as f:
            self.yaml = yaml.safe_load(f)

        super().__init__(self.yaml, *args, **kwargs)
        
        for i, (module_type, module) in enumerate(zip(self.yaml['backbone'] + self.yaml['head'], self.model)):
            if module_type[2] == 'Concat':
                self.model[i] = Concat()
        

# Load the pre-trained vehicle recognition model (you can replace this with any model you choose)
vehicle_recognition_model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)
vehicle_recognition_model = vehicle_recognition_model.to(select_device())

# Initialize DeepSORT
deepsort = DeepSort("deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7")

# Your existing code to read frames from CCTV feeds
video_sources = ["mojo1.mp4","mojo2.mp4"]
video_captures = [cv2.VideoCapture(src) for src in video_sources]

yolo_pt = os.path.join("c:\\","Users","DRUG INNOVATION CORP","multi_tracking","yolov5","yolov5s.pt")
# C:\Users\DRUG INNOVATION CORP\multi_tracking\yolov5\models\yolov5s.yaml
# cfg = os.path.join("c:\\","Users","DRUG INNOVATION CORP","multi_tracking","yolov5","models","yolov5s.yaml") #version 1
cfg = os.path.join("c:\\", "Users", "DRUG INNOVATION CORP", "multi_tracking", "yolov5", "models", "yolov5s_concat.yaml") #Version2


# model = attempt_load(weights, map_location=torch.device('cuda:0') if torch.cuda.is_available() else 'cpu')


device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'
#model = FixedModel(cfg).to(device)#version1
model = OriginalModel(cfg).to(device) #version2
# attempt_download("https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5s.pt", fname="yolov5s.pt")
ckpt = torch.load(yolo_pt, map_location=device)  # Load the weights
model.load_state_dict(ckpt['model'].float().state_dict())  # Load the state dict

model.float()

def compare_features(features, known_vehicle_features, threshold=0.8):
    similarity_scores = cosine_similarity(features, known_vehicle_features)
    max_similarity = np.max(similarity_scores)
    if max_similarity >= threshold:
        vehicle_index = np.argmax(similarity_scores)
        return True, vehicle_index
    else:
        return False, None

def detect_objects(frame):
    img_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).cuda() if torch.cuda.is_available() else img_tensor

    with torch.no_grad():
        detections = model(img_tensor)[0]
        detections = non_max_suppression(detections, conf_thres=0.25, iou_thres=0.45, classes=[0, 2])

    return img_tensor,detections[0]



def extract_vehicle_features(vehicle_image):
    with torch.no_grad():
        vehicle_tensor = torch.from_numpy(vehicle_image).permute(2, 0, 1).float() / 255.0
        vehicle_tensor = vehicle_tensor.unsqueeze(0).cuda() if torch.cuda.is_available() else vehicle_tensor
        features = vehicle_recognition_model(vehicle_tensor)
    return features
def resize_image(image, target_size):
    height, width = image.shape[:2]
    scale = min(target_size / width, target_size / height)
    new_width, new_height = int(width * scale), int(height * scale)
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    return resized_image
while True:
    for i, cap in enumerate(video_captures):
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = resize_image(frame, 640)
        if not ret:
            continue

        detections = detect_objects(frame)
        detections, img_tensor = detect_objects(frame)
        if detections is not None:
            detections[:, :4] = scale_coords(img_tensor.shape[2:], detections[:, :4], frame.shape).round()
            bbox_xywh = []
            confs = []
            for *xyxy, conf, cls in detections:
                x1, y1, x2, y2 = map(int, xyxy)
                vehicle_image = frame[y1:y2, x1:x2]
                features = extract_vehicle_features(vehicle_image)
                # Use the features to compare with known vehicles and filter detections
                # ...
                bbox_xywh.append([x1, y1, x2 - x1, y2 - y1])
                confs.append(conf.item())

            bbox_xywh = np.array(bbox_xywh)
            confs = np.array(confs)

            if bbox_xywh.shape[0] > 0:
                indices = deepsort.update(bbox_xywh, confs, frame)

                for idx, (x1, y1, x2, y2) in enumerate(bbox_xywh[indices]):
                    cv2.rectangle(frame, (x1, y1), (x2 + x1, y2 + y1), (0, 255, 0), 2)
                    cv2.putText(frame, f"{idx}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.imshow(f"Camera {i}", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

for cap in video_captures:
    cap.release()

cv2.destroyAllWindows()


