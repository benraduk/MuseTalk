from __future__ import print_function
import os
import torch
from torch.utils.model_zoo import load_url
from enum import Enum
import numpy as np
import cv2
import math
import onnxruntime as ort
try:
    import urllib.request as request_file
except BaseException:
    import urllib as request_file

from .models import FAN, ResNetDepth
from .utils import *


class LandmarksType(Enum):
    """Enum class defining the type of landmarks to detect.

    ``_2D`` - the detected points ``(x,y)`` are detected in a 2D space and follow the visible contour of the face
    ``_2halfD`` - this points represent the projection of the 3D points into 3D
    ``_3D`` - detect the points ``(x,y,z)``` in a 3D space

    """
    _2D = 1
    _2halfD = 2
    _3D = 3


class NetworkSize(Enum):
    # TINY = 1
    # SMALL = 2
    # MEDIUM = 3
    LARGE = 4

    def __new__(cls, value):
        member = object.__new__(cls)
        member._value_ = value
        return member

    def __int__(self):
        return self.value



class FaceAlignment:
    def __init__(self, landmarks_type, network_size=NetworkSize.LARGE,
                 device='cuda', flip_input=False, face_detector='sfd', verbose=False):
        self.device = device
        self.flip_input = flip_input
        self.landmarks_type = landmarks_type
        self.verbose = verbose

        network_size = int(network_size)

        if 'cuda' in device:
            torch.backends.cudnn.benchmark = True
#             torch.backends.cuda.matmul.allow_tf32 = False
#             torch.backends.cudnn.benchmark = True
#             torch.backends.cudnn.deterministic = False
#             torch.backends.cudnn.allow_tf32 = True
            print('cuda start')


        # Get the face detector
        face_detector_module = __import__('face_detection.detection.' + face_detector,
                                          globals(), locals(), [face_detector], 0)
        
        self.face_detector = face_detector_module.FaceDetector(device=device, verbose=verbose)

    def get_detections_for_batch(self, images):
        images = images[..., ::-1]
        detected_faces = self.face_detector.detect_from_batch(images.copy())
        results = []

        for i, d in enumerate(detected_faces):
            if len(d) == 0:
                results.append(None)
                continue
            d = d[0]
            d = np.clip(d, 0, None)
            
            x1, y1, x2, y2 = map(int, d[:-1])
            results.append((x1, y1, x2, y2))

        return results
    
    
class YOLOv8_face:
    def __init__(self, path = 'face_detection/weights/yolov8n-face.onnx', conf_thres=0.2, iou_thres=0.5):
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres
        self.class_names = ['face']
        self.num_classes = len(self.class_names)
        
        # Initialize ONNX Runtime session
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(path, providers=providers)
        
        # Get model input details
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.input_height = self.input_shape[2] if len(self.input_shape) > 2 else 640
        self.input_width = self.input_shape[3] if len(self.input_shape) > 3 else 640
        
        print(f"YOLOv8 model loaded: {path}")
        print(f"Input shape: {self.input_shape}")
        print(f"Providers: {self.session.get_providers()}")

    def make_anchors(self, feats_hw, grid_cell_offset=0.5):
        """Generate anchors from features."""
        anchor_points = {}
        for i, stride in enumerate(self.strides):
            h,w = feats_hw[i]
            x = np.arange(0, w) + grid_cell_offset  # shift x
            y = np.arange(0, h) + grid_cell_offset  # shift y
            sx, sy = np.meshgrid(x, y)
            # sy, sx = np.meshgrid(y, x)
            anchor_points[stride] = np.stack((sx, sy), axis=-1).reshape(-1, 2)
        return anchor_points

    def softmax(self, x, axis=1):
        x_exp = np.exp(x)
        # 如果是列向量，则axis=0
        x_sum = np.sum(x_exp, axis=axis, keepdims=True)
        s = x_exp / x_sum
        return s
    
    def resize_image(self, srcimg, keep_ratio=True):
        top, left, newh, neww = 0, 0, self.input_width, self.input_height
        if keep_ratio and srcimg.shape[0] != srcimg.shape[1]:
            hw_scale = srcimg.shape[0] / srcimg.shape[1]
            if hw_scale > 1:
                newh, neww = self.input_height, int(self.input_width / hw_scale)
                img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
                left = int((self.input_width - neww) * 0.5)
                img = cv2.copyMakeBorder(img, 0, 0, left, self.input_width - neww - left, cv2.BORDER_CONSTANT,
                                         value=(0, 0, 0))  # add border
            else:
                newh, neww = int(self.input_height * hw_scale), self.input_width
                img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
                top = int((self.input_height - newh) * 0.5)
                img = cv2.copyMakeBorder(img, top, self.input_height - newh - top, 0, 0, cv2.BORDER_CONSTANT,
                                         value=(0, 0, 0))
        else:
            img = cv2.resize(srcimg, (self.input_width, self.input_height), interpolation=cv2.INTER_AREA)
        return img, newh, neww, top, left

    def detect(self, srcimg):
        # Preprocess image
        input_img = self.preprocess_image(srcimg)
        
        # Run inference
        outputs = self.session.run(None, {self.input_name: input_img})
        
        # Post-process results
        det_bboxes, det_conf, landmarks = self.postprocess_outputs(outputs, srcimg.shape)
        
        # Create dummy class IDs (all faces)
        det_classid = np.zeros(len(det_bboxes), dtype=np.int32) if len(det_bboxes) > 0 else np.array([])
        
        return det_bboxes, det_conf, det_classid, landmarks

    def post_process(self, preds, scale_h, scale_w, padh, padw):
        bboxes, scores, landmarks = [], [], []
        for i, pred in enumerate(preds):
            stride = int(self.input_height/pred.shape[2])
            pred = pred.transpose((0, 2, 3, 1))
            
            box = pred[..., :self.reg_max * 4]
            cls = 1 / (1 + np.exp(-pred[..., self.reg_max * 4:-15])).reshape((-1,1))
            kpts = pred[..., -15:].reshape((-1,15)) ### x1,y1,score1, ..., x5,y5,score5

            # tmp = box.reshape(self.feats_hw[i][0], self.feats_hw[i][1], 4, self.reg_max)
            tmp = box.reshape(-1, 4, self.reg_max)
            bbox_pred = self.softmax(tmp, axis=-1)
            bbox_pred = np.dot(bbox_pred, self.project).reshape((-1,4))

            bbox = self.distance2bbox(self.anchors[stride], bbox_pred, max_shape=(self.input_height, self.input_width)) * stride
            kpts[:, 0::3] = (kpts[:, 0::3] * 2.0 + (self.anchors[stride][:, 0].reshape((-1,1)) - 0.5)) * stride
            kpts[:, 1::3] = (kpts[:, 1::3] * 2.0 + (self.anchors[stride][:, 1].reshape((-1,1)) - 0.5)) * stride
            kpts[:, 2::3] = 1 / (1+np.exp(-kpts[:, 2::3]))

            bbox -= np.array([[padw, padh, padw, padh]])  ###合理使用广播法则
            bbox *= np.array([[scale_w, scale_h, scale_w, scale_h]])
            kpts -= np.tile(np.array([padw, padh, 0]), 5).reshape((1,15))
            kpts *= np.tile(np.array([scale_w, scale_h, 1]), 5).reshape((1,15))

            bboxes.append(bbox)
            scores.append(cls)
            landmarks.append(kpts)

        bboxes = np.concatenate(bboxes, axis=0)
        scores = np.concatenate(scores, axis=0)
        landmarks = np.concatenate(landmarks, axis=0)
    
        bboxes_wh = bboxes.copy()
        bboxes_wh[:, 2:4] = bboxes[:, 2:4] - bboxes[:, 0:2]  ####xywh
        classIds = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)  ####max_class_confidence
        
        mask = confidences>self.conf_threshold
        bboxes_wh = bboxes_wh[mask]  ###合理使用广播法则
        confidences = confidences[mask]
        classIds = classIds[mask]
        landmarks = landmarks[mask]
        
        indices = cv2.dnn.NMSBoxes(bboxes_wh.tolist(), confidences.tolist(), self.conf_threshold,
                                   self.iou_threshold).flatten()
        if len(indices) > 0:
            mlvl_bboxes = bboxes_wh[indices]
            confidences = confidences[indices]
            classIds = classIds[indices]
            landmarks = landmarks[indices]
            return mlvl_bboxes, confidences, classIds, landmarks
        else:
            print('nothing detect')
            return np.array([]), np.array([]), np.array([]), np.array([])

    def distance2bbox(self, points, distance, max_shape=None):
        x1 = points[:, 0] - distance[:, 0]
        y1 = points[:, 1] - distance[:, 1]
        x2 = points[:, 0] + distance[:, 2]
        y2 = points[:, 1] + distance[:, 3]
        if max_shape is not None:
            x1 = np.clip(x1, 0, max_shape[1])
            y1 = np.clip(y1, 0, max_shape[0])
            x2 = np.clip(x2, 0, max_shape[1])
            y2 = np.clip(y2, 0, max_shape[0])
        return np.stack([x1, y1, x2, y2], axis=-1)
    
    def draw_detections(self, image, boxes, scores, kpts):
        for box, score, kp in zip(boxes, scores, kpts):
            x, y, w, h = box.astype(int)
            # Draw rectangle
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), thickness=3)
            cv2.putText(image, "face:"+str(round(score,2)), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=2)
            for i in range(5):
                cv2.circle(image, (int(kp[i * 3]), int(kp[i * 3 + 1])), 4, (0, 255, 0), thickness=-1)
                # cv2.putText(image, str(i), (int(kp[i * 3]), int(kp[i * 3 + 1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), thickness=1)
        return image
    
    def preprocess_image(self, image):
        """Preprocess image for ONNX model (FaceFusion style)"""
        # Step 1: Restrict frame (resize while maintaining aspect ratio)
        temp_vision_frame = self._restrict_frame(image, (self.input_width, self.input_height))
        
        # Calculate ratios for coordinate scaling
        self.ratio_height = image.shape[0] / temp_vision_frame.shape[0]
        self.ratio_width = image.shape[1] / temp_vision_frame.shape[1]
        
        # Step 2: Prepare detect frame (pad to exact size)
        detect_vision_frame = self._prepare_detect_frame(temp_vision_frame)
        
        # Step 3: Normalize (0-1 range)
        detect_vision_frame = self._normalize_detect_frame(detect_vision_frame, [0, 1])
        
        return detect_vision_frame
    
    def _restrict_frame(self, vision_frame, target_resolution):
        """Restrict frame size while maintaining aspect ratio"""
        target_width, target_height = target_resolution
        height, width = vision_frame.shape[:2]
        
        # Calculate scale to fit within target resolution
        scale = min(target_width / width, target_height / height)
        
        if scale < 1.0:
            new_width = int(width * scale)
            new_height = int(height * scale)
            vision_frame = cv2.resize(vision_frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        return vision_frame
    
    def _prepare_detect_frame(self, temp_vision_frame):
        """Prepare detect frame by padding to exact size"""
        detect_vision_frame = np.zeros((self.input_height, self.input_width, 3), dtype=np.uint8)
        detect_vision_frame[:temp_vision_frame.shape[0], :temp_vision_frame.shape[1], :] = temp_vision_frame
        detect_vision_frame = np.expand_dims(detect_vision_frame.transpose(2, 0, 1), axis=0).astype(np.float32)
        return detect_vision_frame
    
    def _normalize_detect_frame(self, detect_vision_frame, normalize_range):
        """Normalize detect frame"""
        if normalize_range == [0, 1]:
            return detect_vision_frame / 255.0
        elif normalize_range == [-1, 1]:
            return (detect_vision_frame - 127.5) / 128.0
        return detect_vision_frame
    
    def postprocess_outputs(self, outputs, original_shape):
        """Post-process ONNX model outputs (FaceFusion style)"""
        bounding_boxes = []
        face_scores = []
        face_landmarks_5 = []
        
        if len(outputs) == 0:
            return np.array([]), np.array([]), np.array([])
        
        # Get the main output: shape should be (1, 20, 8400)
        detection = outputs[0]
        detection = np.squeeze(detection).T  # Remove batch dim and transpose: (8400, 20)
        
        # Split into components: [x, y, w, h, conf, ...landmarks...]
        bounding_boxes_raw, face_scores_raw, face_landmarks_5_raw = np.split(detection, [4, 5], axis=1)
        
        # Filter by confidence threshold
        keep_indices = np.where(face_scores_raw.ravel() > self.conf_threshold)[0]
        
        if len(keep_indices) > 0:
            bounding_boxes_raw = bounding_boxes_raw[keep_indices]
            face_scores_raw = face_scores_raw[keep_indices]
            face_landmarks_5_raw = face_landmarks_5_raw[keep_indices]
            
            # Process bounding boxes (convert from center format to corner format)
            for bounding_box_raw in bounding_boxes_raw:
                # Coordinates are already in pixels, scale them back to original image
                x_center, y_center, width, height = bounding_box_raw
                
                # Convert to corner format and scale to original image
                x1 = (x_center - width / 2) * self.ratio_width
                y1 = (y_center - height / 2) * self.ratio_height
                x2 = (x_center + width / 2) * self.ratio_width
                y2 = (y_center + height / 2) * self.ratio_height
                
                bounding_boxes.append(np.array([x1, y1, x2, y2]))
            
            # Process face scores
            face_scores = face_scores_raw.ravel().tolist()
            
            # Process landmarks (5 points * 3 values = 15 values)
            face_landmarks_5_raw[:, 0::3] = face_landmarks_5_raw[:, 0::3] * self.ratio_width  # x coordinates
            face_landmarks_5_raw[:, 1::3] = face_landmarks_5_raw[:, 1::3] * self.ratio_height  # y coordinates
            
            for face_landmark_raw_5 in face_landmarks_5_raw:
                # Reshape to (5, 3) and take only x,y coordinates (ignore confidence)
                landmarks = face_landmark_raw_5.reshape(-1, 3)[:, :2]
                face_landmarks_5.append(landmarks)
        
        return np.array(bounding_boxes), np.array(face_scores), np.array(face_landmarks_5)
    
    def get_detections_for_batch(self, frames):
        """SFD-compatible batch detection interface"""
        results = []
        for frame in frames:
            det_bboxes, det_conf, det_classid, landmarks = self.detect(frame)
            if len(det_bboxes) > 0 and len(det_conf) > 0 and det_conf[0] > self.conf_threshold:
                # det_bboxes are already in xyxy format from postprocess_outputs
                bbox = det_bboxes[0]
                # Convert numpy array to tuple of integers for pipeline compatibility
                bbox_tuple = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
                results.append(bbox_tuple)
            else:
                results.append(None)
        return results

    def _convert_to_xyxy(self, bbox_xywh):
        """Convert YOLOv8 xywh format to SFD xyxy format"""
        cx, cy, w, h = bbox_xywh
        x1 = cx - w/2
        y1 = cy - h/2
        x2 = cx + w/2
        y2 = cy + h/2
        return (x1, y1, x2, y2)
    
ROOT = os.path.dirname(os.path.abspath(__file__))