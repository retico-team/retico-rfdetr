"""
Roboflow Detection Transformer module
=====================================

This module implements the Roboflow Detection Transformer (RFDetr) architecture,
which is designed for object detection tasks.
"""

# Imports
import threading
import numpy as np
import time
from collections import deque
import retico_core
from retico_vision.vision import ImageIU, DetectedObjectsIU
from rfdetr import RFDETRNano, RFDETRSmall, RFDETRMedium, RFDETRLarge
from rfdetr import RFDETRSegNano, RFDETRSegSmall, RFDETRSegMedium, RFDETRSegLarge
import cv2
import supervision as sv
from rfdetr.assets.coco_classes import COCO_CLASSES


class RFDETRModule(retico_core.AbstractModule):

    @staticmethod
    def name():
        return "HuggingFace RFDETR Module"

    @staticmethod
    def description():
        return "An object detection model using RFDETR"

    @staticmethod
    def input_ius():
        return [ImageIU]

    @staticmethod
    def output_iu():
        return DetectedObjectsIU

    MODEL_OPTIONS = {
        "nano": RFDETRNano,
        "small": RFDETRSmall,
        "medium": RFDETRMedium,
        "large": RFDETRLarge,
    }

    SEG_MODEL_OPTIONS = {
        "nano": RFDETRSegNano,
        "small": RFDETRSegSmall,
        "medium": RFDETRSegMedium,
        "large": RFDETRSegLarge,
    }

    def __init__(
        self, model="small", pretrain=None, use_seg=False, show=False, **kwargs
    ):
        """
        Initialize the RFDETR Module
        Args:
            model (str): the name of the RFDETR model will correspond to the model checkpoint
                options: nano, small, medium, large
            pretrain (str): path to custom checkpoint, only needed if wanting to pass a fine-tuned, otherwise auto-loads COCO weights
            use_seg (bool): use segmentation model variation
            show (bool): whether to display the detection results visually
        """
        super().__init__(**kwargs)

        if model and model.lower() in self.MODEL_OPTIONS:
            if use_seg:
                if model.lower() in self.SEG_MODEL_OPTIONS:
                    model_choice = self.SEG_MODEL_OPTIONS[model.lower()]
                    print(f"Using segmentation model of {model_choice}.")
                else:
                    print(
                        "Unknown model option for segmentation. Default to RFDETRSegSmall."
                    )
                    print(
                        "Other options include 'nano' for RFDETRSegNano, 'medium' for RFDETRSegMedium and 'large' for RFDETRSegLarge."
                    )
                    model_choice = self.SEG_MODEL_OPTIONS["small"]
            else:
                model_choice = self.MODEL_OPTIONS[model.lower()]
                print(f"Using {model_choice}.")
        else:
            print("Unknown model option. Default to RFDETRSmall.")
            print(
                "Other options include 'nano' for RFDETRNano, 'medium' for RFDETRMedium and 'large' for RFDETRLarge."
            )
            model_choice = (
                self.MODEL_OPTIONS["small"]
                if not use_seg
                else self.SEG_MODEL_OPTIONS["small"]
            )

        if pretrain:
            print(f"Using custom checkpoint from {pretrain}")
            path_to_chkpnt = pretrain
        else:
            print(f"Using default COCO pretrained weights for {model}")
            path_to_chkpnt = None
        self.model = model_choice()
        self.use_seg = use_seg
        self.queue = deque(maxlen=1)
        self.show = show

    def process_update(self, update_message):
        for iu, ut in update_message:
            if ut != retico_core.UpdateType.ADD:
                continue
            else:
                self.queue.append(iu)

    def _detector_thread(self):
        while self._detector_thread_active:
            time.sleep(0.2)
            if len(self.queue) == 0:
                time.sleep(
                    0.5
                )  # original(0.5) ~ change this for more time between segmentation of each image
                continue

            input_iu = self.queue.popleft()
            image = input_iu.payload

            detections = self.model.predict(image, threshold=0.25)
            labels = [COCO_CLASSES[class_id] for class_id in detections.class_id]

            if self.show:
                annotated_frame = image.copy()
                if self.use_seg:
                    annotated_frame = sv.MaskAnnotator().annotate(
                        annotated_frame, detections
                    )
                else:
                    annotated_frame = sv.BoxAnnotator().annotate(image, detections)
                annotated_frame = sv.LabelAnnotator().annotate(
                    annotated_frame, detections, labels
                )
                frame = np.array(annotated_frame)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.imshow("RF-DETR", frame)
                cv2.waitKey(1)

            if len(detections) == 0:
                continue

            output_iu = self.create_iu(input_iu)
            if self.use_seg:
                output_iu.set_detected_objects(image, detections.mask, "seg")
            else:
                output_iu.set_detected_objects(image, detections.xyxy, "bb")
            um = retico_core.UpdateMessage.from_iu(
                output_iu, retico_core.UpdateType.ADD
            )
            self.append(um)

    def prepare_run(self):
        self._detector_thread_active = True
        threading.Thread(target=self._detector_thread).start()

    def shutdown(self):
        self._detector_thread_active = False
