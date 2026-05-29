"""
Roboflow Detection Transformer module
=====================================

This module implements the Roboflow Detection Transformer (RFDetr) architecture,
which is designed for object detection tasks. It uses HuggingFace pipelines.
"""

# Imports
import threading
import time
from collections import deque
import numpy as np
import torch
import retico_core
from retico_vision.vision import ImageIU, DetectedObjectsIU
import cv2
import supervision as sv
from rfdetr.assets.coco_classes import COCO_CLASSES
from transformers import (
    AutoImageProcessor,
    RfDetrForObjectDetection,
    RfDetrForInstanceSegmentation,
)


class HFRFDETRModule(retico_core.AbstractModule):

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
        "medium": "Roboflow/rf-detr-medium",
        "large": "Roboflow/rf-detr-large",
    }

    SEG_MODEL_OPTIONS = {
        "medium": "stevenbucaille/rf-detr-seg-medium",
        "large": "stevenbucaille/rf-detr-seg-large",
    }

    def __init__(
        self, model="medium", show=False, use_seg=False, threshold=0.25, **kwargs
    ):
        """
        Initialize the RFDETR Module
        Args:
            model (str): the name of the RFDETR model will correspond to the model checkpoint
                options: medium, large
            use_seg (bool): use segmentation model variant
            show (bool): whether to display the detection results visually
            threshold (float): the detection threshold
        """
        super().__init__(**kwargs)

        if model and model.lower() in self.MODEL_OPTIONS:
            if use_seg:
                if model.lower() in self.SEG_MODEL_OPTIONS:
                    model_choice = self.SEG_MODEL_OPTIONS[model.lower()]
                    print(f"Using segmentation model variant of {model_choice}.")
                else:
                    print("Unknown model option for segmentation. Default to medium.")
                    print("Other options include 'large'.")
                    model_choice = self.SEG_MODEL_OPTIONS["medium"]
            else:
                model_choice = self.MODEL_OPTIONS[model.lower()]
                print(f"Using {model_choice}.")
        else:
            print("Unknown model option. Default to RFDETRSmall.")
            print("Other options include 'large'.")
            model_choice = (
                self.MODEL_OPTIONS["medium"]
                if not use_seg
                else self.SEG_MODEL_OPTIONS["medium"]
            )

        self.processor = AutoImageProcessor.from_pretrained(model_choice)
        if use_seg:
            self.model = RfDetrForInstanceSegmentation.from_pretrained(model_choice)
        else:
            self.model = RfDetrForObjectDetection.from_pretrained(model_choice)
        self.use_seg = use_seg
        self.queue = deque(maxlen=1)
        self.show = show
        self.threshold = threshold

    def process_update(self, update_message):
        for iu, ut in update_message:
            if ut != retico_core.UpdateType.ADD:
                continue
            else:
                self.queue.append(iu)

    def _detector_thread(self):
        while self._detector_thread_active:
            time.sleep(1)
            if len(self.queue) == 0:
                time.sleep(
                    0.5
                )  # original(0.5) ~ change this for more time between segmentation of each image
                continue

            input_iu = self.queue.popleft()
            image = input_iu.payload

            inputs = self.processor(images=image, return_tensors="pt")
            outputs = self.model(**inputs)
            target_sizes = torch.tensor([image.size[::-1]])

            if self.use_seg:
                results = self.processor.post_process_instance_segmentation(
                    outputs,
                    target_sizes=[(image.size[1], image.size[0])],
                    threshold=self.threshold,
                )[0]
                if len(results["segments_info"]) == 0:
                    continue
                output_iu = self.create_iu(input_iu)
                output_iu.set_detected_objects(image, results["segmentation"], "seg")
            else:
                results = self.processor.post_process_object_detection(
                    outputs, target_sizes=target_sizes, threshold=0.35
                )[0]
                if len(results["boxes"]) == 0:
                    continue
                output_iu = self.create_iu(input_iu)
                output_iu.set_detected_objects(image, results["boxes"], "bb")

            if self.show:

                detections = sv.Detections.from_transformers(results)
                labels = [
                    self.model.config.id2label[class_id]
                    for class_id in detections.class_id
                ]
                frame = np.array(image)

                if self.use_seg:
                    frame = sv.MaskAnnotator().annotate(frame, detections)
                else:
                    frame = sv.BoxAnnotator().annotate(frame, detections)
                frame = sv.LabelAnnotator().annotate(frame, detections, labels)

                cv2.imshow("RF-DETR", frame)
                cv2.waitKey(1)

            um = retico_core.UpdateMessage.from_iu(
                output_iu, retico_core.UpdateType.ADD
            )
            self.append(um)

    def prepare_run(self):
        self._detector_thread_active = True
        threading.Thread(target=self._detector_thread).start()

    def shutdown(self):
        self._detector_thread_active = False
