import os
import glob
import json
import time
from typing import List
from dataclasses import dataclass, field
from utils import check_file_exists

@dataclass
class AppConfig:
    LATITUDE: float # Latitude of the location to query for imagery
    LONGITUDE: float #Longitude of the location to query for imagery
    MODEL_FILENAME: str # Filename of the model to run inference on any imagery returned
    MODEL_LABEL_FILENAME: str # Filename of the model's labels to run inference on any imagery returned
    INBOX_FOLDER: str # Inbox folder that'll contain the model, labels, and imagery to process
    DETECTION_THRESHOLD: float # Minimum confidence threshold to consider an object detected
    OUTBOX_FOLDER_CHIPS: str # Output folder to store the chips of any objects detected from the imagery
    OUTBOX_FOLDER: str # Output folder to store a copy of the source imagery with the detected objects annotated
    IMG_CHIPPING_SCALE: int # Maximum size of the chip to extract from the source imagery compared to the source tensor size
    NUM_OF_WORKERS: int # Number of workers to use for the image processing
    IMG_CHIPPING_PADDING: float # Amount of overlap pixels between chips extracted from the source imagery
    DETECTION_LABELS: List[str] = field(default_factory=list) # Detection labels
    TYPE_MAPPING = {
        'IMG_CHIPPING_PADDING': float,
        'LATITUDE': float,
        'LONGITUDE': float,
        'DETECTION_THRESHOLD': float,
        'IMG_CHIPPING_SCALE': int,
        'NUM_OF_WORKERS': int,
    }

    @classmethod
    def from_json(cls, json_path: str='/var/spacedev/xfer/app-python-shipdetector-onnx/inbox/app-config.json'):
        check_file_exists(json_path)
        with open(json_path, 'r') as f:
            data = json.load(f)

        os.makedirs(os.path.join(cls.OUTBOX_FOLDER, cls.OUTBOX_FOLDER_CHIPS), exist_ok=True)
        os.makedirs(cls.OUTBOX_FOLDER, exist_ok=True)
        return cls(**{k: cls.__annotations__.get(k, str)(v) for k, v in data.items()})



    # def __init__(self, json_path: str='/var/spacedev/xfer/app-python-shipdetector-onnx/inbox/app-config.json'):
    #     for key, value in data.items():
    #         expected_type = self.TYPE_MAPPING.get(key)
    #         if expected_type:
    #             setattr(self, key, expected_type(value))
    #         else:
    #             setattr(self, key, value)

    #     check_file_exists(os.path.join(self.INBOX_FOLDER, self.MODEL_LABEL_FILENAME))
    #     check_file_exists(os.path.join(self.INBOX_FOLDER, self.MODEL_FILENAME))

    #     with open(os.path.join(self.INBOX_FOLDER, self.MODEL_LABEL_FILENAME), encoding='UTF-8') as f:
    #         self.DETECTION_LABELS = [l.strip() for l in f.readlines()]
