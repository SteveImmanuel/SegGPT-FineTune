"""
Strong type the app config
"""
import os
import glob
from dataclasses import dataclass, field
import json
from typing import List
import time

@dataclass
class AppConfig:
    """
    Strong type the apps parameters supplied by arguments
    """

    LATITUDE: float
    """
    Latitude of the location to query for imagery
    """

    LONGITUDE: float
    """
    Longitude of the location to query for imagery
    """

    MODEL_FILENAME: str
    """
    Filename of the model to run inference on any imagery returned
    """

    MODEL_LABEL_FILENAME: str
    """
    Filename of the model's labels to run inference on any imagery returned
    """

    INBOX_FOLDER: str
    """
    Inbox folder that'll contain the model, labels, and imagery to process
    """

    DETECTION_THRESHOLD: float
    """
    Minimum confidence threshold to consider an object detected
    """

    OUTBOX_FOLDER_CHIPS: str
    """
    Output folder to store the chips of any objects detected from the imagery
    """

    OUTBOX_FOLDER: str
    """
    Output folder to store a copy of the source imagery with the detected objects annotated
    """

    IMG_CHIPPING_SCALE: int
    """
    Maximum size of the chip to extract from the source imagery compared to the source tensor size
    """

    NUM_OF_WORKERS: int
    """
    Number of workers to use for the image processing
    """

    IMG_CHIPPING_PADDING: float
    """
    Amount of overlap pixels between chips extracted from the source imagery
    """

    DETECTION_LABELS: List[str] = field(default_factory=list)
    """
    Detection labels
    """

    TYPE_MAPPING = {
        'IMG_CHIPPING_PADDING': float,
        'LATITUDE': float,
        'LONGITUDE': float,
        'DETECTION_THRESHOLD': float,
        'IMG_CHIPPING_SCALE': int,
        'NUM_OF_WORKERS': int,
    }

    def __init__(self, file_path='/var/spacedev/xfer/app-python-shipdetector-onnx/inbox/app-config.json'):
        """
        Initializes the AppConfig object by loading the configuration from a JSON file.

        Args:
            file_path (str, optional): Path to the configuration JSON file. Defaults to '/var/spacedev/xfer/app-python-shipdetector-onnx/inbox/app-config.json'.
        """

        def check_file_exists(path):
            """
            Checks if a file exists at the given path.

            Args:
                path (str): Path to the file.

            Raises:
                FileNotFoundError: If the file does not exist.
            """
            # Wait for the file to appear if it's not available immediately
            file_available_timeout = time.time() + 120  # seconds
            while (not os.path.isfile(path)) and (time.time() < file_available_timeout):
                time.sleep(1)

            if not os.path.isfile(path):
                raise FileNotFoundError(f"The file {path} does not exist.")

        def ensure_dir_exists(path):
            """
            Ensures that a directory exists at the given path. If the directory does not exist, it will be created.

            Args:
                path (str): Path to the directory.
            """
            if not os.path.isdir(path):
                os.makedirs(path, exist_ok=True)

        check_file_exists(file_path)
        with open(file_path, 'r') as f:
            data = json.load(f)
        for key, value in data.items():
            expected_type = self.TYPE_MAPPING.get(key)
            if expected_type:
                setattr(self, key, expected_type(value))
            else:
                setattr(self, key, value)

        check_file_exists(os.path.join(self.INBOX_FOLDER, self.MODEL_LABEL_FILENAME))
        check_file_exists(os.path.join(self.INBOX_FOLDER, self.MODEL_FILENAME))

        with open(os.path.join(self.INBOX_FOLDER, self.MODEL_LABEL_FILENAME), encoding='UTF-8') as f:
            self.DETECTION_LABELS = [l.strip() for l in f.readlines()]

        ensure_dir_exists(os.path.join(self.OUTBOX_FOLDER, self.OUTBOX_FOLDER_CHIPS))
        ensure_dir_exists(self.OUTBOX_FOLDER)