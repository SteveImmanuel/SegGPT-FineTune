import cv2
import onnx
import onnxruntime
import numpy as np
import logging

class SegmentationModel:
    def __init__(self, model_path: str):
        self.logger = logging.getLogger(__name__)
        self.logger.info(f'Loading model from {model_path}')
        self.session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])

        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = tuple(self.session.get_inputs()[0].shape[2:])
        self.logger.info(f"Expected input '{self.input_name}' with shape {self.session.get_inputs()[0].shape}")


    def inference(self, img: np.ndarray):
        if img.shape[0] != self.input_shape[0] or img.shape[1] != self.input_shape[1]:
            self.logger.warning(f'Input image shape {img.shape} does not match expected shape {self.input_shape}. Resizing.')

        resized_img = cv2.resize(img, self.input_shape)
        resized_img = np.transpose(resized_img, (2, 0, 1))
        resized_img = resized_img[None, ...].astype(np.float32)

        out: np.ndarray = self.session.run(0, {self.input_name: resized_img})
        return out