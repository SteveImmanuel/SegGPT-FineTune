import cv2

import numpy as np
import onnx
import onnxruntime
import PIL.Image

class ObjectDetection:
    """
    Runs Inference on a visual image using ONNX
    """

    def __init__(self, model_filename):
        self.session = onnxruntime.InferenceSession(str(model_filename))
        assert len(self.session.get_inputs()) == 1
        self.input_shape = self.session.get_inputs()[0].shape[2:]
        self.input_name = self.session.get_inputs()[0].name
        self.input_type = {'tensor(float)': np.float32, 'tensor(float16)': np.float16}[self.session.get_inputs()[0].type]
        self.output_names = [o.name for o in self.session.get_outputs()]

        self.is_bgr = False
        self.is_range255 = False
        onnx_model = onnx.load(model_filename)
        for metadata in onnx_model.metadata_props:
            if metadata.key == 'Image.BitmapPixelFormat' and metadata.value == 'Bgr8':
                self.is_bgr = True
            elif metadata.key == 'Image.NominalPixelRange' and metadata.value == 'NominalRange_0_255':
                self.is_range255 = True


    def predict_image(self, opencvimg):
        """
        Run interference on the image
        """

        image = PIL.Image.fromarray(cv2.cvtColor(opencvimg, cv2.COLOR_BGR2RGB))

        # Resize the image to match the input shape
        image = image.resize(self.input_shape)

        # Convert the image back to a numpy array and normalize pixel values if necessary
        input_array = np.array(image, dtype=np.float32)[np.newaxis, :, :, :]
        input_array = input_array.transpose((0, 3, 1, 2))  # => (N, C, H, W)
        if self.is_bgr:
            input_array = input_array[:, (2, 1, 0), :, :]
        if not self.is_range255:
            input_array = input_array / 255  # => Pixel values should be in range [0, 1]


        outputs = self.session.run(self.output_names, {self.input_name: input_array.astype(self.input_type)})
        return {name: outputs[i] for i, name in enumerate(self.output_names)}