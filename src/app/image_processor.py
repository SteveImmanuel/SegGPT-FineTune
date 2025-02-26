"""
Processes the frame
"""
import datetime
import cv2
import os
import queue
import threading
from pathlib import Path
from app_config import AppConfig
from ship_detection import ShipDetection
from object_detection import ObjectDetection

IMAGE_QUEUE: queue.Queue = queue.Queue()

import logging
import spacefx
logger = spacefx.logger(level=logging.INFO)

class ImageProcessor:
    """
    Process the images sent by the monitor
    """
    image_gutter: int = 10
    ui_font = cv2.FONT_HERSHEY_SIMPLEX

    def __init__(self):

        self.app_config = AppConfig()

        print("Starting Image Processor...", end=" ")

        for _ in range(0, self.app_config.NUM_OF_WORKERS):
            image_processor = threading.Thread(target=self.monitor_queue)
            image_processor.daemon = True
            image_processor.start()

        print("success")

    @staticmethod
    def add_image_to_queue(imagefile:str):
        """
        Add an image to the queue for processing
        """

        IMAGE_QUEUE.put(imagefile)

    from pathlib import Path

    def monitor_queue(self):
        """
        Monitors the image queue, processes each image, and saves the results.
        """
        # Initialize the ship detection model
        ship_detection = ObjectDetection(Path(self.app_config.INBOX_FOLDER, self.app_config.MODEL_FILENAME))

        # Calculate the maximum chip size based on the model's input shape and the chipping scale
        chip_max_height = round(ship_detection.input_shape[0] * self.app_config.IMG_CHIPPING_SCALE)
        chip_max_width = round(ship_detection.input_shape[1] * self.app_config.IMG_CHIPPING_SCALE)

        # Start monitoring the image queue
        while True:
            # Get the next image from the queue
            input_image_path = Path(IMAGE_QUEUE.get())
            logger.info(f"Processing {input_image_path}")

            # Read the image into memory
            raw_image = cv2.imread(str(input_image_path))
            img_height, img_width, img_channels = raw_image.shape

            # Log the image and chip sizes
            print(f"...image size: {img_width}x{img_height}")
            print(f"...tensor size: {ship_detection.input_shape[0]}x{ship_detection.input_shape[1]}")
            print(f"...maximum scale factor: {self.app_config.IMG_CHIPPING_SCALE} ({chip_max_height}x{chip_max_width})")

            # Save the original image
            self.save_image(raw_image, Path(self.app_config.OUTBOX_FOLDER, f"{input_image_path.stem}_orig.jpg"))

            # Run ship detection on the image or on each chip of the image
            if img_width > chip_max_width or img_height > chip_max_height:
                all_detections = self.run_ship_detection_large_image(ship_detection=ship_detection, raw_image=raw_image, chip_max_height=chip_max_height, chip_max_width=chip_max_width)
            else:
                all_detections = self.run_ship_detection(ship_detection=ship_detection, raw_image=raw_image)

            # Prepare the filename for the augmented image
            augmented_file_path = Path(self.app_config.OUTBOX_FOLDER, f"{input_image_path.stem}_augmented.jpg")
            logger.info(f"Detected {len(all_detections)} ships.  Building augmented image '{augmented_file_path}'")

            # Loop over each detection
            for i, detection in enumerate(all_detections, start=1):
                print(f"Writing ship detection #{i}")

                # Draw the detection on the image
                raw_image = self.write_hitboxes(raw_image=raw_image, detection=detection, ship_num=i)

                # Calculate the coordinates of the ship image with padding
                ship_start_x = max(0, detection.x_coordinate - self.app_config.IMG_CHIPPING_PADDING)
                ship_start_y = max(0, detection.y_coordinate - self.app_config.IMG_CHIPPING_PADDING)
                ship_end_x = min(img_width, detection.x_coordinate + detection.width + self.app_config.IMG_CHIPPING_PADDING)
                ship_end_y = min(img_height, detection.y_coordinate + detection.height + self.app_config.IMG_CHIPPING_PADDING)

                # Extract the ship image from the raw image
                cropped_ship_img = raw_image[int(ship_start_y):int(ship_end_y), int(ship_start_x):int(ship_end_x)]

                # Save the ship image
                self.save_image(cropped_ship_img, Path(self.app_config.OUTBOX_FOLDER, self.app_config.OUTBOX_FOLDER_CHIPS, f"{input_image_path.stem}_ship_{i}.jpg"))

            # Save the augmented image
            self.save_image(raw_image, augmented_file_path)

            logger.info(f"Finished processing {input_image_path}")

    def save_image(self, image, path):
        """
        Saves the given image to the specified path.
        """
        logger.info(f"Saving image to '{path}'")
        cv2.imwrite(str(path), image)

    def run_ship_detection_large_image(self, ship_detection:ObjectDetection, raw_image, chip_max_height:int, chip_max_width:int):
        """
        Runs ship detection on a large image by dividing it into smaller chips and running detection on each chip.

        Args:
            ship_detection (ObjectDetection): The ship detection model used for prediction.
            raw_image (numpy.ndarray): The raw image on which ship detection is performed.
            chip_max_height (int): The maximum height of each chip.
            chip_max_width (int): The maximum width of each chip.

        Returns:
            list: A list of ShipDetection objects representing the detected ships in the image.
        """
        # Initialize an empty list to store all detections
        all_detections = []

        # Get the shape of the raw image
        orig_img_height, orig_img_width, _ = raw_image.shape

        # Loop through the rows and columns of the image, creating chips
        for chip_y_start in range(0, orig_img_height, chip_max_height):
            for chip_x_start in range(0, orig_img_width, chip_max_width):
                # Calculate the end coordinates of the chip, ensuring they don't exceed the image dimensions
                chip_y_end = min(chip_y_start + chip_max_height, orig_img_height)
                chip_x_end = min(chip_x_start + chip_max_width, orig_img_width)

                # Extract the chip from the raw image
                raw_image_chip = raw_image[chip_y_start:chip_y_end, chip_x_start:chip_x_end]

                # Run ship detection on the chip
                chipped_detections = self.run_ship_detection(ship_detection=ship_detection, raw_image=raw_image_chip)

                # Adjust the coordinates of the detections based on the chip's position in the image
                for chipped_detection in chipped_detections:
                    chipped_detection.x_coordinate += chip_x_start
                    chipped_detection.y_coordinate += chip_y_start

                    # Log the detection
                    logger.info(f"Ship detected at ({chipped_detection.x_coordinate}, {chipped_detection.y_coordinate}).  Width: {chipped_detection.width}  Height: {chipped_detection.height}")

                    # Add the detection to the list of all detections
                    all_detections.append(chipped_detection)

        # Return the list of all detections
        return all_detections

    def run_ship_detection(self, ship_detection:ObjectDetection, raw_image):
        """
        Runs ship detection on the given raw image using the provided ship detection model.

        Args:
            ship_detection (ObjectDetection): The ship detection model used for prediction.
            raw_image (numpy.ndarray): The raw image on which ship detection is performed.

        Returns:
            list: A list of ShipDetection objects representing the detected ships in the image.
        """
        # Get the shape of the raw image
        orig_img_height, orig_img_width, _ = raw_image.shape

        # Run the ship detection model on the raw image
        ship_predictions = ship_detection.predict_image(raw_image)

        # Parse the predictions using the detection labels
        ship_predictions = self.parse_predictions(self.app_config.DETECTION_LABELS, ship_predictions)

        # Filter out predictions below the detection threshold and create ShipDetection objects for the rest
        all_detections = [
            ShipDetection(
                probability=ship_hitbox['probability'],
                x_coordinate=round(orig_img_width * ship_hitbox['boundingBox']['left']),
                y_coordinate=round(orig_img_height * ship_hitbox['boundingBox']['top']),
                width=round(orig_img_width * ship_hitbox['boundingBox']['width']),
                height=round(orig_img_height * ship_hitbox['boundingBox']['height'])
            )
            for ship_hitbox in ship_predictions
            if ship_hitbox['probability'] >= self.app_config.DETECTION_THRESHOLD
        ]

        # Return the list of all detections
        return all_detections


    def write_hitboxes(self, raw_image, detection:ShipDetection, ship_num:int):
        """
        Writes hitboxes and prediction text on the input image.

        Args:
            raw_image (numpy.ndarray): The input image.
            detection (ShipDetection): The detected ship object.
            ship_num (int): The ship number.

        Returns:
            numpy.ndarray: The image with hitboxes and prediction text.
        """
        # Set blending factor, box color, and thickness
        alpha = 0.2  # Transparency factor for blending the images
        color = (0, 0, 255)  # Color for the hitbox and text background (Blue, Green, Red)
        hitbox_thickness = 2  # Thickness of the hitbox outline

        # Make a copy of the original image to use for blending
        orig_img = raw_image.copy()

        # Get image dimensions
        img_height, img_width, img_channels = raw_image.shape

        # Calculate the starting and ending points of the hitbox
        hitbox_start_point = (detection.x_coordinate, detection.y_coordinate)
        hitbox_end_point = (detection.x_coordinate + detection.width, detection.y_coordinate + detection.height)

        # Draw the hitbox on the image
        ship_highlight = cv2.rectangle(raw_image, hitbox_start_point, hitbox_end_point, color, hitbox_thickness)

        # Blend the original image and the image with the hitbox
        raw_image = cv2.addWeighted(orig_img, alpha, ship_highlight, 1 - alpha, 0)

        # Initialize hitbox header position for text display
        hitbox_header_start_x = detection.x_coordinate
        hitbox_header_start_y = detection.y_coordinate + detection.height
        hitbox_header_width = detection.width + 1  # Just slightly wider than the ship width
        hitbox_header_height = 20  # Fixed height for the text background

        # If the text would go beyond the bottom of the image, move it above the hitbox
        if hitbox_header_start_y + hitbox_header_height > img_height:
            hitbox_header_start_y = detection.y_coordinate - hitbox_header_height

        # Define the starting point for the header
        hitbox_header_start_point = (hitbox_header_start_x, hitbox_header_start_y)

        # Calculate the starting point for the prediction text
        prediction_text_start_point = (hitbox_header_start_x + round(hitbox_header_width * .05),
                                       hitbox_header_start_y + hitbox_header_height - round(hitbox_header_height * .15))

        # Set the scale and thickness for the prediction text
        prediction_text_scale = 0.40
        prediction_text_thickness = 1
        prediction_text_color = (255, 255, 255)  # White color for the text

        # Format the text to display the ship number and detection probability
        text_to_place = f"Ship #{ship_num}: {detection.probability:.2%}"

        # Calculate the size of the text to be placed
        text_size, _ = cv2.getTextSize(text_to_place, self.ui_font, prediction_text_scale, prediction_text_thickness)
        text_w, text_h = text_size

        # Add padding to the text width if needed
        text_w += round(text_w * .05)

        # Ensure the text width is not less than the ship width
        text_w = max(text_w, detection.width)

        # Draw a filled rectangle as the background for the text
        cv2.rectangle(raw_image, hitbox_header_start_point, (hitbox_header_start_x + text_w, hitbox_header_start_y + text_h * 3), color, -1)

        # Place the prediction text on top of the background rectangle
        raw_image = cv2.putText(raw_image, text_to_place, prediction_text_start_point, self.ui_font,
                                prediction_text_scale, prediction_text_color, prediction_text_thickness, cv2.LINE_AA)

        # Return the image with the drawn hitbox and text
        return raw_image



    def parse_predictions(self, labels, predictions):
        """
        Combines and flattens the predictions array from tensorflow
        """

        assert set(predictions.keys()) == set(['detected_boxes', 'detected_classes', 'detected_scores'])
        formatted_predictions = []
        for box, class_id, score in zip(predictions['detected_boxes'][0], predictions['detected_classes'][0], predictions['detected_scores'][0]):
            formatted_prediction = {
            'probability': round(float(score), 8),
            'tagId': int(class_id),
            'tagName': labels[class_id],
            'boundingBox': {
                'left': round(float(box[0]), 8),
                'top': round(float(box[1]), 8),
                'width': round(float(box[2] - box[0]), 8),
                'height': round(float(box[3] - box[1]), 8)
                }
            }

            formatted_predictions.append(formatted_prediction)

        return formatted_predictions
