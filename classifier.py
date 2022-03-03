#!/usr/bin/env python

"""
classifier.py: This script will take either a video file or capture footage from the first available camera on the
system and pass it through an InceptionV3 model for inference. Finally the frame will be shown along with the results
from the inference on the top left. Press Q to quit the program.
"""
import json
from argparse import ArgumentParser

import cv2
import numpy as np
from noussdk.communication.mappers.mongodb_mapper import LabelToMongo
from openvino.inference_engine import IECore

from utils.cosmonio_utils import draw_label_on_image, resize_and_pad_to_square, array_to_img, img_to_array

__author__ = "COSMONiO Development Team"
__copyright__ = "COSMONiO © All rights reserved"
__credits__ = ["COSMONiO Development Team"]
__version__ = "1.0"
__maintainer__ = "COSMONiO Development Team"
__email__ = "support@cosmonio.com"
__status__ = "Production"
__updated__ = "09.11.2020"


class Streamer:
    def __init__(self, file_path: str):
        self.frame = None

        if file_path != '':
            file_path = file_path.replace("\\", "/")
            # capture from video file
            self.capture = cv2.VideoCapture(file_path)
        else:
            # capture from first available camera on the system
            self.capture = cv2.VideoCapture(0)

    def __del__(self):
        self.capture.release()

    def check_frame_available(self) -> bool:
        """
        Checks if a frame is available from the capture.
        :return: bool
        """
        frame_available, self.frame = self.capture.read()
        return frame_available

    def get_frame(self) -> np.array:
        """
        :return: latest available frame
        """
        return self.frame


def classifier(streamer: Streamer):
    """
    Starts a loop that does inference on available frames from passed streamer. If there are no frames available
    the program will stop.
    """
    # Create a labels from labels.json
    with open("model/labels.json") as label_file:
        label_data = json.load(label_file)
    labels = [LabelToMongo().backward(label) for label in label_data]

    with open("model/configurable_parameters.json") as config:
        data = json.load(config)
        model_architecture = data.get("learning_architecture", {}).get("model_architecture", {}).get("value",
                                                                                                     "inception_v3")

    if model_architecture == "inception_v3":
        image_size = 299
    else:
        image_size = 224

    ie = IECore()
    net = ie.read_network('model/inference_model.xml', 'model/inference_model.bin')
    input_blob = next(iter(net.input_info))
    output_blob = next(iter(net.outputs))
    # Specify target device
    exec_net = ie.load_network(network=net, device_name="CPU")
    del net

    # While a next frame is available do inference
    while streamer.check_frame_available():
        frame = streamer.get_frame()
        pil_img = array_to_img(frame)

        # prepare frame for input
        resized_frame = resize_and_pad_to_square(frame, image_size)
        resized_frame = resized_frame[:, :, [2, 1, 0]]
        x = img_to_array(resized_frame)
        x = np.moveaxis(x, -1, 0)
        preprocessed_numpy = [((x / 255.0) - 0.5) * 2]
        batch_tensor = np.stack(preprocessed_numpy, axis=0)

        # inference frame
        result = exec_net.infer(inputs={input_blob: batch_tensor})[output_blob]

        # Get the label
        if len(labels) > 1:
            # more than one labels
            label_index = np.argmax(result)
            label = labels[label_index]
        elif len(labels) == 1 and result[0] >= 0.5:
            # single-label setup
            label = labels[0]
        else:
            label = None

        # Draw label on frame
        if label is not None:
            draw_label_on_image(pil_img, label)

        # Show results
        display_image = np.array(pil_img)
        cv2.imshow("Classifier", display_image)
        if ord("q") == cv2.waitKey(1):
            break


def main():
    arguments_parser = ArgumentParser()
    arguments_parser.add_argument("--file", required=False,
                                  help="Specify a videofile location, if nothing is specified it will grab the webcam",
                                  default='')
    arguments = arguments_parser.parse_args()
    file = str(arguments.file)
    classifier(Streamer(file))


if __name__ == '__main__':
    main()
