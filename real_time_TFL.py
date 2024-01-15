import cv2 as cv
import numpy as np
from tensorflow.lite.python.interpreter import Interpreter
# tensorflow 2.8.0 ale dziala
# protobuf package 3.20.x or lower.


################################################################################################
#
# Mariusz Wisniewski KD-712
#
# Podstawy rozpoznawania obrazów
# 4606-PS-00000CJ-C001
# Agnieszka Jastrzębska
#
# Detekcja twarzy i obecnosci maski na twarzy
#
# Detekcja w czasie rzeczywistym z kamery
# Wykrywanie twarzy i maski przeprowadzane przez jeden model TFLite
# doszkolony na podstawie 853 zdjec z zaznaczonymi glowami w maskach lub bez
#
# 2024-01-10
#
################################################################################################


def tflite_detect_camera(modelpath, lblpath, min_conf=0.5):

    # Load the label map into memory
    with open(lblpath, 'r') as f:
        labels = [line.strip() for line in f.readlines()]

    # Load the Tensorflow Lite model into memory
    interpreter = Interpreter(model_path=modelpath)
    interpreter.allocate_tensors()

    # Get model details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]
    print(width, height)

    float_input = (input_details[0]['dtype'] == np.float32)

    input_mean = 127.5
    input_std = 127.5

    vid = cv.VideoCapture(0)

    while True:
        ret, image = vid.read()

        # Resize to expected shape [1xHxWx3]
        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        imH, imW, _ = image.shape
        image_resized = cv.resize(image_rgb, (width, height))
        input_data = np.expand_dims(image_resized, axis=0)

        # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
        if float_input:
            input_data = (np.float32(input_data) - input_mean) / input_std

        # Perform the actual detection by running the model with the image as input
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        # Retrieve detection results
        boxes = interpreter.get_tensor(output_details[1]['index'])[0]  # Bounding box coordinates of detected objects
        classes = interpreter.get_tensor(output_details[3]['index'])[0]  # Class index of detected objects
        scores = interpreter.get_tensor(output_details[0]['index'])[0]  # Confidence of detected objects

        # Loop over all detections and draw detection box if confidence is above minimum threshold
        for i in range(len(scores)):
            if ((scores[i] > min_conf) and (scores[i] <= 1.0)):
                # Get bounding box coordinates and draw box
                # Interpreter can return coordinates that are outside of image dimensions,
                # need to force them to be within image using max() and min()
                ymin = int(max(1, (boxes[i][0] * imH)))
                xmin = int(max(1, (boxes[i][1] * imW)))
                ymax = int(min(imH, (boxes[i][2] * imH)))
                xmax = int(min(imW, (boxes[i][3] * imW)))

                # Draw label
                object_name = labels[int(classes[i])]  # Look up object name from "labels" array using class index

                label = '%s: %d%%' % (object_name, int(scores[i] * 100))  # Example: 'person: 72%'
                if object_name == "without_mask" or object_name == "mask_weared_incorrect":
                    cv.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
                    cv.putText(image, label, (xmin, ymin - 8), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)  # Draw label text
                else:
                    cv.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 128, 0), 2)
                    cv.putText(image, label, (xmin, ymin - 8), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,128,0), 2)  # Draw label text

        cv.imshow('frame', image)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    cv.destroyAllWindows()

    return


if __name__ == '__main__':

    # Set up variables for running user's model
    # Path to .tflite model file
    PATH_TO_MODEL = './custom_model_lite_mask/detect.tflite'
    # Path to labelmap.txt file
    PATH_TO_LABELS = './custom_model_lite_mask/labelmap.txt'
    # Confidence threshold (try changing this to 0.01 if you don't see any detection results)
    min_conf_threshold = 0.3

    # Run inferencing function!
    tflite_detect_camera(PATH_TO_MODEL, PATH_TO_LABELS, min_conf_threshold)

