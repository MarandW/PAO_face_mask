import cv2
import numpy as np
from tensorflow.lite.python.interpreter import Interpreter
# tensorflow 2.8.0
# Downgrade the protobuf package to 3.20.x or lower.

import modyfikacje as m
import os
import random


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




    image_bg = m.wczytaj_tlo()

    cv2.imshow('Image', image_bg)
    cv2.waitKey(0)

#    IMG_SIZE = (224, 224)

    path = os.getcwd()

    path = os.path.join(path, 'dataset_big_test')
    path_test = os.path.join(path, 'test_50')

    fnames, cnames, class_names = m.lista_obrazkow_testowych(path_test)

    print(class_names)

    # jesli ma byc powtarzalne
    np.random.seed(4)

    output_name = "test_results_TFL.txt"

    f = open(output_name, "w") # zerowanie pliku
    f.close()


#    vid = cv2.VideoCapture(0)

#    while True:
#        ret, image = vid.read()

    for fi in range(len(fnames)):
        # Retrieve a batch of images from the test set
        image_path = os.path.join(path_test, cnames[fi], fnames[fi])
        print(image_path, cnames[fi], fnames[fi])
        image_f = cv2.imread(image_path)  # surowy obrazek

        #    cv2.imshow('Image', image_f)
        #    cv2.waitKey(0)

        x_offset = random.randint(50, 500)
        y_offset = random.randint(50, 300)

        dklasa = -1
        if cnames[fi] == 'with_mask':
            dklasa = 1
        if cnames[fi] == 'without_mask':
            dklasa = 0

        for size in range(-275, 0, 25):
            for bright in range(-150, 150, 50):
                for noise in range(0, 150, 25):
                    # czysty obrazek tla do modyfkacji
                    image_bgt = image_bg.copy()

                    # zmiana rozmiaru obrazka (do standardowego)
                    image_s = m.skaluj_obrazek(image_f, 300, size)

                    #                cv2.imshow('Image size', image_s)
                    #                cv2.waitKey(0)

                    # zmiana jasnosci obrazka
                    image_br = m.skaluj_jasnosc(image_s, bright)

                    #                cv2.imshow('Image br', image_br)
                    #                cv2.waitKey(0)

                    # wklejenie na tlo

                    #                print(x_offset,int(image_br.shape[1]/2),y_offset,int(image_br.shape[0]/2))
                    m.wklej_obrazek(image_bgt, image_br, x_offset - int(image_br.shape[1] / 2),
                                    y_offset - int(image_br.shape[0] / 2))

                    # dodawanie szumu do calosci
                    image = m.dodaj_szum(image_bgt, noise)

                    #iterater trough faces
                    imy = image.copy()




                    # Resize to expected shape [1xHxWx3]
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    imH, imW, _ = image.shape
                    image_resized = cv2.resize(image_rgb, (width, height))
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
                    klasa = -1
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
                                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
                                cv2.putText(image, label, (xmin, ymin - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)  # Draw label text
                                klasa = 0
                            else:
                                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 128, 0), 2)
                                cv2.putText(image, label, (xmin, ymin - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,128,0), 2)  # Draw label text
                                klasa = 1

                    if klasa >= 0 :
                        # print(fnames[i], dklasa, klasa, result[0][0], size, bright, noise)
                        out_line = f'{fnames[fi]} {fi} {dklasa} {klasa} {scores[0] * 100:.1f} {size} {bright} {noise} \n'
                        print(out_line)
                    else:
                        # print(fnames[i], dklasa, klasa, -1.0, size, bright, noise)
                        out_line = f'{fnames[fi]} {fi} {dklasa} {klasa} -100.0 {size} {bright} {noise} \n'
                        print(out_line)
                    f = open(output_name, "a")
                    f.write(out_line)
                    f.close()

                    cv2.imshow('frame', image)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

#    vid.release()
    cv2.destroyAllWindows()

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

