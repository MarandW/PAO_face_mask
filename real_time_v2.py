import cv2
import keras
import numpy as np
from cvzone.FaceDetectionModule import FaceDetector


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
# Detekcja w czasie rzeczywistym z kamery - wersja poprawiona
# Wykrywanie twarzy oparte o FaceDetector z biblioteki cvzone
# Rozpoznanie maski oparte o model InceptionV3 (lub inny) z dodatkowymi warstwami
# doszkolony na podstawie 5551 zdjec podzielonych na dwie kategorie
#
# 2024-01-10
#
################################################################################################


## miejsce by wybrac model
#imagenet = keras.models.load_model("IncepctionV3_0_big_imagenet.h5")
##imagenet = keras.models.load_model("IncepctionV3_mod_big_imagenet.h5")
imagenet = keras.models.load_model("modele/MobileNetV2_nowymod1_01_imagenet.h5")
#imagenet = keras.models.load_model("MobileNetV2_mod_imagenet.h5")

detector = FaceDetector()

vid = cv2.VideoCapture(0)

# jesli ktos chce nagrac
#video_writer = cv2.VideoWriter("output1.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 8, (640, 480))

while True:

    # Capture the video frame
    # by frame
    ret, im = vid.read()

    #print(im)
    # Display the resulting frame
    if not ret:
        break
    else:
        #iterater trough faces
        imy = im.copy()
        img, bboxs = detector.findFaces(im)
#        print(bboxs)
        for bbb in bboxs:
            (x, y, w, h) = bbb['bbox']
            # zabezpieczenie na szybko przed glupimi wartosciami
            if x<0 :
                x=0
            if y<0 :
                y=0
            if w<=0 :
                w=1
            if h<=0 :
                h=1

            # tu brak zabezpieczenia przed wyjsciem poza obrazek
            face_img = imy[int(y):int(y) + int(h), int(x):int(x) + int(w)]
            rerect_sized = cv2.resize(face_img, (224, 224))
            normalized = rerect_sized / 255.0
            reshaped = np.reshape(normalized, (1, 224, 224, 3))
            reshaped = np.vstack([reshaped])
            result = imagenet.predict(reshaped)
#            print(result)

            if result[0][0] > 0.5:
                cv2.putText(imy, "No mask!", (x, y-8),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
                cv2.rectangle(imy, (x, y), (x + w, y + h), (0, 0, 255), 2)
            if result[0][0] <= 0.5:
                cv2.putText(imy, "Mask" + str(), (x, y-8),cv2. FONT_HERSHEY_SIMPLEX,0.5,(0,128,0),2)
                cv2.rectangle(imy, (x, y), (x + w, y + h), (0, 128, 0), 2)

#    video_writer.write(im)
    cv2.imshow('frame', imy)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
#video_writer.release()

# Destroy all the windows
cv2.destroyAllWindows()