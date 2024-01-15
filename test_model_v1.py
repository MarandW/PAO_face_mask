import cv2
import keras
import numpy as np
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
# Detekcja w czasie rzeczywistym z kamery - wersja pierwotna
# Wykrywanie twarzy oparte o haarcascades
# Rozpoznanie maski oparte o model InceptionV3 z dodatkowymi warstwami
# doszkolony na podstawie 3276 zdjec podzielonych na dwie kategorie
#
# 2024-01-10
#
################################################################################################


imagenet = keras.models.load_model("modele/InceptionV3_0_imagenet.h5")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

filname_test_results = "test_results.txt"

image_bg = m.wczytaj_tlo()

cv2.imshow('Image', image_bg)
cv2.waitKey(0)

IMG_SIZE = (224, 224)

path = os.getcwd()

path = os.path.join(path, 'dataset_big_test')
path_test = os.path.join(path, 'test_50')

fnames, cnames, class_names = m.lista_obrazkow_testowych(path_test)

print(class_names)

# jesli ma byc powtarzalne
np.random.seed(4)


#vid = cv2.VideoCapture(0)

#video_writer = cv2.VideoWriter("output1.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 8, (640, 480))

f = open(filname_test_results, "w")
f.close()

for i in range(len(fnames)):
# Retrieve a batch of images from the test set
    image_path = os.path.join(path_test, cnames[i], fnames[i])
    print(image_path, cnames[i], fnames[i])
    image_f = cv2.imread(image_path)  # surowy obrazek

#    cv2.imshow('Image', image_f)
#    cv2.waitKey(0)

    x_offset = random.randint(50, 500)
    y_offset = random.randint(50, 300)

    dklasa = -1
    if cnames[i] == 'with_mask' :
        dklasa = 1
    if cnames[i] == 'without_mask':
        dklasa = 0


    for size in range(-275,0,25):
        for bright in range(-150,150,50):
            for noise in range(0,150,25):
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
                m.wklej_obrazek(image_bgt,image_br,x_offset-int(image_br.shape[1]/2),y_offset-int(image_br.shape[0]/2))

                # dodawanie szumu do calosci
                im = m.dodaj_szum(image_bgt,noise)

                # Capture the video frame
                # by frame
                # ret, im = vid.read()

                # print(im)
                # Display the resulting frame
                gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

                #if not ret:
                #    break
                #else:

                #iterater trough faces
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                klasa = -1
                for (x, y, w, h) in faces:
                    face_img = im[y:y + h, x:x + w]
                    rerect_sized = cv2.resize(face_img, (224, 224))
                    normalized = rerect_sized / 255.0
                    reshaped = np.reshape(normalized, (1, 224, 224, 3))
                    reshaped = np.vstack([reshaped])
                    result = imagenet.predict(reshaped)


                    if result[0][0] > 0.5:
                        cv2.putText(im, "No mask!", (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
                        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 2)
                        klasa = 0
                    if result[0][0] <= 0.5:
                        cv2.putText(im, "Mask" + str(), (x, y-10),cv2. FONT_HERSHEY_SIMPLEX,0.5,(0,128,0),2)
                        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 128, 0), 2)
                        klasa = 1

                if len(faces) >0 :
                    #print(fnames[i], dklasa, klasa, result[0][0], size, bright, noise)
                    out_line = f'{fnames[i]} {i} {dklasa} {klasa} {result[0][0]*100:.1f} {size} {bright} {noise} \n'
                    print(out_line)
                else :
                    #print(fnames[i], dklasa, klasa, -1.0, size, bright, noise)
                    out_line = f'{fnames[i]} {i} {dklasa} {klasa} -100.0 {size} {bright} {noise} \n'
                    print(out_line)
                f = open(filname_test_results, "a")
                f.write(out_line)
                f.close()
                #    video_writer.write(im)
                cv2.imshow('frame', im)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break


# After the loop release the cap object
#vid.release()
#video_writer.release()

# Destroy all the windows
cv2.destroyAllWindows()