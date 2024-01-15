import os
import cv2
import random
import numpy as np


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
# zestaw funkcji przygotowujacych zmodyfikowane obrazki do testowania
#
# 2024-01-10
#
################################################################################################


def lista_obrazkow_testowych(dir_path):
    # lista plikow z obrazkami do wczytania
    print(dir_path)
    fnames = []
    cnames = []
    cnames_tab = []

    # lista klas
    for cname in os.listdir(dir_path):
        # check if current path is a file
        # class_path = dir_path + cname
        class_path = os.path.join(dir_path, cname)
        print(cname, class_path)
        cnames_tab.append(cname)

        for imagename in os.listdir(class_path):
            # print(class_path, imagename)
            file_path = os.path.join(class_path, imagename)
            # print(file_path)
            if os.path.isfile(file_path):
                # check only text files
                if file_path.endswith('.jpg'):
                    fnames.append(imagename)
                    cnames.append(cname)
                    print(imagename,cname)
#    print(len(fnames))

    return (fnames, cnames, cnames_tab)


def skaluj_obrazek(image_f, size_base, mod_size=0):
    size = size_base + mod_size
#    print(size)
    r = size / image_f.shape[0]
    dim = (int(image_f.shape[1] * r), int(size))
    image = cv2.resize(image_f, dim, interpolation=cv2.INTER_AREA)
    return image


def wklej_obrazek(image_bgt,image,x_offset,y_offset):
#    x_offset = random.randint(50,500)
#    y_offset = random.randint(50,300)
    if y_offset+image.shape[0] > image_bgt.shape[0] :
        y_offset = image_bgt.shape[0]-image.shape[0]-1
    if x_offset+image.shape[1] > image_bgt.shape[1] :
        x_offset = image_bgt.shape[1]-image.shape[1]-1
    if y_offset <= 0 :
        y_offset = 1
    if x_offset <= 0 :
        x_offset = 1
#    print(y_offset,x_offset)
    image_bgt[y_offset:y_offset+image.shape[0], x_offset:x_offset+image.shape[1]] = image

    return image_bgt


def skaluj_jasnosc(image, brightness):
    contrast = 1.
#    brightness = 2.  # Brightness control (-20 - 20)
#    print(brightness)

    image_out = cv2.addWeighted(image, contrast, image, 0, brightness)
    return image_out


def dodaj_szum(image,std):
    # Generate random Gaussian noise
    mean = 0
    stddev = std, std/2, std
    noise = np.zeros(image.shape, np.uint8)
    cv2.randn(noise, mean, stddev)

    # Add noise to image
    image_out = cv2.add(image, noise)
    return image_out


def wczytaj_tlo():
    image_bg = cv2.imread(os.path.join(os.getcwd(), 'miasto_dzien1.jpg'))

    r = 500.0 / image_bg.shape[1]
    dim = (500, int(image_bg.shape[0] * r))
    image_bg = cv2.resize(image_bg, dim, interpolation=cv2.INTER_AREA)
    return image_bg


#########################################################################################################

# main do testowania tworzenia symulacji

if __name__ == '__main__':

    image_bg = wczytaj_tlo()

    cv2.imshow('Image', image_bg)
    cv2.waitKey(0)

    IMG_SIZE = (224, 224)

    path = os.getcwd()

    path = os.path.join(path, 'dataset_big')
    path_test = os.path.join(path, 'test')

    fnames, cnames, class_names = lista_obrazkow_testowych(path_test)

    print(class_names)

    # jesli ma byc powtarzalne
    np.random.seed(4)

    for i in range(len(fnames)):
    # Retrieve a batch of images from the test set
        image_path = os.path.join(path_test, cnames[i], fnames[i])
        print(image_path, cnames[i], fnames[i])
        image_f = cv2.imread(image_path)  # surowy obrazek

    #    cv2.imshow('Image', image_f)
    #    cv2.waitKey(0)

        x_offset = random.randint(50, 500)
        y_offset = random.randint(50, 300)

        for size in range(-250,50,50):
            for bright in range(-80,60,20):
                for noise in range(0,50,10):
                    # czysty obrazek tla do modyfkacji
                    image_bgt = image_bg.copy()

                    # zmiana rozmiaru obrazka (do standardowego)
                    image_s = skaluj_obrazek(image_f, 300, size)

    #                cv2.imshow('Image size', image_s)
    #                cv2.waitKey(0)

                    # zmiana jasnosci obrazka
                    image_br = skaluj_jasnosc(image_s, bright)

    #                cv2.imshow('Image br', image_br)
    #                cv2.waitKey(0)

                    # wklejenie na tlo

    #                print(x_offset,int(image_br.shape[1]/2),y_offset,int(image_br.shape[0]/2))
                    wklej_obrazek(image_bgt,image_br,x_offset-int(image_br.shape[1]/2),y_offset-int(image_br.shape[0]/2))

                    # dodawanie szumu do calosci
                    image_final = dodaj_szum(image_bgt,noise)

                    # detekcja

                    # wynik
                    print(size, bright, noise)
                    cv2.imshow('Image', image_final)
                    cv2.waitKey(100)
