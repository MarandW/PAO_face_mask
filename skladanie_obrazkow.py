import cv2

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
# zestaw funkcji generujacych ilustracje do raportu
#
# 2024-01-10
#
################################################################################################


path1 = 'wykresy v1/'
path2 = 'wykresy v2/'
path3 = 'wykresy v3/'
path4 = 'wykresy TFL/'

paths = [path1, path2, path3, path4]

## wykres poprawnosci detekcji

for path in paths:
    imagename1 = 'map_size_vs_br_poprawna_praca_programu.png'
    imagename2 = 'map_size_vs_br_poprawna_praca_programu_n.png'

    imagename3 = 'map_size_vs_br_falszywa_detekcja.png'
    imagename4 = 'map_size_vs_br_falszywa_detekcja_n.png'


    image1 = cv2.imread(path + imagename1)
    image2 = cv2.imread(path + imagename2)

    image3 = cv2.imread(path + imagename3)
    image4 = cv2.imread(path + imagename4)

    v_img1 = cv2.vconcat([image1, image2])

    v_img2 = cv2.vconcat([image3, image4])

    h_img1 = cv2.hconcat([v_img1, v_img2])

    cv2.imshow("h_img", h_img1)
    if path == path1 :
        cv2.imwrite('figure_poprawnosc_dzialania_InceptionV3.png', h_img1)
    if path == path2 :
        cv2.imwrite('figure_poprawnosc_dzialania_InceptionV3_cvzone.png', h_img1)
    if path == path3 :
        cv2.imwrite('figure_poprawnosc_dzialania_MobileNetV2_cvzone.png', h_img1)
    if path == path4 :
        cv2.imwrite('figure_poprawnosc_dzialania_SSD_MobileNetV2_FPNLite.png', h_img1)

    key = cv2.waitKey(0)

# Skutecznosc wykrycia twarzy

# wielkosc jasnosc

imagename1 = 'map_size_vs_br_szum0_udane_wykrycie_twarzy.png'
imagename2 = 'map_size_vs_br_szum0_udane_wykrycie_twarzy_bez_maski.png'
imagename3 = 'map_size_vs_br_szum0_udane_wykrycie_twarzy_z_maska.png'

path = path1
image1 = cv2.imread(path + imagename1)
image2 = cv2.imread(path + imagename2)
image3 = cv2.imread(path + imagename3)
h_img1 = cv2.hconcat([image1, image2, image3])

path = path2
image1 = cv2.imread(path + imagename1)
image2 = cv2.imread(path + imagename2)
image3 = cv2.imread(path + imagename3)
h_img2 = cv2.hconcat([image1, image2, image3])

path = path3
image1 = cv2.imread(path + imagename1)
image2 = cv2.imread(path + imagename2)
image3 = cv2.imread(path + imagename3)
h_img3 = cv2.hconcat([image1, image2, image3])

path = path4
image1 = cv2.imread(path + imagename1)
image2 = cv2.imread(path + imagename2)
image3 = cv2.imread(path + imagename3)
h_img4 = cv2.hconcat([image1, image2, image3])

v_img1 = cv2.vconcat([h_img1, h_img2, h_img3, h_img4])

cv2.imshow("v_img", v_img1)
cv2.imwrite('figure_wykrycie_twarzy_1.png', v_img1)

key = cv2.waitKey(0)

# wielkosc szum

imagename1 = 'map_size_vs_noise_br3_udane_wykrycie_twarzy.png'
imagename2 = 'map_size_vs_noise_br3_udane_wykrycie_twarzy_bez_maski.png'
imagename3 = 'map_size_vs_noise_br3_udane_wykrycie_twarzy__z_maska.png'
path = path1
image1 = cv2.imread(path + imagename1)
image2 = cv2.imread(path + imagename2)
image3 = cv2.imread(path + imagename3)
h_img1 = cv2.hconcat([image1, image2, image3])

path = path2
image1 = cv2.imread(path + imagename1)
image2 = cv2.imread(path + imagename2)
image3 = cv2.imread(path + imagename3)
h_img2 = cv2.hconcat([image1, image2, image3])

path = path3
image1 = cv2.imread(path + imagename1)
image2 = cv2.imread(path + imagename2)
image3 = cv2.imread(path + imagename3)
h_img3 = cv2.hconcat([image1, image2, image3])

path = path4
image1 = cv2.imread(path + imagename1)
image2 = cv2.imread(path + imagename2)
image3 = cv2.imread(path + imagename3)
h_img4 = cv2.hconcat([image1, image2, image3])

v_img1 = cv2.vconcat([h_img1, h_img2, h_img3, h_img4])

cv2.imshow("v_img", v_img1)
cv2.imwrite('figure_wykrycie_twarzy_2.png', v_img1)

key = cv2.waitKey(0)


# Skutecznosc wykrycia twarzy

# wielkosc jasnosc

imagename1 = 'map_size_vs_br_szum0_udane_wykrycie_twarzy.png'
imagename2 = 'map_size_vs_br_szum0_prawidlowe_okreslenie_twarzy_bez_maski.png'
imagename3 = 'map_size_vs_br_szum0_prawidlowe_okreslenie_twarzy_z_maska.png'

path = path1
image1 = cv2.imread(path + imagename1)
image2 = cv2.imread(path + imagename2)
image3 = cv2.imread(path + imagename3)
h_img1 = cv2.hconcat([image1, image2, image3])

path = path2
image1 = cv2.imread(path + imagename1)
image2 = cv2.imread(path + imagename2)
image3 = cv2.imread(path + imagename3)
h_img2 = cv2.hconcat([image1, image2, image3])

path = path3
image1 = cv2.imread(path + imagename1)
image2 = cv2.imread(path + imagename2)
image3 = cv2.imread(path + imagename3)
h_img3 = cv2.hconcat([image1, image2, image3])

path = path4
image1 = cv2.imread(path + imagename1)
image2 = cv2.imread(path + imagename2)
image3 = cv2.imread(path + imagename3)
h_img4 = cv2.hconcat([image1, image2, image3])

v_img1 = cv2.vconcat([h_img1, h_img2, h_img3, h_img4])

cv2.imshow("v_img", v_img1)
cv2.imwrite('figure_wykrycie_maski_1.png', v_img1)

key = cv2.waitKey(0)



imagename1 = 'map_size_vs_br_szum0_udane_wykrycie_twarzy.png'
imagename2 = 'map_size_vs_br_szum3_prawidlowe_okreslenie_twarzy_bez_maski.png'
imagename3 = 'map_size_vs_br_szum3_prawidlowe_okreslenie_twarzy_z_maska.png'

path = path1
image1 = cv2.imread(path + imagename1)
image2 = cv2.imread(path + imagename2)
image3 = cv2.imread(path + imagename3)
h_img1 = cv2.hconcat([image1, image2, image3])

path = path2
image1 = cv2.imread(path + imagename1)
image2 = cv2.imread(path + imagename2)
image3 = cv2.imread(path + imagename3)
h_img2 = cv2.hconcat([image1, image2, image3])

path = path3
image1 = cv2.imread(path + imagename1)
image2 = cv2.imread(path + imagename2)
image3 = cv2.imread(path + imagename3)
h_img3 = cv2.hconcat([image1, image2, image3])

path = path4
image1 = cv2.imread(path + imagename1)
image2 = cv2.imread(path + imagename2)
image3 = cv2.imread(path + imagename3)
h_img4 = cv2.hconcat([image1, image2, image3])

v_img1 = cv2.vconcat([h_img1, h_img2, h_img3, h_img4])

cv2.imshow("v_img", v_img1)
cv2.imwrite('figure_wykrycie_maski_1_szum.png', v_img1)

key = cv2.waitKey(0)

# wielkosc szum

imagename1 = 'map_size_vs_noise_br3_udane_wykrycie_twarzy.png'
imagename2 = 'map_size_vs_noise_br3_prawidlowe_okreslenie_twarzy_bez_maski.png'
imagename3 = 'map_size_vs_noise_br3_prawidlowe_okreslenie_twarzy_z_maska.png'
path = path1
image1 = cv2.imread(path + imagename1)
image2 = cv2.imread(path + imagename2)
image3 = cv2.imread(path + imagename3)
h_img1 = cv2.hconcat([image1, image2, image3])

path = path2
image1 = cv2.imread(path + imagename1)
image2 = cv2.imread(path + imagename2)
image3 = cv2.imread(path + imagename3)
h_img2 = cv2.hconcat([image1, image2, image3])

path = path3
image1 = cv2.imread(path + imagename1)
image2 = cv2.imread(path + imagename2)
image3 = cv2.imread(path + imagename3)
h_img3 = cv2.hconcat([image1, image2, image3])

path = path4
image1 = cv2.imread(path + imagename1)
image2 = cv2.imread(path + imagename2)
image3 = cv2.imread(path + imagename3)
h_img4 = cv2.hconcat([image1, image2, image3])

v_img1 = cv2.vconcat([h_img1, h_img2, h_img3, h_img4])

cv2.imshow("v_img", v_img1)
cv2.imwrite('figure_wykrycie_maski_2.png', v_img1)

key = cv2.waitKey(0)

# skrajny przypadek

# wielkosc jasnosc

# wielkosc szum

imagename1 = 'map_size_vs_noise_br0_udane_wykrycie_twarzy.png'
imagename2 = 'map_size_vs_noise_br0_prawidlowe_okreslenie_twarzy_bez_maski.png'
imagename3 = 'map_size_vs_noise_br0_prawidlowe_okreslenie_twarzy_z_maska.png'
path = path1
image1 = cv2.imread(path + imagename1)
image2 = cv2.imread(path + imagename2)
image3 = cv2.imread(path + imagename3)
h_img1 = cv2.hconcat([image1, image2, image3])

path = path2
image1 = cv2.imread(path + imagename1)
image2 = cv2.imread(path + imagename2)
image3 = cv2.imread(path + imagename3)
h_img2 = cv2.hconcat([image1, image2, image3])

path = path3
image1 = cv2.imread(path + imagename1)
image2 = cv2.imread(path + imagename2)
image3 = cv2.imread(path + imagename3)
h_img3 = cv2.hconcat([image1, image2, image3])

path = path4
image1 = cv2.imread(path + imagename1)
image2 = cv2.imread(path + imagename2)
image3 = cv2.imread(path + imagename3)
h_img4 = cv2.hconcat([image1, image2, image3])

v_img1 = cv2.vconcat([h_img1, h_img2, h_img3, h_img4])

cv2.imshow("v_img", v_img1)
cv2.imwrite('figure_wykrycie_maski_2_ciemno.png', v_img1)

key = cv2.waitKey(0)

# wielkosc szum

imagename1 = 'map_size_vs_noise_br5_udane_wykrycie_twarzy.png'
imagename2 = 'map_size_vs_noise_br5_prawidlowe_okreslenie_twarzy_bez_maski.png'
imagename3 = 'map_size_vs_noise_br5_prawidlowe_okreslenie_twarzy_z_maska.png'
path = path1
image1 = cv2.imread(path + imagename1)
image2 = cv2.imread(path + imagename2)
image3 = cv2.imread(path + imagename3)
h_img1 = cv2.hconcat([image1, image2, image3])

path = path2
image1 = cv2.imread(path + imagename1)
image2 = cv2.imread(path + imagename2)
image3 = cv2.imread(path + imagename3)
h_img2 = cv2.hconcat([image1, image2, image3])

path = path3
image1 = cv2.imread(path + imagename1)
image2 = cv2.imread(path + imagename2)
image3 = cv2.imread(path + imagename3)
h_img3 = cv2.hconcat([image1, image2, image3])

path = path4
image1 = cv2.imread(path + imagename1)
image2 = cv2.imread(path + imagename2)
image3 = cv2.imread(path + imagename3)
h_img4 = cv2.hconcat([image1, image2, image3])

v_img1 = cv2.vconcat([h_img1, h_img2, h_img3, h_img4])

cv2.imshow("v_img", v_img1)
cv2.imwrite('figure_wykrycie_maski_2_jasno.png', v_img1)

key = cv2.waitKey(0)

