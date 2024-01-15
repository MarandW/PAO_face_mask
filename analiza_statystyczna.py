import numpy as np
import matplotlib.pyplot as plt


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
# zestaw funkcji analizujacej wyniki i generujacej obrazki
#
# 2024-01-10
#
################################################################################################


tabela = np.zeros(shape=(11,6,6,2,3))


def czytaj_plik(fname, tabela):

    with open(fname) as f:
        lines = f.readlines()
        for line in lines:
            if len(line) > 1:
    #            print(line)
    #            print(len(line.split()))
    #            filename = line.split(' ')
                filename, id, _kategoria, _detekcja, _prob, _size, _br, _noise, _ = line.split(' ')
                kategoria = int(_kategoria)
                detekcja = int(_detekcja)
                prob = float(_prob)
                size = int(_size)
                br = int(_br)
                noise = int(_noise)
    #            print(int((size+275)/25), int((br+150)/50), int(noise/25), kategoria, detekcja+1)
                tabela[int((size+275)/25)][int((br+150)/50)][int(noise/25)][kategoria][detekcja+1] += 1

    return tabela


def size_vs_noise_br_detekcja(map_size_vs_noise, total, br, detekcja, invers=False):
    for size in range(11):
#        for br in range(6):
        for noise in range(6):
#            noise = 0  # wynik dla braku szumu
            for kategoria in range(2):
                # wszystkie razem = 666
                # for detekcja in range(3):
                #    map_size_vs_br[size][br] += tabela[size][br][noise][kategoria][detekcja]
#                detekcja = 0  # tylko brak detekcji
                map_size_vs_noise[size][noise] += tabela[size][br][noise][kategoria][detekcja]
            if invers :
                map_size_vs_noise[size][noise] = 100. - map_size_vs_noise[size][noise] / total * 100.
            else :
                map_size_vs_noise[size][noise] = map_size_vs_noise[size][noise] / total * 100.

    np.set_printoptions(precision=1)
    print("size vs br - udane wykrycie twarzy")
    print(map_size_vs_noise)

    return map_size_vs_noise



def size_vs_noise_br_kategoria_detekcja(map_size_vs_noise, total, br, kategoria, detekcja, invers=False):
    for size in range(11):
#        for br in range(6):
        for noise in range(6):
#            noise = 0  # wynik dla braku szumu
            #            for kategoria in range(2):
#            kategoria = 0  # tylko bez maski
            # wszystkie razem = 666
            # for detekcja in range(3):
            #    map_size_vs_br[size][br] += tabela[size][br][noise][kategoria][detekcja]
#            detekcja = 0  # tylko brak detekcji

            map_size_vs_noise[size][noise] += tabela[size][br][noise][kategoria][detekcja]
            if invers :
                map_size_vs_noise[size][noise] = 100. - map_size_vs_noise[size][noise] / total * 100.
            else :
                map_size_vs_noise[size][noise] = map_size_vs_noise[size][noise] / total * 100.

    np.set_printoptions(precision=1)
#    print("size vs br - udane wykrycie twarzy bez maski")
    print(map_size_vs_noise)

    return (map_size_vs_noise)


def size_vs_br_dobre(map_size_vs_br, total, invers=False):
    for size in range(11):
        for br in range(6):

            # wypelnianie komorki
            for noise in range(6):
                for kategoria in range(2):
                    for detekcja in range(3):
                        # map_size_vs_br[size][br] += tabela[size][br][noise][kategoria][detekcja]
                        # Warunek na skutecznosc metody
                        if noise < 2 and ( (kategoria == 0 and detekcja == 1) or (kategoria == 1 and detekcja == 2) ):
                            # szum 0 lub 1 i poprawna detekcja bez maski lub poprawna detekcja z maska
                            map_size_vs_br[size][br] += tabela[size][br][noise][kategoria][detekcja]

            # wynik w komorce i jego inwersja
            if invers :
                map_size_vs_br[size][br] = 100. - map_size_vs_br[size][br] / total /2. * 100.
            else :
                map_size_vs_br[size][br] = map_size_vs_br[size][br] / total /2. * 100.
            # dzieli na 2 bo suma dwoch szumow

    np.set_printoptions(precision=1)
#    print("size vs br - udane wykrycie twarzy")
    print(map_size_vs_br)

    return map_size_vs_br


def size_vs_br_dobre_n(map_size_vs_br, total, invers=False):
    for size in range(11):
        for br in range(6):

            # wypelnianie komorki
            for noise in range(6):
                for kategoria in range(2):
                    for detekcja in range(3):
                        # map_size_vs_br[size][br] += tabela[size][br][noise][kategoria][detekcja]
                        # Warunek na skutecznosc metody
                        if 1 < noise < 4 and ( (kategoria == 0 and detekcja == 1) or (kategoria == 1 and detekcja == 2) ):
                            # szum 0 lub 1 i poprawna detekcja bez maski lub poprawna detekcja z maska
                            map_size_vs_br[size][br] += tabela[size][br][noise][kategoria][detekcja]

            # wynik w komorce i jego inwersja
            if invers :
                map_size_vs_br[size][br] = 100. - map_size_vs_br[size][br] / total /2. * 100.
            else :
                map_size_vs_br[size][br] = map_size_vs_br[size][br] / total /2. * 100.
            # dzieli na 2 bo suma dwoch szumow

    np.set_printoptions(precision=1)
#    print("size vs br - udane wykrycie twarzy")
    print(map_size_vs_br)

    return map_size_vs_br


def size_vs_br_zle(map_size_vs_br, total, invers=False):
    for size in range(11):
        for br in range(6):

            # wypelnianie komorki
            for noise in range(6):
                for kategoria in range(2):
                    for detekcja in range(3):
                        # map_size_vs_br[size][br] += tabela[size][br][noise][kategoria][detekcja]
                        # Warunek na skutecznosc metody
                        if noise < 2 and ( (kategoria == 0 and detekcja == 2) or (kategoria == 1 and detekcja == 1) ):
                            # szum 0 lub 1 i poprawna detekcja bez maski lub poprawna detekcja z maska
                            map_size_vs_br[size][br] += tabela[size][br][noise][kategoria][detekcja]

            # wynik w komorce i jego inwersja
            if invers :
                map_size_vs_br[size][br] = 100. - map_size_vs_br[size][br] / total /2. * 100.
            else :
                map_size_vs_br[size][br] = map_size_vs_br[size][br] / total /2. * 100.
            # dzieli na 2 bo suma dwoch szumow

    np.set_printoptions(precision=1)
#    print("size vs br - udane wykrycie twarzy")
    print(map_size_vs_br)

    return map_size_vs_br


def size_vs_br_zle_n(map_size_vs_br, total, invers=False):
    for size in range(11):
        for br in range(6):

            # wypelnianie komorki
            for noise in range(6):
                for kategoria in range(2):
                    for detekcja in range(3):
                        # map_size_vs_br[size][br] += tabela[size][br][noise][kategoria][detekcja]
                        # Warunek na skutecznosc metody
                        if 1< noise < 4 and ( (kategoria == 0 and detekcja == 2) or (kategoria == 1 and detekcja == 1) ):
                            # szum 0 lub 1 i poprawna detekcja bez maski lub poprawna detekcja z maska
                            map_size_vs_br[size][br] += tabela[size][br][noise][kategoria][detekcja]

            # wynik w komorce i jego inwersja
            if invers :
                map_size_vs_br[size][br] = 100. - map_size_vs_br[size][br] / total /2. * 100.
            else :
                map_size_vs_br[size][br] = map_size_vs_br[size][br] / total /2. * 100.
            # dzieli na 2 bo suma dwoch szumow

    np.set_printoptions(precision=1)
#    print("size vs br - udane wykrycie twarzy")
    print(map_size_vs_br)

    return map_size_vs_br


def podsumowanie_wyniku(map_size_vs_br, map_size_vs_br_n, fpname, typ):
    dobre = 0
    i = 0
    trudne = 0
    j = 0
    for size in range(11):
        for br in range(6):
            # warunki na sumowanie wyniku dobrego
            if 4 < size < 11 and 1 < br < 5 :
                dobre += map_size_vs_br[size][br]
                i += 1
            # warunki na sumowanie wyniku trudne
            if 0 < size < 5 and 0 < br < 6 :
                trudne += map_size_vs_br[size][br]
                j += 1
            if 4 < size < 11 and ( br == 1 or br == 5) :
                trudne += map_size_vs_br[size][br]
                j += 1

            if 0 < size < 11 and 0 < br < 6 :
                trudne += map_size_vs_br_n[size][br]
                j += 1


    dobre = dobre / i
    trudne = trudne / j
    out_line = f'dobre: {dobre:.2f} trudne: {trudne:.2f} \n'
    print(out_line)
    f = open(fpname, typ)
    f.write(out_line)
    f.close()


def size_vs_br_noise_detekcja(map_size_vs_br, total, noise, detekcja, invers=False):
    for size in range(11):
        for br in range(6):
            #        for noise in range(6):
#            noise = 0  # wynik dla braku szumu
            for kategoria in range(2):
                # wszystkie razem = 666
                # for detekcja in range(3):
                #    map_size_vs_br[size][br] += tabela[size][br][noise][kategoria][detekcja]
#                detekcja = 0  # tylko brak detekcji
                map_size_vs_br[size][br] += tabela[size][br][noise][kategoria][detekcja]
            if invers :
                map_size_vs_br[size][br] = 100. - map_size_vs_br[size][br] / total * 100.
            else :
                map_size_vs_br[size][br] = map_size_vs_br[size][br] / total * 100.

    np.set_printoptions(precision=1)
#    print("size vs br - udane wykrycie twarzy")
    print(map_size_vs_br)

    return map_size_vs_br


def size_vs_br_noise_kategoria_detekcja(map_size_vs_br, total, noise, kategoria, detekcja, invers=False):
    for size in range(11):
        for br in range(6):
            #        for noise in range(6):
#            noise = 0  # wynik dla braku szumu
            #            for kategoria in range(2):
#            kategoria = 0  # tylko bez maski
            # wszystkie razem = 666
            # for detekcja in range(3):
            #    map_size_vs_br[size][br] += tabela[size][br][noise][kategoria][detekcja]
#            detekcja = 0  # tylko brak detekcji

            map_size_vs_br[size][br] += tabela[size][br][noise][kategoria][detekcja]
            if invers :
                map_size_vs_br[size][br] = 100. - map_size_vs_br[size][br] / total * 100.
            else :
                map_size_vs_br[size][br] = map_size_vs_br[size][br] / total * 100.

    np.set_printoptions(precision=1)
#    print("size vs br - udane wykrycie twarzy bez maski")
    print(map_size_vs_br)

    return (map_size_vs_br)


def wykres_size_vs_noise(map_size_vs_noise, title_text='Title',filename='wykres.png'):
    lsize = np.arange(25, 300, 25)
    for i in range(len(lsize)):
        lsize[i] = round(lsize[i])

    lbr = np.arange(0.0, 150, 25)
    for i in range(len(lbr)):
        lbr[i] = round(lbr[i])

    fig, ax = plt.subplots()
    cax = ax.imshow(map_size_vs_noise.T, interpolation='none', vmin=0, vmax=100)
    ax.set_xticks(np.arange(len(lsize)), labels=lsize)
    ax.set_yticks(np.arange(len(lbr)), labels=lbr)
    # cbar = fig.colorbar(cax,ticks=ticks_at,format='%1.2g')
    # cbar = fig.colorbar(cax,format='%1.2g')

    cbar = fig.colorbar(cax, fraction=0.026, pad=0.04)
    w = ax.invert_yaxis()
    # ax.set_title('size vs br - brak face detection total', fontsize = 14, fontweight ='bold')
    ax.set_title(title_text)
    plt.xlabel("Rozmiar [pix]")
    plt.ylabel("szum")

    # Loop over data dimensions and create text annotations.
    for i in range(11):
        for j in range(6):
            text = ax.text(i, j, '{:.0f}'.format(map_size_vs_noise[i, j]),
                           ha="center", va="center", color="w")

    plt.savefig(filename)
    plt.show()



def wykres_size_vs_br(map_size_vs_br, title_text='Title',filename='wykres.png'):
    lsize = np.arange(25, 300, 25)
    for i in range(len(lsize)):
        lsize[i] = round(lsize[i])

    lbr = np.arange(-150.0, 150, 50)
    for i in range(len(lbr)):
        lbr[i] = round(lbr[i])

    fig, ax = plt.subplots()
    cax = ax.imshow(map_size_vs_br.T, interpolation='none', vmin=0, vmax=100)
    ax.set_xticks(np.arange(len(lsize)), labels=lsize)
    ax.set_yticks(np.arange(len(lbr)), labels=lbr)
    # cbar = fig.colorbar(cax,ticks=ticks_at,format='%1.2g')
    # cbar = fig.colorbar(cax,format='%1.2g')

    cbar = fig.colorbar(cax, fraction=0.026, pad=0.04)
    w = ax.invert_yaxis()
    # ax.set_title('size vs br - brak face detection total', fontsize = 14, fontweight ='bold')
    ax.set_title(title_text)
    plt.xlabel("Rozmiar [pix]")
    plt.ylabel("Jasność")

    # Loop over data dimensions and create text annotations.
    for i in range(11):
        for j in range(6):
            text = ax.text(i, j, '{:.0f}'.format(map_size_vs_br[i, j]),
                           ha="center", va="center", color="w")

    plt.savefig(filename)
    plt.show()


def analizuj_wynik(fname,model,path):

    czytaj_plik(fname, tabela)

    for i in range(11):
        total_bezmaks = tabela[i][0][0][0][0] + tabela[i][0][0][0][1] + tabela[i][0][0][0][2]
        total_maks = tabela[i][0][0][1][0] + tabela[i][0][0][1][1] + tabela[i][0][0][1][2]
        total = total_bezmaks + total_maks
    print("total:", total,total_bezmaks,total_maks)

    # mapy sukcesu albo porazki

    #########################################################

    # jak dobra jest detekcja
    outname = path +"ostateczny_wnik_" + model + ".txt"
    out_line = model + "\n"
    f = open(outname, "w")
    f.write(out_line)
    f.close()

    map_size_vs_br = np.zeros(shape=(11, 6))
    size_vs_br_dobre(map_size_vs_br, total, invers=False)
    ttext = model + '\n Poprawne wykonanie zadania \n szum std = 0 i 25'
    fout = path + 'map_size_vs_br_poprawna_praca_programu.png'
    wykres_size_vs_br(map_size_vs_br, title_text=ttext,
                      filename=fout)

    # jeszcze warstwa z szumem
    map_size_vs_br_n = np.zeros(shape=(11, 6))
    size_vs_br_dobre_n(map_size_vs_br_n, total, invers=False)
    ttext = model + '\n Poprawne wykonanie zadania \n szum std = 50 i 75'
    fout = path + 'map_size_vs_br_poprawna_praca_programu_n.png'
    wykres_size_vs_br(map_size_vs_br_n, title_text=ttext,
                  filename=fout)

    podsumowanie_wyniku(map_size_vs_br, map_size_vs_br_n, outname, "a")

    #########################################################

    map_size_vs_br = np.zeros(shape=(11, 6))
    size_vs_br_zle(map_size_vs_br, total, invers=False)
    ttext = model + '\n Fałszywa detekcja \n szum = 0 i 25'
    fout = path + 'map_size_vs_br_falszywa_detekcja.png'
    wykres_size_vs_br(map_size_vs_br, title_text=ttext,
                      filename=fout)

    map_size_vs_br_n = np.zeros(shape=(11, 6))
    size_vs_br_zle_n(map_size_vs_br_n, total, invers=False)
    ttext = model + '\n Fałszywa detekcja \n szum = 50 i 75'
    fout = path + 'map_size_vs_br_falszywa_detekcja_n.png'
    wykres_size_vs_br(map_size_vs_br_n, title_text=ttext,
                  filename=fout)

    podsumowanie_wyniku(map_size_vs_br, map_size_vs_br_n, outname, "a")

    #########################################################

    ## jasnosc 0

    #########################################################

    map_size_vs_noise = np.zeros(shape=(11, 6))
    size_vs_noise_br_detekcja(map_size_vs_noise, total, br=0, detekcja=0, invers=True)
    ttext = model + '\n Udane wykrycie twarzy \n jasność = -150'
    fout = path + 'map_size_vs_noise_br0_udane_wykrycie_twarzy.png'
    wykres_size_vs_noise(map_size_vs_noise, title_text=ttext,
                      filename=fout)

    #########################################################

    map_size_vs_noise = np.zeros(shape=(11, 6))
    size_vs_noise_br_kategoria_detekcja(map_size_vs_noise, total_bezmaks, br=0, kategoria=0,detekcja=0, invers=True)
    ttext = model + '\n Udane wykrycie twarzy bez maski\n jasność = -150'
    fout = path + 'map_size_vs_noise_br0_udane_wykrycie_twarzy_bez_maski.png'
    wykres_size_vs_noise(map_size_vs_noise,title_text=ttext,
                      filename=fout)

    #########################################################

    map_size_vs_noise = np.zeros(shape=(11, 6))
    size_vs_noise_br_kategoria_detekcja(map_size_vs_noise, total_maks, br=0, kategoria=1,detekcja=0, invers=True)
    ttext = model + '\n Udane wykrycie twarzy z maską\n jasność = -150'
    fout = path + 'map_size_vs_noise_br0_udane_wykrycie_twarzy__z_maska.png'
    wykres_size_vs_noise(map_size_vs_noise,title_text=ttext,
                      filename=fout)

    #####################################################################################

    map_size_vs_noise = np.zeros(shape=(11, 6))
    size_vs_noise_br_kategoria_detekcja(map_size_vs_noise, total_bezmaks, br=0, kategoria=0,detekcja=1)
    ttext = model + '\n Prawidłowe rozpoznanie twarzy bez maski\n jasność = -150'
    fout = path + 'map_size_vs_noise_br0_prawidlowe_okreslenie_twarzy_bez_maski.png'
    wykres_size_vs_noise(map_size_vs_noise,title_text=ttext,
                      filename=fout)

    #####################################################################################

    map_size_vs_noise = np.zeros(shape=(11, 6))
    size_vs_noise_br_kategoria_detekcja(map_size_vs_noise, total_maks, br=0, kategoria=1,detekcja=2)
    ttext = model + '\n Prawidłowe rozpoznanie twarzy z maską\n jasność = -150'
    fout = path + 'map_size_vs_noise_br0_prawidlowe_okreslenie_twarzy_z_maska.png'
    wykres_size_vs_noise(map_size_vs_noise,title_text=ttext,
                      filename=fout)

    #####################################################################################

    map_size_vs_noise = np.zeros(shape=(11, 6))
    size_vs_noise_br_kategoria_detekcja(map_size_vs_noise, total_bezmaks, br=0, kategoria=0,detekcja=2)
    ttext = model + '\n Nieprawidłowe rozpoznanie twarzy bez maski\n jasność = -150'
    fout = path + 'map_size_vs_noise_br0_nieprawidlowe_okreslenie_twarzy_bez_maski.png'
    wykres_size_vs_noise(map_size_vs_noise,title_text=ttext,
                      filename=fout)

    #####################################################################################

    map_size_vs_noise = np.zeros(shape=(11, 6))
    size_vs_noise_br_kategoria_detekcja(map_size_vs_noise, total_maks, br=0, kategoria=1,detekcja=1)
    ttext = model + '\n Nieprawidłowe rozpoznanie twarzy z maską\n jasność = -150'
    fout = path + 'map_size_vs_noise_br0_nieprawidlowe_okreslenie_twarzy_z_maska.png'
    wykres_size_vs_noise(map_size_vs_noise,title_text=ttext,
                      filename=fout)

    #####################################################################################



    ## jasnosc 3

    #########################################################

    map_size_vs_noise = np.zeros(shape=(11, 6))
    size_vs_noise_br_detekcja(map_size_vs_noise, total, br=3, detekcja=0, invers=True)
    ttext = model + '\n Udane wykrycie twarzy \n jasność = 0'
    fout = path + 'map_size_vs_noise_br3_udane_wykrycie_twarzy.png'
    wykres_size_vs_noise(map_size_vs_noise, title_text=ttext,
                      filename=fout)

    #########################################################

    map_size_vs_noise = np.zeros(shape=(11, 6))
    size_vs_noise_br_kategoria_detekcja(map_size_vs_noise, total_bezmaks, br=3, kategoria=0,detekcja=0, invers=True)
    ttext = model + '\n Udane wykrycie twarzy bez maski\n jasność = 0'
    fout = path + 'map_size_vs_noise_br3_udane_wykrycie_twarzy_bez_maski.png'
    wykres_size_vs_noise(map_size_vs_noise,title_text=ttext,
                      filename=fout)

    #########################################################

    map_size_vs_noise = np.zeros(shape=(11, 6))
    size_vs_noise_br_kategoria_detekcja(map_size_vs_noise, total_maks, br=3, kategoria=1,detekcja=0, invers=True)
    ttext = model + '\n Udane wykrycie twarzy z maską\n jasność = 0'
    fout = path + 'map_size_vs_noise_br3_udane_wykrycie_twarzy__z_maska.png'
    wykres_size_vs_noise(map_size_vs_noise,title_text=ttext,
                      filename=fout)

    #####################################################################################

    map_size_vs_noise = np.zeros(shape=(11, 6))
    size_vs_noise_br_kategoria_detekcja(map_size_vs_noise, total_bezmaks, br=3, kategoria=0,detekcja=1)
    ttext = model + '\n Prawidłowe rozpoznanie twarzy bez maski\n jasność = 0'
    fout = path + 'map_size_vs_noise_br3_prawidlowe_okreslenie_twarzy_bez_maski.png'
    wykres_size_vs_noise(map_size_vs_noise,title_text=ttext,
                      filename=fout)

    #####################################################################################

    map_size_vs_noise = np.zeros(shape=(11, 6))
    size_vs_noise_br_kategoria_detekcja(map_size_vs_noise, total_maks, br=3, kategoria=1,detekcja=2)
    ttext = model + '\n Prawidłowe rozpoznanie twarzy z maską\n jasność = 0'
    fout = path + 'map_size_vs_noise_br3_prawidlowe_okreslenie_twarzy_z_maska.png'
    wykres_size_vs_noise(map_size_vs_noise,title_text=ttext,
                      filename=fout)

    #####################################################################################

    map_size_vs_noise = np.zeros(shape=(11, 6))
    size_vs_noise_br_kategoria_detekcja(map_size_vs_noise, total_bezmaks, br=3, kategoria=0,detekcja=2)
    ttext = model + '\n Nieprawidłowe rozpoznanie twarzy bez maski\n jasność = 0'
    fout = path + 'map_size_vs_noise_br3_nieprawidlowe_okreslenie_twarzy_bez_maski.png'
    wykres_size_vs_noise(map_size_vs_noise,title_text=ttext,
                      filename=fout)

    #####################################################################################

    map_size_vs_noise = np.zeros(shape=(11, 6))
    size_vs_noise_br_kategoria_detekcja(map_size_vs_noise, total_maks, br=3, kategoria=1,detekcja=1)
    ttext = model + '\n Nieprawidłowe rozpoznanie twarzy z maską\n jasność = 0'
    fout = path + 'map_size_vs_noise_br3_nieprawidlowe_okreslenie_twarzy_z_maska.png'
    wykres_size_vs_noise(map_size_vs_noise,title_text=ttext,
                      filename=fout)

    #####################################################################################

    ## jasnosc 5

    #########################################################

    map_size_vs_noise = np.zeros(shape=(11, 6))
    size_vs_noise_br_detekcja(map_size_vs_noise, total, br=5, detekcja=0, invers=True)
    ttext = model + '\n Udane wykrycie twarzy \n jasność = +100'
    fout = path + 'map_size_vs_noise_br5_udane_wykrycie_twarzy.png'
    wykres_size_vs_noise(map_size_vs_noise, title_text=ttext,
                      filename=fout)

    #########################################################

    map_size_vs_noise = np.zeros(shape=(11, 6))
    size_vs_noise_br_kategoria_detekcja(map_size_vs_noise, total_bezmaks, br=5, kategoria=0,detekcja=0, invers=True)
    ttext = model + '\n Udane wykrycie twarzy bez maski\n jasność = +100'
    fout = path + 'map_size_vs_noise_br5_udane_wykrycie_twarzy_bez_maski.png'
    wykres_size_vs_noise(map_size_vs_noise,title_text=ttext,
                      filename=fout)

    #########################################################

    map_size_vs_noise = np.zeros(shape=(11, 6))
    size_vs_noise_br_kategoria_detekcja(map_size_vs_noise, total_maks, br=5, kategoria=1,detekcja=0, invers=True)
    ttext = model + '\n Udane wykrycie twarzy z maską\n jasność = +100'
    fout = path + 'map_size_vs_noise_br5_udane_wykrycie_twarzy__z_maska.png'
    wykres_size_vs_noise(map_size_vs_noise,title_text=ttext,
                      filename=fout)

    #####################################################################################

    map_size_vs_noise = np.zeros(shape=(11, 6))
    size_vs_noise_br_kategoria_detekcja(map_size_vs_noise, total_bezmaks, br=5, kategoria=0,detekcja=1)
    ttext = model + '\n Prawidłowe rozpoznanie twarzy bez maski\n jasność = +100'
    fout = path + 'map_size_vs_noise_br5_prawidlowe_okreslenie_twarzy_bez_maski.png'
    wykres_size_vs_noise(map_size_vs_noise,title_text=ttext,
                      filename=fout)

    #####################################################################################

    map_size_vs_noise = np.zeros(shape=(11, 6))
    size_vs_noise_br_kategoria_detekcja(map_size_vs_noise, total_maks, br=5, kategoria=1,detekcja=2)
    ttext = model + '\n Prawidłowe rozpoznanie twarzy z maską\n jasność = +100'
    fout = path + 'map_size_vs_noise_br5_prawidlowe_okreslenie_twarzy_z_maska.png'
    wykres_size_vs_noise(map_size_vs_noise,title_text=ttext,
                      filename=fout)

    #####################################################################################

    map_size_vs_noise = np.zeros(shape=(11, 6))
    size_vs_noise_br_kategoria_detekcja(map_size_vs_noise, total_bezmaks, br=5, kategoria=0,detekcja=2)
    ttext = model + '\n Nieprawidłowe rozpoznanie twarzy bez maski\n jasność = +100'
    fout = path + 'map_size_vs_noise_br5_nieprawidlowe_okreslenie_twarzy_bez_maski.png'
    wykres_size_vs_noise(map_size_vs_noise,title_text=ttext,
                      filename=fout)

    #####################################################################################

    map_size_vs_noise = np.zeros(shape=(11, 6))
    size_vs_noise_br_kategoria_detekcja(map_size_vs_noise, total_maks, br=5, kategoria=1,detekcja=1)
    ttext = model + '\n Nieprawidłowe rozpoznanie twarzy z maską\n jasność = +100'
    fout = path + 'map_size_vs_noise_br5_nieprawidlowe_okreslenie_twarzy_z_maska.png'
    wykres_size_vs_noise(map_size_vs_noise,title_text=ttext,
                      filename=fout)

    #####################################################################################


    ## szum 0

    #########################################################

    map_size_vs_br = np.zeros(shape=(11, 6))
    size_vs_br_noise_detekcja(map_size_vs_br, total, noise=0, detekcja=0, invers=True)
    ttext = model + '\n Udane wykrycie twarzy \n szum = 0'
    fout = path + 'map_size_vs_br_szum0_udane_wykrycie_twarzy.png'
    wykres_size_vs_br(map_size_vs_br, title_text=ttext,
                      filename=fout)

    #########################################################

    map_size_vs_br = np.zeros(shape=(11, 6))
    size_vs_br_noise_kategoria_detekcja(map_size_vs_br, total_bezmaks, noise=0, kategoria=0,detekcja=0, invers=True)
    ttext = model + '\n Udane wykrycie twarzy bez maski\n szum = 0'
    fout = path + 'map_size_vs_br_szum0_udane_wykrycie_twarzy_bez_maski.png'
    wykres_size_vs_br(map_size_vs_br,title_text=ttext,
                      filename=fout)

    #########################################################

    map_size_vs_br = np.zeros(shape=(11, 6))
    size_vs_br_noise_kategoria_detekcja(map_size_vs_br, total_maks, noise=0, kategoria=1,detekcja=0, invers=True)
    ttext = model + '\n Udane wykrycie twarzy z maską\n szum = 0'
    fout = path + 'map_size_vs_br_szum0_udane_wykrycie_twarzy_z_maska.png'
    wykres_size_vs_br(map_size_vs_br,title_text=ttext,
                      filename=fout)

    #####################################################################################

    map_size_vs_br = np.zeros(shape=(11, 6))
    size_vs_br_noise_kategoria_detekcja(map_size_vs_br, total_bezmaks, noise=0, kategoria=0,detekcja=1)
    ttext = model + '\n Prawidłowe rozpoznanie twarzy bez maski\n szum = 0'
    fout = path + 'map_size_vs_br_szum0_prawidlowe_okreslenie_twarzy_bez_maski.png'
    wykres_size_vs_br(map_size_vs_br,title_text=ttext,
                      filename=fout)

    ###########################################################################################

    map_size_vs_br = np.zeros(shape=(11, 6))
    size_vs_br_noise_kategoria_detekcja(map_size_vs_br, total_maks, noise=0, kategoria=1,detekcja=2)
    ttext = model + '\n Prawidłowe rozpoznanie twarzy z maską\n szum = 0'
    fout = path + 'map_size_vs_br_szum0_prawidlowe_okreslenie_twarzy_z_maska.png'
    wykres_size_vs_br(map_size_vs_br,title_text=ttext,
                      filename=fout)

    #####################################################################################

    map_size_vs_br = np.zeros(shape=(11, 6))
    size_vs_br_noise_kategoria_detekcja(map_size_vs_br, total_bezmaks, noise=0, kategoria=0,detekcja=2)
    ttext = model + '\n Nieprawidłowe rozpoznanie twarzy bez maski\n szum = 0'
    fout = path + 'map_size_vs_br_szum0_nieprawidlowe_okreslenie_twarzy_bez_maski.png'
    wykres_size_vs_br(map_size_vs_br,title_text=ttext,
                      filename=fout)

    ###########################################################################################

    map_size_vs_br = np.zeros(shape=(11, 6))
    size_vs_br_noise_kategoria_detekcja(map_size_vs_br, total_maks, noise=0, kategoria=1,detekcja=1)
    ttext = model + '\n Nieprawidłowe rozpoznanie twarzy z maską\n szum = 0'
    fout = path + 'map_size_vs_br_szum0_nieprawidlowe_okreslenie_twarzy_z_maska.png'
    wykres_size_vs_br(map_size_vs_br,title_text=ttext,
                      filename=fout)

    ###########################################################################################

    ## szum 3

    #########################################################

    map_size_vs_br = np.zeros(shape=(11, 6))
    size_vs_br_noise_detekcja(map_size_vs_br, total, noise=3, detekcja=0, invers=True)
    ttext = model + '\n Udane wykrycie twarzy \n szum std = 75'
    fout = path + 'map_size_vs_br_szum3_udane_wykrycie_twarzy.png'
    wykres_size_vs_br(map_size_vs_br, title_text=ttext,
                      filename=fout)

    #########################################################

    map_size_vs_br = np.zeros(shape=(11, 6))
    size_vs_br_noise_kategoria_detekcja(map_size_vs_br, total_bezmaks, noise=3, kategoria=0, detekcja=0, invers=True)
    ttext = model + '\n Udane wykrycie twarzy bez maski\n szum std = 75'
    fout = path + 'map_size_vs_br_szum3_udane_wykrycie_twarzy_bez_maski.png'
    wykres_size_vs_br(map_size_vs_br, title_text=ttext,
                      filename=fout)

    #########################################################

    map_size_vs_br = np.zeros(shape=(11, 6))
    size_vs_br_noise_kategoria_detekcja(map_size_vs_br, total_maks, noise=3, kategoria=1, detekcja=0, invers=True)
    ttext = model + '\n Udane wykrycie twarzy z maską\n szum std = 75'
    fout = path + 'map_size_vs_br_szum3_udane_wykrycie_twarzy_z_maska.png'
    wykres_size_vs_br(map_size_vs_br, title_text=ttext,
                      filename=fout)

    #####################################################################################

    map_size_vs_br = np.zeros(shape=(11, 6))
    size_vs_br_noise_kategoria_detekcja(map_size_vs_br, total_bezmaks, noise=3, kategoria=0, detekcja=1)
    ttext = model + 'Rozmiar vs jasność \n Prawidłowe rozpoznanie twarzy bez maski\n szum std = 75'
    fout = path + 'map_size_vs_br_szum3_prawidlowe_okreslenie_twarzy_bez_maski.png'
    wykres_size_vs_br(map_size_vs_br, title_text=ttext,
                      filename=fout)

    ###########################################################################################

    map_size_vs_br = np.zeros(shape=(11, 6))
    size_vs_br_noise_kategoria_detekcja(map_size_vs_br, total_maks, noise=3, kategoria=1, detekcja=2)
    ttext = model + '\n Prawidłowe rozpoznanie twarzy z maską\n szum std = 75'
    fout = path + 'map_size_vs_br_szum3_prawidlowe_okreslenie_twarzy_z_maska.png'
    wykres_size_vs_br(map_size_vs_br, title_text=ttext,
                      filename=fout)

    #####################################################################################

    map_size_vs_br = np.zeros(shape=(11, 6))
    size_vs_br_noise_kategoria_detekcja(map_size_vs_br, total_bezmaks, noise=3, kategoria=0, detekcja=2)
    ttext = model + '\n Nieprawidłowe rozpoznanie twarzy bez maski\n szum std = 75'
    fout = path + 'map_size_vs_br_szum3_nieprawidlowe_okreslenie_twarzy_bez_maski.png'
    wykres_size_vs_br(map_size_vs_br, title_text=ttext,
                      filename=fout)

    ###########################################################################################

    map_size_vs_br = np.zeros(shape=(11, 6))
    size_vs_br_noise_kategoria_detekcja(map_size_vs_br, total_maks, noise=3, kategoria=1, detekcja=1)
    ttext = model + '\n Nieprawidłowe rozpoznanie twarzy z maską\n szum std = 75'
    fout = path + 'map_size_vs_br_szum3_nieprawidlowe_okreslenie_twarzy_z_maska.png'
    wykres_size_vs_br(map_size_vs_br, title_text=ttext,
                      filename=fout)

    ###########################################################################################

    ## szum 5

    #########################################################

    map_size_vs_br = np.zeros(shape=(11, 6))
    size_vs_br_noise_detekcja(map_size_vs_br, total, noise=5, detekcja=0, invers=True)
    ttext = model + '\n Udane wykrycie twarzy \n szum std = 125'
    fout = path + 'map_size_vs_br_szum5_udane_wykrycie_twarzy.png'
    wykres_size_vs_br(map_size_vs_br, title_text=ttext,
                      filename=fout)

    #########################################################

    map_size_vs_br = np.zeros(shape=(11, 6))
    size_vs_br_noise_kategoria_detekcja(map_size_vs_br, total_bezmaks, noise=5, kategoria=0, detekcja=0, invers=True)
    ttext = model + '\n Udane wykrycie twarzy bez maski\n szum std = 125'
    fout = path + 'map_size_vs_br_szum5_udane_wykrycie_twarzy_bez_maski.png'
    wykres_size_vs_br(map_size_vs_br, title_text=ttext,
                      filename=fout)

    #########################################################

    map_size_vs_br = np.zeros(shape=(11, 6))
    size_vs_br_noise_kategoria_detekcja(map_size_vs_br, total_maks, noise=5, kategoria=1, detekcja=0, invers=True)
    ttext = model + '\n Udane wykrycie twarzy z maską\n szum std = 125'
    fout = path + 'map_size_vs_br_szum5_udane_wykrycie_twarzy_z_maska.png'
    wykres_size_vs_br(map_size_vs_br, title_text=ttext,
                      filename=fout)

    #####################################################################################

    map_size_vs_br = np.zeros(shape=(11, 6))
    size_vs_br_noise_kategoria_detekcja(map_size_vs_br, total_bezmaks, noise=5, kategoria=0, detekcja=1)
    ttext = model + '\n Prawidłowe rozpoznanie twarzy bez maski\n szum std = 125'
    fout = path + 'map_size_vs_br_szum5_prawidlowe_okreslenie_twarzy_bez_maski.png'
    wykres_size_vs_br(map_size_vs_br, title_text=ttext,
                      filename=fout)

    ###########################################################################################

    map_size_vs_br = np.zeros(shape=(11, 6))
    size_vs_br_noise_kategoria_detekcja(map_size_vs_br, total_maks, noise=5, kategoria=1, detekcja=2)
    ttext = model + '\n Prawidłowe rozpoznanie twarzy z maską\n szum std = 125'
    fout = path + 'map_size_vs_br_szum5_prawidlowe_okreslenie_twarzy_z_maska.png'
    wykres_size_vs_br(map_size_vs_br, title_text=ttext,
                      filename=fout)

    #####################################################################################

    map_size_vs_br = np.zeros(shape=(11, 6))
    size_vs_br_noise_kategoria_detekcja(map_size_vs_br, total_bezmaks, noise=5, kategoria=0, detekcja=2)
    ttext = model + '\n Nieprawidłowe rozpoznanie twarzy bez maski\n szum std = 125'
    fout = path + 'map_size_vs_br_szum5_nieprawidlowe_okreslenie_twarzy_bez_maski.png'
    wykres_size_vs_br(map_size_vs_br, title_text=ttext,
                      filename=fout)

    ###########################################################################################

    map_size_vs_br = np.zeros(shape=(11, 6))
    size_vs_br_noise_kategoria_detekcja(map_size_vs_br, total_maks, noise=5, kategoria=1, detekcja=1)
    ttext = model + '\n Nieprawidłowe rozpoznanie twarzy z maską\n szum std = 125'
    fout = path + 'map_size_vs_br_szum5_nieprawidlowe_okreslenie_twarzy_z_maska.png'
    wykres_size_vs_br(map_size_vs_br,
                      title_text=ttext,
                      filename=fout)

    ###########################################################################################



if __name__ == '__main__':

    # lista do analizy
    # wyniki testowania
    fnames = ('test_results_v1.txt','test_results_v2.txt','test_results_v3.txt','test_results_TFL.txt')
    # nazwy modeli do wyswietlenia w tytulach
    models = ('InceptionV3','InceptionV3_cvzone','MobileNetV2_cvzone','SSD_MobileNetV2_FPNLite')
    # katalogi wyjsciowe dla podsumowan i wykresow
    paths = ('wykresy v1/','wykresy v2/','wykresy v3/','wykresy TFL/')

    # petla po wszystkich wejsciowych
    for g in range(4):
        analizuj_wynik(fnames[g],models[g],paths[g])
