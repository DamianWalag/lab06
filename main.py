from PIL import Image, ImageDraw
from ast import arg
import argparse
import numpy as np
import scipy.signal as sp
from rich.progress import track
import numba 
import time
# obsluga argumentow
parser = argparse.ArgumentParser()

parser.add_argument('-n', '--bok_siatki', default=1000)
parser.add_argument('-j', '--wartosc_J', default=10)
parser.add_argument('-b', '--wartosc_beta', default=0)
parser.add_argument('-B', '--wartosc_B', default=1)
parser.add_argument('-N', '--liczba_krokow', default=20)
parser.add_argument('-g', '--gestosc_dodatnich_spinow', default=0.5)
parser.add_argument('-p', '--prefix_nazw_rysonkow', default=[])
parser.add_argument('-a', '--nazwa_pliku_z_animacja_bez_rozszerzenia', default=[])
parser.add_argument('-m', '--nazwa_pliku_z_magnetyzacja', default=[])

args = parser.parse_args()


def generator(rozmiar, N, spiny):
        index = 0
        magnetyzacja = 0

        while index < N:
            magnetyzacja = (np.sum(spiny)/(rozmiar*rozmiar))
            index += 1
            yield magnetyzacja, index
def obrazek(rozmiar, spiny):
        img = Image.new('RGB', (2000, 2000), (0, 0, 0))
        draw = ImageDraw.Draw(img)
        krok = 2000/rozmiar
        for j in range(rozmiar):
            for i in range(rozmiar):
                if spiny[i, j] == 1:
                    draw.rectangle((0+krok*j, 0+krok*i, 2000/rozmiar + krok*j, 2000/rozmiar+krok*i), (0, 255, 0))

        return img
#petla + numba jest szybsze niz convolve2d o jakies 2-3 razy 
@numba.njit()
def hamiltonian_numba(spiny, J, B):
    # H = 0
    # kernel = (np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]))
    # wrap załatwia periodyczne warunki brzegowe B)
    # convo = sp.convolve2d(spiny, kernel, mode='same', boundary='wrap')
    convo = np.zeros_like(spiny)
    height, width = spiny.shape
    for i in range(height):
        for j in range(width):
            convo[i, j] = (spiny[(i - 1) % height, j] + spiny[(i + 1) % height, j] + spiny[i, (j - 1) % width] + spiny[i, (j + 1) % width]) 
    # print(convo)
    # print(convo1)
    # suma = np.sum(spiny * convo)
    # dziele przez 2 bo jest liczone podwojnie
    # suma = suma/2
    # H = - J * suma - B * np.sum(spiny)
    return convo
#bez numby
def hamiltonian(spiny, J, B):
    kernel = (np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]))
    convo = sp.convolve2d(spiny, kernel, mode='same', boundary='wrap')
    # convo = np.zeros_like(spiny)
    # height, width = spiny.shape
    # for i in range(height):
    #     for j in range(width):
    #         convo[i, j] = (spiny[(i - 1) % height, j] + spiny[(i + 1) % height, j] + spiny[i, (j - 1) % width] + spiny[i, (j + 1) % width]) 

    return convo
@numba.njit()
def krok_numba(convo, rozmiar, spiny,B,beta,J):
        c = np.random.randint(0, rozmiar, size=(rozmiar**2))
        d = np.random.randint(0, rozmiar, size=(rozmiar**2))

        for i, j in zip(c, d):
            # licze od razu roznice a nie caly Hamiltionian
            zmiana_energii = (-2) * (-J * spiny[i, j] * convo[i, j] - B * spiny[i, j])
            # warunki na zaakceptowanie zmiany spinu
            if zmiana_energii <= 0:
                spiny[i, j] = -spiny[i, j]
            elif np.random.uniform(0.0, 1.0) < np.exp(-beta*zmiana_energii):
                spiny[i, j] = -spiny[i, j]
#bez numby
def krok(convo, rozmiar, spiny,B,beta,J):
        c = np.random.randint(0, rozmiar, size=(rozmiar**2))
        d = np.random.randint(0, rozmiar, size=(rozmiar**2))

        for i, j in zip(c, d):
            # licze od razu roznice a nie caly Hamiltionian
            zmiana_energii = (-2) * (-J * spiny[i, j] * convo[i, j] - B * spiny[i, j])
            # warunki na zaakceptowanie zmiany spinu
            if zmiana_energii <= 0:
                spiny[i, j] = -spiny[i, j]
            elif np.random.uniform(0.0, 1.0) < np.exp(-beta*zmiana_energii):
                spiny[i, j] = -spiny[i, j]

def main(fast = True):
        rozmiar = int(args.bok_siatki)
        J = float(args.wartosc_J)
        beta = float(args.wartosc_beta)
        B = float(args.wartosc_B)
        N = int(args.liczba_krokow)
        prefix = args.prefix_nazw_rysonkow
        gestosc = float(args.gestosc_dodatnich_spinow)
        animacja = args.nazwa_pliku_z_animacja_bez_rozszerzenia
        plik_m = args.nazwa_pliku_z_magnetyzacja

        #init
        spiny = np.concatenate((-np.ones(int(np.floor(rozmiar**2 * (1-gestosc)))), np.ones(int(np.ceil(rozmiar**2 * gestosc)))))
        np.random.shuffle(spiny)
        spiny = spiny.reshape(rozmiar, rozmiar)
        
        images = []
        if len(args.nazwa_pliku_z_magnetyzacja) != 0:
            outfile = open(f"{args.nazwa_pliku_z_magnetyzacja}.txt", 'w')
        
        rys_time = 0
        #zaczynam liczyc czas
        start = time.time()
        for m, index in track(generator(rozmiar, N, spiny), total=N):
           
            if fast == True:
                convo = hamiltonian_numba(spiny, J, B)
            if fast == False:
                convo = hamiltonian(spiny, J, B)
           
            #jesli podano prefix to zapisujemy png
            start_rys = time.time()
            if prefix != []:   
                img = obrazek(rozmiar, spiny)
                img.save(f'{prefix}{index}.png')
            #jesli podano gif to zapisujemy
            if animacja != []:
                #jesli jednoczesnie nie podano prefixu to trzeba te rysunki i tak wygenerowac
                if prefix == []:
                    img = obrazek(rozmiar, spiny)
                images.append(img)
            #jesli podano plik na magnetyzacje to zapisujemy
            if len(args.nazwa_pliku_z_magnetyzacja) != 0:
                outfile.write(str(m) + '\n')
            stop_rys = time.time()
            rys_time += stop_rys-start_rys
            
            if fast == True:
                krok_numba(convo, rozmiar, spiny,B,beta,J)
            if fast == False:
                krok(convo, rozmiar, spiny,B,beta,J)
            
            
        # koncze liczyc czas (odejmuje tez czas potrzebny na obrazki)
        stop = time.time()
        if fast == True:
            print("czas wykonania z numbą:", stop - start - rys_time)
        if fast == False:
            print("czas wykonania bez numby:", stop - start - rys_time)
        if animacja != []:
           # img = Image.new('RGB', (2000,2000), (0,0,0))
            images[0].save(f'{animacja}.gif', save_all=True, append_images = images[1:], duration = 300, loop=0)




# odpalam dwie wersje funkji main() tzn taka z numbą i bez.
# czas wykonania to czas obliczen (wykonania funkcji hamiltonian() i krok())
# dla 20 krokow i siatki o boku 1000 przykladowy wynik bez rysowania to:
# numba t = 2.03s
# bez numby t = 73.98s 
# bez numby ale z convolve2d w funkcji hamiltonian() t = 58.41s
main(fast = True)
main(fast = False)