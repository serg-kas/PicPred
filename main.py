'''
Тестовая программа для демонстрации модуля предикта продолжения графика по картинке
'''

from os import listdir
from random import randint

import warnings
warnings.filterwarnings("ignore")

# Модуль для обработки картинки
import picpred
model_PATH = 'modelGA.h5'
xScaler_PATH = 'xScaler.bin'
yScaler_PATH = 'yScaler.bin'

img_PATH = 'images'          # папка откуда берем картинки для обработки
out_PATH = 'images_out'      # папка куда помещаем результат

if __name__ == '__main__':
    # ЗАГРУЗИМ КАРТИНКУ для обработки
    # В папке img_PATH должны быть только картинки допустимых форматов
    img_files = sorted(listdir(img_PATH))
    out_files = sorted(listdir(out_PATH))
    files = []     # список файлов для обработки
    for f in img_files:
        if not (('out_'+f) in out_files):
            files.append(f)
    if len(files)==0:        # если все файлы уже были обработаны
        files = img_files    # то будем брать любой из имеющихся
    # если список не пуст выбираем случайный файл
    if len(files) > 0:
        idx = randint(0, len(files) - 1)
        img_FILE = img_PATH + '/' + files[idx]
        out_FILE = out_PATH + '/' + 'out_' + files[idx]

        out_FILE_PATH = picpred.pic_predict(img_FILE, model_PATH, xScaler_PATH, yScaler_PATH, out_FILE)
        print('Путь к готовой картинке:', out_FILE_PATH)

        # Если хотим вывести готовую картинку
        import cv2
        out_image = cv2.imread(out_FILE_PATH)
        out_image = cv2.resize(out_image, (900, 500), interpolation=cv2.INTER_AREA)
        cv2.imshow('out_image', out_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        print('В папке images пусто')


