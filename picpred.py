# На вход получает пути к файлам:
# img_FILE - путь к исходной картинке
# model_PATH - модель
# xScaler_PATH - нормировщик MinMaxScaler для x
# yScaler_PATH - нормировщик MinMaxScaler для y
# out_FILE - путь куда записать готовую картинку
# Возвращает:
# out_FILE - путь к готовой картинке
def pic_predict(img_FILE, model_PATH, xScaler_PATH, yScaler_PATH, out_FILE):

    import numpy as np
    import cv2
    # import pandas as pd
    from tensorflow.keras.models import load_model
    from joblib import load

    # Параметры
    N = 25         # сколько отсчетов обрабатывает модель для предсказания следующего шага
    N_pred = 5     # на сколько шагов вперед предсказываем (последовательными предиктами)

    DEBUG = False  # режим отладки
    if DEBUG:
        print('Режим отладки включен')

    # Функция получения истории котировок
    # возвращает датасет за заданное число дней
    def stocks_history(N_days=100):
        # Источник котировок Yahoo finance
        import yfinance as yf
        from datetime import date, timedelta
        start_day = date.today() - timedelta(days=N_days)
        # Загружаем данные
        data = yf.download(['GMKN.ME'], start=str(start_day), interval='1h')
        # Выкинем ненужные столбцы
        del data['Adj Close']
        del data['Open']
        del data['High']
        del data['Low']
        del data['Volume']
        # Датасет возвращаем
        return data

    # Загружаем и НАЧИНАЕМ ОБРАБОТКУ картинки
    image = cv2.imread(img_FILE)
    # Переводим ее в ч/б цвет
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if DEBUG:
        print('Загружена картинка:', image.shape)
    # удалим СТОЛБЦЫ где нет ЧЕРНОГО цвета
    cropped_image = []
    for col in range(image.shape[1]):
        curr_column = image[:, col]
        if curr_column.min() < 64:
            cropped_image.append(curr_column)
    cropped_image = np.array(cropped_image)
    image = cropped_image.transpose()
    if DEBUG:
        print('Обрезана картинка:', image.shape)
        # cv2.imshow('cropped image', image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    # Получаем исторические данные по котировкам за 100 дней
    data = stocks_history(100)
    if DEBUG:
        print('Загрузили датасет', type(data), data.shape)
    # Диапазон котировок на всем датасете
    A = np.array(list(data['Close'])).min()
    B = np.array(list(data['Close'])).max()
    if DEBUG:
        print('Котировки минимальная {0} ,максимальная {1} на всем датасете'.format(A, B))
    # Размеры для ресайза распарсиваемого графика
    H = int(B - A)
    W = int(image.shape[1] * H / image.shape[0])
    if DEBUG:
        print('Высота для ресайза {0} ,ширина для ресайза, {1}'.format(H, W))
    # Ресайз картинки
    img_parced = cv2.resize(image, (W, H), interpolation=cv2.INTER_LINEAR)
    if DEBUG:
        print('Картинка после ресайза', img_parced.shape)
        # cv2.imshow('resized image', img_parced)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    # Датасет для распарсенных данных
    data_parced = data[-N:].copy()
    if DEBUG:
        print('Новый датасет для заполнения:', data_parced.shape)
    # Диапазон котировок на N отсчетах
    MIN = np.array(list(data_parced['Close'])).min()
    MAX = np.array(list(data_parced['Close'])).max()
    if DEBUG:
        print('Котировки минимальная {0} ,максимальная {1} на диапазоне {2} отсчетов'.format(MIN, MAX, N))
    #
    start = img_parced.shape[1] % N - 1
    step = img_parced.shape[1] // N
    if DEBUG:
        print('Начало, шаг и конец цикла по графику:', start, step, start + step * N)
    # Цикл по графику (заполняем датасет)
    for i in range(N):
        data_parced['Close'].iloc[i] = MAX - img_parced[:, start + step * (i + 1)].argmin()
        if DEBUG:
            print('i =', i, '; start+step*(i+1) =', start + step * (i+1), '; Close =', data_parced['Close'].iloc[i])

    # Сохраним столбец Close в список для итогового графика
    Close_list_out = list(data_parced['Close'])
    if DEBUG:
        print('Список Close_list_out до предикта:', len(Close_list_out), Close_list_out)

    # Загружаем модель
    model = load_model(model_PATH)
    # Загружаем нормировщики
    xScaler = load(xScaler_PATH)
    yScaler = load(yScaler_PATH)

    # Делаем предикт N_pred раз
    for pred in range(N_pred):
        # Нормируем
        xTest = xScaler.transform(np.array(data_parced))

        xVal = [[xTest]]
        xVal = np.array(xVal)

        # Делаем предикт и возвращаем масштаб предиктнутых данных
        predVal = yScaler.inverse_transform(model.predict(xVal[0]))
        predVal = round(predVal[0][0], 1)

        # Добавляем предикт в список для графика
        Close_list_out.append(predVal)
        if DEBUG:
            print('pred=', pred, 'predVal=', predVal)
        # Добавляем предикт к распарсенным данным
        new_row = {'Close': predVal}
        data_parced = data_parced.append(new_row, ignore_index=True)
        # Берем последние N отсчетов
        data_parced = data_parced[-N:]

    if DEBUG:
        print('Список Close_list_out после предикта', len(Close_list_out), Close_list_out)

    # Сохраняем график с предиктом
    import matplotlib.pyplot as plt
    plt.figure(figsize=(18, 10))
    plt.scatter(range(1, N+1), Close_list_out[0:N],
                label='Распарсенный график')
    plt.scatter(range(N+1, N+N_pred+1), Close_list_out[N:N+N_pred],
                label='Прогноз')
    plt.xlabel('Время')
    plt.ylabel('Значение Close')
    plt.legend()
    plt.suptitle(out_FILE)
    plt.savefig(out_FILE)
    # plt.show()

    return out_FILE


