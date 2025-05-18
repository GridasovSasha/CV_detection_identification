import cv2 # подключаем библиотеку машинного зрения
import os # библиотека для вызова системных функций
import numpy as np
from PIL import Image

path = os.path.dirname(os.path.abspath(__file__)) # получаем абсолютный путь к директории
dataPath = 'dataSet\Masha_grey'

# LBPH — это алгоритм распознавания лиц, основанный на локальных бинарных шаблонах (Local Binary Patterns)
recognizer = cv2.face.LBPHFaceRecognizer_create()
# Загрузка каскадного классификатора Хаара для обнаружения лиц
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# получаем картинки и подписи из датасета
def get_images_and_labels(datapath):
    # получаем путь к картинкам
    image_paths = [os.path.join(datapath, f) for f in os.listdir(datapath)]

    # списки картинок и подписей на старте пустые
    images = []
    labels = []
    label_ids = {}  # Словарь для сопоставления строковых меток с целочисленными идентификаторами
    current_id = 0

    # перебираем все картинки в датасете
    for image_path in image_paths:
        # Извлеките метку из имени файла (первые 5 символов)
        label_str = os.path.split(image_path)[1][:5]

        # Преобразование строковой метки в целочисленный ID
        if label_str not in label_ids:
            label_ids[label_str] = current_id
            current_id += 1
        label = label_ids[label_str]

        image_pil = Image.open(image_path).convert('L') # читаем картинку и сразу переводим в ч/б
        image = np.array(image_pil, 'uint8') # переводим картинку в numpy-массив
        faces = faceCascade.detectMultiScale(image) # определяем лицо на картинке

        # если лицо найдено
        for (x, y, width, height) in faces:
            images.append(image[y: y + height, x: x + width]) # добавляем его к списку картинок
            labels.append(label) # добавляем id пользователя в список подписей
            cv2.imshow("Adding faces to training set...", image[y: y + height, x: x + width]) # выводим текущую картинку на экран
            cv2.waitKey(100) # добавляем задержку

    # возвращаем список картинок и подписей
    return images, np.array(labels, dtype=np.int32), label_ids  # Return label mapping for reference

# получаем список картинок и подписей
images, labels, label_mapping = get_images_and_labels(dataPath)

# обучаем модель распознавания на наших картинках и учим сопоставлять её лица и подписи к ним
recognizer.train(images, labels)

# сохраняем модель
recognizer.save(path+r'/trainer/trainer.yml')

# удаляем из памяти все созданные окнаы
cv2.destroyAllWindows()
