# Программа для обнаружения лица и распознавания личности на основе данных trainner.yml
import cv2  # Для обработки изображений (обнаружение лиц, изменение размера, преобразование)
import numpy as np  # Для преобразования изображений в числовые массивы
import os  # Для работы с директориями (пути к файлам и папкам)
from PIL import Image  # Библиотека Pillow для работы с изображениями

labels = ['Masha']  # Список имен, индекс каждого имени соответствует ID лица
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')  # Загрузка каскадного классификатора Хаара для обнаружения лиц
recognizer = cv2.face.LBPHFaceRecognizer_create()  # Создание LBPH (Local Binary Patterns Histograms) распознавателя лиц
recognizer.read('trainer/trainer.yml')  # Загрузка обученных данных из YAML-файла

cap = cv2.VideoCapture(0)  # Получение видеопотока с камеры

while(True):  # Основной бесконечный цикл обработки видео
    ret, img = cap.read()  # Разбиение видео на кадры
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Преобразование кадра видео в оттенки серого
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)  # Обнаружение лиц на кадре
    for (x, y, w, h) in faces:  # Цикл по каждому обнаруженному лицу
        roi_gray = gray[y:y+h, x:x+w]  # Выделение области лица и преобразование ее в оттенки серого

        id_, conf = recognizer.predict(roi_gray)  # Распознавание лица

        if conf >= 80:  # Проверка, если уверенность распознавания больше 80%
            font = cv2.FONT_HERSHEY_SIMPLEX  # Стиль шрифта для имени
            name = labels[id_]  # Получение имени из списка по ID лица
            cv2.putText(img, name, (x, y), font, 1, (0, 0, 255), 2)  # Отображение имени на кадре

        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Отображение прямоугольника вокруг лица

    cv2.imshow('Preview', img)  # Отображение видео

    # обработка выходы
    end_key = cv2.waitKey(30) & 0xFF
    if end_key == 27:
        break

# Когда все сделано, освободить захват
cap.release()  # Освобождение ресурсов камеры
cv2.destroyAllWindows()  # Закрытие всех окон
