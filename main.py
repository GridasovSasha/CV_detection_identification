# необходимый импорт
import cv2 # подключаем библиотеку машинного зрения
import os # библиотека для вызова системных функций

# Путь к каскаду для распознавания лиц
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# счётчик изображений
i=0
# расстояния от распознанного лица до рамки
offset=50
# запрашиваем номер пользователя
name=input('Введите номер пользователя: ')
# получаем доступ к камере
video=cv2.VideoCapture(0) # 0 - порядковый номер видео адаптора

while True:
    red, img = video.read()

    cv2.imshow('From camera', img)

    # обработка выходы
    end_key = cv2.waitKey(30) & 0xFF
    if end_key == 27:
        break

video.release()
cv2.destroyAllWindows()
