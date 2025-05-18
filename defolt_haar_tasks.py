import cv2  # Для обработки изображений (обнаружение лиц, изменение размера, преобразование)

cap = cv2.VideoCapture(0)  # Получение видеопотока с камеры

# haarcascade_smile.xml
# haarcascade_eye.xml и прочие
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') # Загрузка каскадного классификатора Хаара для обнаружения лиц

while(True):  # Основной бесконечный цикл обработки видео
    ret, img = cap.read()  # Разбиение видео на кадры
    faces = face_cascade.detectMultiScale(img, scaleFactor=1.5, minNeighbors=5) #получаем массив лиц

    for (x, y, width, height) in faces:
        cv2.rectangle(img, (x,y), (x+width, y+height), (100, 0 , 255), 2) # Отрисовка прямоугольника вокруг лица

    cv2.imshow('All face', img) # Отображение видео

    # Завершенеи цикла по нажатию клавиши Esc
    k = cv2.waitKey(30) & 0xFF
    if k == 27:
        break


# Когда все сделано, освободить захват
cap.release()  # Освобождение ресурсов камеры
cv2.destroyAllWindows()  # Закрытие всех окон
