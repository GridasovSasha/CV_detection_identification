import cv2  # Для обработки изображений (обнаружение лиц, изменение размера, преобразование)

# добавим функцию для размытия лица
def blure_face(img):
    (h, w) = img.shape[:2]
    hB = int(h / 3.0)
    wB = int(w / 3.0)
    if hB%2 == 0:
        hB +=1
    if wB%2 == 0:
        wB +=1
    return cv2.GaussianBlur(img, (hB,wB), 0)


cap = cv2.VideoCapture(0)  # Получение видеопотока с камеры

# haarcascade_smile.xml
# haarcascade_eye.xml и прочие
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') # Загрузка каскадного классификатора Хаара для обнаружения лиц

while(True):  # Основной бесконечный цикл обработки видео
    ret, img = cap.read()  # Разбиение видео на кадры
    faces = face_cascade.detectMultiScale(img, scaleFactor=2, minNeighbors=5) #получаем массив лиц

    for (x, y, width, height) in faces:
        #cv2.rectangle(img, (x,y), (x+width, y+height), (100, 0 , 255), 2) # Отрисовка прямоугольника вокруг лица
        img[y:y+height, x:x+width] = blure_face(img[y:y+height, x:x+width])

    cv2.imshow('All face blure', img) # Отображение видео c блюром

    # Завершенеи цикла по нажатию клавиши Esc
    k = cv2.waitKey(30) & 0xFF
    if k == 27:
        break


# Когда все сделано, освободить захват
cap.release()  # Освобождение ресурсов камеры
cv2.destroyAllWindows()  # Закрытие всех окон
