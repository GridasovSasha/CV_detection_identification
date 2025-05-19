import cv2 # подключаем библиотеку компьютерного зрения
import os # библиотека для вызова системных функций


path = os.path.dirname(os.path.abspath(__file__)) # получаем путь к этому скрипту
recognizer = cv2.face.LBPHFaceRecognizer_create() # создаём новый распознаватель лиц
recognizer.read(path+r'/trainer/trainer.yml') # добавляем в него обученную модель
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml") # Загрузка каскадного классификатора Хаара для обнаружения лиц

cap = cv2.VideoCapture(0) # Получение видеопотока с камеры
font = cv2.FONT_HERSHEY_SIMPLEX # настраиваем шрифт для вывода подписей

# Добавляем словарь для соответствия ID и имен(в нашем случае одно имя)

while True:
    ret, img = cap.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100)) #получаем массив лиц

    # перебираем все найденные лица
    for(x, y, width, height) in faces:

        nbr_predicted, confidence = recognizer.predict(gray[y:y+height,x:x+width]) # получаем имя пользователя
        cv2.rectangle(img, (x-20,y-20), (x+width+20, y+height+20), (100, 0 , 255), 2) # Отрисовка прямоугольника вокруг лица

        # Кастомизация имени
        if nbr_predicted == 0:
            name = 'Sasha'

        # Добавляем информацию об уверенности
        text = f"{name} ({confidence:.1f})"
        if confidence < 50:
            cv2.putText(img, text, (x-40,y-30),font, 1, (0,255,0), 2)
        cv2.imshow('Face recognition',img) # выводим окно с изображением с камеры

    # Завершенеи цикла по нажатию клавиши Esc7
    k = cv2.waitKey(30) & 0xFF
    if k == 27:
        break

# Когда все сделано, освободить захват
cap.release() # Освобождение ресурсов камеры
cv2.destroyAllWindows()  # Закрытие всех окон
