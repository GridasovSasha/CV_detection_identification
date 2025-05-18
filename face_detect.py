import cv2 # подключаем библиотеку компьютерного зрения
import os # библиотека для вызова системных функций


path = os.path.dirname(os.path.abspath(__file__)) # получаем путь к этому скрипту
recognizer = cv2.face.LBPHFaceRecognizer_create() # создаём новый распознаватель лиц
recognizer.read(path+r'/trainer/trainer.yml') # добавляем в него обученную модель
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml") # Загрузка каскадного классификатора Хаара для обнаружения лиц

cap = cv2.VideoCapture(0) # Получение видеопотока с камеры
font = cv2.FONT_HERSHEY_SIMPLEX # настраиваем шрифт для вывода подписей

# Добавляем словарь для соответствия ID и имен(в нашем случае одно имя)
label_mapping = {0: 'masha'}  # ID 0 соответствует имени 'masha'

while True:

    ret, img = cap.read()
    if not ret:
        break

    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5) #получаем массив лиц

    # перебираем все найденные лица
    for(x, y, width, height) in faces:

        nbr_predicted, confidence = recognizer.predict(gray[y-30:y+height+30,x-30:x+width+30]) # получаем имя пользователя
        cv2.rectangle(img, (x-30,y-30), (x+width+30, y+height+30), (100, 0 , 255), 2) # Отрисовка прямоугольника вокруг лица

        # Получаем имя из label_mapping
        name = label_mapping.get(nbr_predicted, f"Unknown ({nbr_predicted})")

        # Кастомизация имени
        if name == 'masha':
            name = 'Masha <3'

        # Добавляем информацию об уверенности
        text = f"{name} ({confidence:.1f})"
        if confidence > 75:
            cv2.putText(img, text, (x, y+10), font, 0.8, (0, 0, 255), 2)
        cv2.imshow('Face recognition',img) # выводим окно с изображением с камеры

    # Завершенеи цикла по нажатию клавиши Esc
    k = cv2.waitKey(30) & 0xFF
    if k == 27:
        break

# Когда все сделано, освободить захват
cap.release() # Освобождение ресурсов камеры
cv2.destroyAllWindows()  # Закрытие всех окон
