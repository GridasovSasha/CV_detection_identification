# Вывод изображения с камеры
import cv2

capture = cv2.VideoCapture(0) # 0 - порядковый номер видео адаптора

while True:
    red, img = capture.read()

    cv2.imshow('From camera', img)

    # обработка выходы
    end_key = cv2.waitKey(30) & 0xFF
    if end_key == 27:
        break

capture.release()
cv2.destroyAllWindows()
