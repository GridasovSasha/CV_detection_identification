import cv2
import os
import numpy as np
from PIL import Image

path = os.path.dirname(os.path.abspath(__file__))
dataPath = 'dataSet'

recognizer = cv2.face.LBPHFaceRecognizer_create()
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def get_images_and_labels(datapath):
    image_paths = [os.path.join(datapath, f) for f in os.listdir(datapath)]

    images = []
    labels = []
    label_ids = {}  # Dictionary to map string labels to integer IDs
    current_id = 0

    for image_path in image_paths:
        # Extract label from filename (first 5 characters)
        label_str = os.path.split(image_path)[1][:5]

        # Convert string label to integer ID
        if label_str not in label_ids:
            label_ids[label_str] = current_id
            current_id += 1
        label = label_ids[label_str]

        image_pil = Image.open(image_path).convert('L')
        image = np.array(image_pil, 'uint8')
        faces = faceCascade.detectMultiScale(image)

        for (x, y, w, h) in faces:
            images.append(image[y: y + h, x: x + w])
            labels.append(label)
            cv2.imshow("Adding faces to training set...", image[y: y + h, x: x + w])
            cv2.waitKey(100)

    return images, np.array(labels, dtype=np.int32), label_ids  # Return label mapping for reference

images, labels, label_mapping = get_images_and_labels(dataPath)

# Train the recognizer
recognizer.train(images, labels)

# Save the trained model
recognizer.save('D:/DS_2025/CV_detection_identification/trainer/trainer.yml')

# Save label mapping for future reference
import pickle
with open('D:/DS_2025/CV_detection_identification/trainer/label_mapping.pkl', 'wb') as f:
    pickle.dump(label_mapping, f)

cv2.destroyAllWindows()
