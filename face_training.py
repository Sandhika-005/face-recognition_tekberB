import cv2
import numpy as np
import os

def checkDataset(directory="dataset/"):
    if os.path.exists(directory) and len(os.listdir(directory)) != 0:
        return True

    return False


def organizeDataset(path="dataset/"):

    imagePaths = [
        os.path.join(path, p)
        for p in os.listdir(path)
    ]

    faces = []
    ids = np.array([], dtype=int)

    for imagePath in imagePaths:

        img = cv2.imread(
            imagePath,
            cv2.IMREAD_GRAYSCALE
        )

        filename = os.path.basename(imagePath)

        # Format:
        # Sandhika_1_1.jpg
        # Dhana_2_1.jpg
        # Yossi_3_1.jpg

        person_id = int(
            filename.split("_")[1]
        )

        faces.append(img)
        ids = np.append(ids, person_id)

    return faces, ids


if not checkDataset():
    print("Dataset not found")

else:

    recognizer = cv2.face.LBPHFaceRecognizer.create()

    faceCascade = cv2.CascadeClassifier(
        "haarcascade_frontalface_default.xml"
    )

    print("Training faces...")

    faces, ids = organizeDataset()

    recognizer.train(faces, ids)

    print("Training finished!")

    recognizer.write("face-model.yml")

    print("Model saved as 'face-model.yml'")