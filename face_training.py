import cv2
import numpy as np
from PIL import Image
import os


def checkDataset(directory="dataset"):
    if os.path.exists(directory):
        if len(os.listdir(directory)) > 0:
            return True

    return False


def getImagesAndLabels(path):

    imagePaths = [
        os.path.join(path, f)
        for f in os.listdir(path)
    ]

    faceSamples = []
    ids = []

    detector = cv2.CascadeClassifier(
        "haarcascade_frontalface_default.xml"
    )

    for imagePath in imagePaths:

        PIL_img = Image.open(
            imagePath
        ).convert("L")

        img_numpy = np.array(
            PIL_img,
            "uint8"
        )


        filename = os.path.split(
            imagePath
        )[1]

        id = int(
            filename.split(".")[1]
        )

        faces = detector.detectMultiScale(
            img_numpy
        )

        for (x, y, w, h) in faces:

            faceSamples.append(
                img_numpy[y:y+h, x:x+w]
            )

            ids.append(id)

    return faceSamples, ids

if not checkDataset():

    print("Dataset tidak ditemukan.")

else:

    print("Training faces...")

    recognizer = cv2.face.LBPHFaceRecognizer.create()

    faces, ids = getImagesAndLabels(
        "dataset"
    )

    recognizer.train(
        faces,
        np.array(ids)
    )

    recognizer.write(
        "face-model.yml"
    )

    print("\nTraining selesai!")
    print("Model berhasil disimpan:")
    print("face-model.yml")