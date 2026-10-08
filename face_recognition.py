import cv2

recognizer = cv2.face.LBPHFaceRecognizer.create()

recognizer.read(
    "face-model.yml"
)

faceCascade = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

names = {
    1: "Sandhika",
    2: "Dhana",
    3: "Yossi"
}

colors = {
    1: (255, 0, 0),   # Biru
    2: (0, 255, 0),   # Hijau
    3: (0, 0, 255)    # Merah
}

RECOGNITION_THRESHOLD = 70

font = cv2.FONT_HERSHEY_COMPLEX

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Webcam tidak dapat dibuka.")
    exit()

while True:

    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    for (x, y, w, h) in faces:

        predicted_id, confidence = recognizer.predict(
            gray[y:y+h, x:x+w]
        )

        if predicted_id in names and confidence < RECOGNITION_THRESHOLD:

            name = names[predicted_id]
            color = colors[predicted_id]

            similarity = max(
                0,
                min(
                    100,
                    round(100 - confidence)
                )
            )

        else:

            name = "Unknown"
            color = (255, 255, 255)

            similarity = 0

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            color,
            2
        )

        cv2.putText(
            frame,
            name,
            (x + 5, y - 25),
            font,
            1,
            color,
            2
        )

        cv2.putText(
            frame,
            f"Similarity: {similarity}%",
            (x + 5, y + h + 25),
            font,
            0.7,
            color,
            2
        )

    cv2.imshow(
        "Face Recognition",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()