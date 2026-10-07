import cv2

recognizer = cv2.face.LBPHFaceRecognizer.create()

recognizer.read("face-model.yml")

faceCascade = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

font = cv2.FONT_HERSHEY_COMPLEX

names = {
    1: "Sandhika",
    2: "Dhana",
    3: "Yossi"
}

# Warna menggunakan format BGR OpenCV
colors = {
    1: (255, 0, 0),   # Biru - Sandhika
    2: (0, 255, 0),   # Hijau - Dhana
    3: (0, 0, 255)    # Merah - Yossi
}

cap = cv2.VideoCapture(0)

while True:

    _, frame = cap.read()

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

        if predicted_id in names and confidence < 100:

            name = names[predicted_id]

            # Mengubah distance menjadi score seperti kode awal
            similarity = round(100 - confidence)

            color = colors[predicted_id]

        else:

            name = "Unknown"
            similarity = 0

            # Putih untuk wajah yang tidak dikenali
            color = (255, 255, 255)

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
            (x + 5, y + h - 10),
            font,
            0.7,
            color,
            2
        )

    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()