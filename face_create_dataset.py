import cv2
import os

faceCascade = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

people = {
    1: "Sandhika",
    2: "Dhana",
    3: "Yossi"
}

dataset_path = "dataset/"

if not os.path.exists(dataset_path):
    os.mkdir(dataset_path)

cap = cv2.VideoCapture(0)

for person_id, person_name in people.items():

    print(f"\nSilakan {person_name} menghadap kamera...")
    print("Pengambilan dataset dimulai.")

    count = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            print("Kamera tidak dapat digunakan.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        for (x, y, w, h) in faces:

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            count += 1

            file_name = (
                dataset_path
                + person_name
                + "_"
                + str(person_id)
                + "_"
                + str(count)
                + ".jpg"
            )

            cv2.imwrite(
                file_name,
                gray[y:y+h, x:x+w]
            )

            cv2.putText(
                frame,
                f"{person_name}: {count}/100",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

        cv2.imshow("Camera", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if count >= 100:
            break

    print(f"Dataset {person_name} selesai: {count} foto.")

    if person_id != 3:
        input("Tekan ENTER untuk lanjut ke orang berikutnya...")

cap.release()
cv2.destroyAllWindows()