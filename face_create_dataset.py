import cv2
import os

faceCascade = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

dataset_path = "dataset"

if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

people = {
    1: "Sandhika",
    2: "Dhana",
    3: "Yossi"
}

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Webcam tidak dapat dibuka.")
    exit()

for person_id, person_name in people.items():

    count = 0

    print("\n====================================")
    print(f"Dataset untuk : {person_name}")
    print(f"ID           : {person_id}")
    print("Target       : 100 foto")
    print("====================================")

    while True:

        ret, frame = cap.read()

        if not ret:
            print("Frame kamera gagal dibaca.")
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

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (255, 0, 0),
                2
            )

            count += 1

            filename = (
                dataset_path
                + "/User."
                + str(person_id)
                + "."
                + str(count)
                + ".jpg"
            )

            cv2.imwrite(
                filename,
                gray[y:y+h, x:x+w]
            )

            cv2.putText(
                frame,
                f"{person_name}: {count}/100",
                (x + 5, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

        cv2.imshow(
            "Face Dataset Collection",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("\nDataset collection dihentikan.")
            cap.release()
            cv2.destroyAllWindows()
            exit()

        if count >= 100:
            break

    print(
        f"{person_name} dataset selesai: "
        f"{count}/100 foto"
    )

    if person_id < len(people):

        cv2.destroyWindow(
            "Face Dataset Collection"
        )

        input(
            f"\nTekan ENTER untuk mulai "
            f"dataset {people[person_id + 1]}..."
        )

        cv2.namedWindow(
            "Face Dataset Collection"
        )

print("\n====================================")
print("SEMUA DATASET SELESAI")
print("====================================")
print("Sandhika : 100 foto")
print("Dhana    : 100 foto")
print("Yossi    : 100 foto")

cap.release()
cv2.destroyAllWindows()