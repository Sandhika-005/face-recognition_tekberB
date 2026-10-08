import cv2
import os
import time

# =========================================
# LOAD HAAR CASCADE
# =========================================

faceCascade = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

if faceCascade.empty():
    print("Haar Cascade tidak ditemukan.")
    print("Pastikan file haarcascade_frontalface_default.xml")
    print("berada di folder yang sama dengan file Python.")
    exit()


# =========================================
# DATASET FOLDER
# =========================================

dataset_path = "dataset"

if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)


# =========================================
# DAFTAR ORANG
# =========================================

people = {
    1: "Sandhika",
    2: "Dhana",
    3: "Yossi"
}


# =========================================
# PENGATURAN DATASET
# =========================================

TARGET_PHOTOS = 100
DURATION = 15.0

# 15 detik / 100 foto = 0,15 detik
CAPTURE_INTERVAL = DURATION / TARGET_PHOTOS


# =========================================
# WEBCAM
# =========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Webcam tidak dapat dibuka.")
    exit()


# =========================================
# PROSES SETIAP ORANG
# =========================================

for person_id, person_name in people.items():

    print("\n====================================")
    print(f"Nama   : {person_name}")
    print(f"ID     : {person_id}")
    print(f"Target : {TARGET_PHOTOS} foto")
    print(f"Waktu  : {DURATION} detik")
    print("====================================")

    input(
        f"\nSilakan siapkan {person_name}. "
        f"Tekan ENTER untuk mulai..."
    )

    count = 0

    start_time = time.time()
    next_capture_time = start_time

    while True:

        ret, frame = cap.read()

        if not ret:
            print("Frame kamera gagal dibaca.")
            break

        # =================================
        # WAKTU
        # =================================

        current_time = time.time()
        elapsed_time = current_time - start_time
        remaining_time = max(
            0,
            DURATION - elapsed_time
        )

        # =================================
        # GRAYSCALE
        # =================================

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        # =================================
        # DETEKSI WAJAH
        # =================================

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        # =================================
        # TAMPILKAN WAJAH
        # =================================

        for (x, y, w, h) in faces:

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (255, 0, 0),
                2
            )

            # =================================
            # AMBIL FOTO SESUAI INTERVAL
            # =================================

            if (
                current_time >= next_capture_time
                and count < TARGET_PHOTOS
            ):

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

                # Jadwal foto berikutnya
                next_capture_time = (
                    start_time
                    + count * CAPTURE_INTERVAL
                )

            # =================================
            # INFO ORANG
            # =================================

            cv2.putText(
                frame,
                person_name,
                (x + 5, y - 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Dataset: {count}/{TARGET_PHOTOS}",
                (x + 5, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

        # =================================
        # INFO WAKTU
        # =================================

        cv2.putText(
            frame,
            f"Time: {remaining_time:.1f}s",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "Face Dataset Collection",
            frame
        )

        # =================================
        # TEKAN Q UNTUK BERHENTI
        # =================================

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):

            print("\nDataset dihentikan oleh user.")

            cap.release()
            cv2.destroyAllWindows()
            exit()

        # =================================
        # BERHENTI JIKA 100 FOTO
        # =================================

        if count >= TARGET_PHOTOS:

            print(
                f"{person_name}: "
                f"{count}/{TARGET_PHOTOS} foto selesai."
            )

            break

        # =================================
        # BERHENTI SETELAH 15 DETIK
        # =================================

        if elapsed_time >= DURATION:

            print(
                f"{person_name}: waktu 15 detik selesai."
            )

            print(
                f"Foto berhasil diambil: "
                f"{count}/{TARGET_PHOTOS}"
            )

            break


    # =====================================
    # HASIL ORANG SAAT INI
    # =====================================

    if count >= TARGET_PHOTOS:

        print(
            f"{person_name} BERHASIL: "
            f"{count}/{TARGET_PHOTOS} foto."
        )

    else:

        print(
            f"{person_name} BELUM LENGKAP: "
            f"{count}/{TARGET_PHOTOS} foto."
        )


# =========================================
# SEMUA ORANG SELESAI
# =========================================

print("\n====================================")
print("SEMUA DATASET SELESAI")
print("====================================")

print("Sandhika : selesai")
print("Dhana    : selesai")
print("Yossi    : selesai")

cap.release()
cv2.destroyAllWindows()