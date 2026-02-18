import cv2
import mediapipe as mp

# Mediapipe face mesh (468 nokta)
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Kamera
kamera = cv2.VideoCapture(0)

# FOTOĞRAFLARI YÜKLE
img_sol = cv2.imread("C:\Program Files\Python_VSCode_OpenCV\sag.jpg")
img_sag = cv2.imread("C:\Program Files\Python_VSCode_OpenCV\sol.jpg")
img_duz = cv2.imread("C:\Program Files\Python_VSCode_OpenCV\duz.jpg")

aktif_resim = img_duz  # başlangıçta düz

while True:
    ret, frame = kamera.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    sonuc = face_mesh.process(rgb)

    if sonuc.multi_face_landmarks:
        for yuz in sonuc.multi_face_landmarks:
            # Önemli noktalar
            burun = yuz.landmark[1]
            sol_goz = yuz.landmark[33]
            sag_goz = yuz.landmark[263]

            h, w, _ = frame.shape
            burun_x = int(burun.x * w)
            sol_x = int(sol_goz.x * w)
            sag_x = int(sag_goz.x * w)

            sol_mesafe = abs(burun_x - sol_x)
            sag_mesafe = abs(burun_x - sag_x)

            if sol_mesafe > sag_mesafe + 15:
                durum = "Kafa Saga Donmus"
                aktif_resim = img_sag
            elif sag_mesafe > sol_mesafe + 15:
                durum = "Kafa Sola Donmus"
                aktif_resim = img_sol
            else:
                durum = "Duz Bakiyor"
                aktif_resim = img_duz

            cv2.putText(frame, durum, (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Kamera ve fotoğrafı aynı anda göster
    cv2.imshow("Kamera", frame)

    if aktif_resim is not None:
        cv2.imshow("Duruma Gore Resim", aktif_resim)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

kamera.release()
cv2.destroyAllWindows()