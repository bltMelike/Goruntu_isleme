import cv2
import mediapipe as mp
#mediapipe ile 468 tane yüz noktası belirlenir
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

kamera = cv2.VideoCapture(0)

while True:
    ret, frame = kamera.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    sonuc = face_mesh.process(rgb)

    if sonuc.multi_face_landmarks:
        for yuz in sonuc.multi_face_landmarks:
            # ÖNEMLİ NOKTALAR
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
                durum = "Kafa Saga donmuş"
            elif sag_mesafe > sol_mesafe + 15:
                durum = "Kafa Sola Donmuş"
            else:
                durum = "Düz bakiyor"

            cv2.putText(frame, durum, (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (30, 50, 40), 2)

    cv2.imshow("Head Direction", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

kamera.release()
cv2.destroyAllWindows()