import cv2 
#cv2 kütüphanesini çağırdık

yuz_tanima=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')


kamera=cv2.VideoCapture(0)#varsayılan kamera=0
if not kamera.isOpened():
    print("Kamera açılamadı!")
    exit()

while True:
    ret, frame = kamera.read()  # kare al

    if not ret:
        print("Kare okunamadı!")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = yuz_tanima.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(30, 30)#30×30 pikselden küçük yüzleri hiç deneme
        #yanlış yüz algılamayı önlemek için 
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow("Webcam Yuz Tanima", frame) #gösterilcek görüntü frame=webcamden gelen anlık görüntü
    #waitkey görüntünün yenilenmesini sağlar
    #1 milisaniye bekle ve klavyeden tuş basıldı mı kontrol et
    # q tuşuna basınca çık
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

kamera.release()# webcami serbest bırakır donanımı kapatır
cv2.destroyAllWindows()

#MANTIĞI
#xml dosyası istatiksel özellikleri tutar
#haar özellikleri= siyah beyaz dikdörtgenlerden oluşan kontrast farkı ölçümü
#yüzdeki bazı bölgeler doğal kontratlıdır
#Gözler	Göz çevresi koyu, yanaklar açık
#Göz–burun hattı	Dikey kontrast
#Burun köprüsü	Ortası açık, yanları koyu
#Alın–göz hattı	Yatay geçiş
#bu kontrastı yakaladığında burda yüz olabilir diyor
#cascade(aşamalı olarak elenerek gider)