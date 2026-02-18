import cv2 
#cv2 kütüphanesini çağırdık

yuz_tanima=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
#opencv içerisinde bulunan CascadeClassifer sınıfından nesne ürettik
#parametre olarak da yazdığımız xml dosyasını kullanıyoruz
# bu dosyada yüzün tanınması için yüzün çeşitli özellikleri mevcut
#gözler arasındaki mesafe burun uzunluğu vs gibi özellikleri tanıdığında yüz tanımlanır
#direkt dosya yolunu parametre olarak verdik

image=cv2.imread('takim.jpg')#imread fonksiyonu ile görüntü okunur
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#resmimizi tek renk gri yaptık
faces=yuz_tanima.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=5)
#nesne içindeki fonksiyonu çağırdık görüntünün boyut oranını ayarladık
#her seferinde %1.5 küçültüyor min neighbors ise resimdeki dikdörtgencikleri algılar 5 tane olma koşulu koyuyor

for(x,y,w,h) in faces:
# yüzler 4 farklı değer ddöndürüyor
# x --> yüzün sol üst köşesinin yatay kordinatı
# y --> yüzün sol üst köşesinin dikey kordinatı
# w --> yüzün genişliği
# h --> yüzün yüksekliği
    cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
    #burda kullanıcıya resmi göstericez bir dikdörtgen çizicz
    #ilk parametre image gösterilecek resim ikinci solt üst köşenin kordinatı
    #3. parametre sağ alt köşenin kordinatı 4. parametre oluşturulacak resmin rengi yeşil ve kırmızı 0 mavi 225 dikdörtgen mavi
    #son parametre ise oluşturulacak dikdörtgenin kalınlığı
cv2.namedWindow('YÜZ TANIMA',cv2.WINDOW_NORMAL)
cv2.imshow('YÜZ TANIMA',image)
cv2.waitKey(0)
cv2.destroyAllWindows()
