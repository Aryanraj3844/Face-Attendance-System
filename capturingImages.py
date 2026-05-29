import cv2

cam = cv2.VideoCapture(0)


face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
if face_detector.empty():
    print("Face Cascade Load Failed!")
    exit()
else:
    print("Face Cascade Loaded Successfully!")

name = input("Enter Your Name: ")

count = 0

while True:
    ret, img = cam.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

        count += 1

        cv2.imwrite(
            "Image-Basic/" + str(name) + "." + str(count) + ".jpg",
            gray[y:y+h,x:x+w]
        )

        cv2.imshow('image', img)

    k = cv2.waitKey(100) & 0xff

    if k == 27:
        break
    elif count >= 20:
        break

cam.release()
cv2.destroyAllWindows()