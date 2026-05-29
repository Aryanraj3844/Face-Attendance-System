import face_recognition
import cv2
import csv
from datetime import datetime

video_capture = cv2.VideoCapture(0)

known_image = face_recognition.load_image_file("Image-Basic/Aryan.1.jpg")
known_encoding = face_recognition.face_encodings(known_image)[0]

known_face_encodings = [known_encoding]
known_face_names = ["Aryan"]

students = known_face_names.copy()

face_locations = []
face_encodings = []

now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

f = open(current_date + '.csv', 'w+', newline='')
lnwriter = csv.writer(f)

while True:
    _, frame = video_capture.read()

    small_frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)

    rgb_small_frame = small_frame[:, :, ::-1]

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(
        rgb_small_frame,
        face_locations
    )

    for face_encoding in face_encodings:

        matches = face_recognition.compare_faces(
            known_face_encodings,
            face_encoding
        )

        face_name = ""

        if True in matches:
            first_match_index = matches.index(True)
            face_name = known_face_names[first_match_index]

            if face_name in students:
                students.remove(face_name)
                lnwriter.writerow([face_name, "Present"])

    cv2.imshow("Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
f.close()