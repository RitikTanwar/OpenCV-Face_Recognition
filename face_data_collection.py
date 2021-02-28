import cv2
import numpy as np

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
skip = 0
face_data = []
data_setPath = 'C:/Users/Ritik Tanwar/Documents/python learning/openCV/data/'
filename = input('Enter the name of person ')
while True:
    ret, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if ret == False:
        continue
    # detectMultiScale(frame,scaling_factor,no. of neighbours)
    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
    # print(faces)
    faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
    face_section = 0
    # Picking largest face according to area(w*h)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)

        # Extract region of interst i.e. cropping out reqd face
        offset = 10  # Padding of 10px in each side
        # in cv coordinate of frame is y,x
        face_section = frame[y-offset:y+h+offset, x-offset:x+w+offset]
        face_section = cv2.resize(face_section, (100, 100))
        skip += 1
        if skip % 10 == 0:
            face_data.append(face_section)
            print(len(face_data))
    cv2.imshow('Video Camera', frame)
    cv2.imshow('Face section', face_section)
    # Storing every 10th face
    # if skip%10==0:
    #     pass
    pressed_key = cv2.waitKey(1) & 0xFF
    if pressed_key == ord('q'):
        break
    # Convert our face data list to numpy array
face_data = np.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0], -1))
print(face_data.shape)
np.save(data_setPath + filename + '.npy', face_data)
print("Data succesfully saved at ", data_setPath + filename + '.npy')
cap.release()
cv2.destroyAllWindows()
