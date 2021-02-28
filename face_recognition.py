import os
import numpy as np
import cv2


def dist(x1, x2):
    return np.sqrt(sum((x1-x2)**2))


def knn(train, test, k=5):
    vals = []
    m = train.shape[0]
    # print(train[:, -1])
    for i in range(m):
        ix = train[i, :-1]
        iy = train[i, -1]
        d = dist(test, ix)
        vals.append([d, iy])
    # print("vals", vals)
    val = sorted(vals, key=lambda x: x[0])[:k]
    # vals = vals[:k]
    # labels = np.array(vals)
    # labels = labels[:, -1]
    label = np.array(val)[:, -1]
    # print("label", label)
    np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
    output = np.unique(label, return_counts=True)
    idx = np.argmax(output[1])
    # print(output[0][idx])
    return output[0][idx]


# Init camera
cam = cv2.VideoCapture(1, cv2.CAP_DSHOW)

# Face detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

data_setPath = 'C:/Users/Ritik Tanwar/Documents/python learning/openCV/data/'
face_data = []
labels = []

class_id = 0  # Labels for given file
name = {}  # Mapping id with name

# Data Preparation
for fx in os.listdir(data_setPath):
    if fx.endswith('.npy'):
        name[class_id] = fx[:-4]
        print("Loaded ", fx)
        data_item = np.load(data_setPath+fx)
        face_data.append(data_item)
        # Create labels for class
        target = class_id*np.ones((data_item.shape[0],))
        class_id += 1
        labels.append(target)

# print("labels", labels)
face_dataset = np.concatenate(face_data, axis=0)
face_labels = np.concatenate(labels, axis=0).reshape((-1, 1))

print(face_dataset.shape)
print(face_labels.shape)

train_set = np.concatenate((face_dataset, face_labels), axis=1)
# print(train_set)


while True:
    ret, frame = cam.read()
    if ret == False:
        continue

    faces = face_cascade.detectMultiScale(frame, 1.3, 5)
    # face_section = 0
    for (x, y, w, h) in faces:
        # Get the region of interest
        offset = 10
        face_section = frame[y-offset:y+h+offset, x-offset:x+w+offset]
        face_section = cv2.resize(face_section, (100, 100))

        out = knn(train_set, face_section.flatten())

        # Display on the screen
        # print(int(out))
        pred_name = name[int(out)]
        cv2.putText(frame, pred_name, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)

    cv2.imshow("Faces", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
