import cv2

cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if(ret == False):
        continue

    cv2.imshow("Video Frame", frame)
    cv2.imshow("Gray Frame", gray_frame)

    # Wait for user to enter 'q' key and exit the cam
    key_pressed = cv2.waitKey(1) & 0xFF  # 0xFF->11111111
    # cv2.waitKey(0) -> program will stop when any key is pressed
    if key_pressed == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
