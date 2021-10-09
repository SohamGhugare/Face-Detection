import cv2

# Fetching video capture from webcam
cap = cv2.VideoCapture(0)

# Cascades

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_fullbody.xml")

while True:
    # Capture frame-by-frame from webcam
    _, frame = cap.read()

    # Gray scale for classification
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)
    body = body_cascade.detectMultiScale(gray, 1.2, 5)

    # Drawing the face
    for (x, y, width, height) in faces:
        cv2.rectangle(frame, (x, y), (x + width, y+height), (255, 0, 0), 3)

    # Display the resulting frame
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
