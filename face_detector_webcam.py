import cv2
from random import randrange

# Load some pre-trained data on face frontals from opencv (haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# To capture video from webcam
webcam = cv2.VideoCapture(0)

# Iterate over the frames
while True:

    # Read current frame
    successful_frame_read, frame = webcam.read()

    # Convert to grayscale
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect face(s) from image(s)
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    # Draw rectangles around the faces
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('My Face Detector', frame)
    key = cv2.waitKey(1)

    # Stop if Q key is pressed (ascii key)
    if key == 81 or key == 113:
        break

# Release VideoCapture object
webcam.release()


print('cool')