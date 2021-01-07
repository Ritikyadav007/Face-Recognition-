import cv2

trained_data= cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

webcam = cv2.VideoCapture(0)

while True:

    success_frame_read, frame=webcam.read()

    grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Detecting faces
    face_coordinates = trained_data.detectMultiScale(grayscaled_frame)

    #drawing rectangles around the faces
    for(x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2) 

    cv2.imshow('Faces Detected', frame)
    key=cv2.waitKey(1)

    #to stop the webcame if S key is pressed
    if key==115:
        break

webcam.release()

print("Code Completed")