import cv2

# Face classifier
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_detector = cv2.CascadeClassifier('haarcascade_smile.xml')

# grab webcam feed
webcam = cv2.VideoCapture(0)

while True:
    #Read the current frame from the webcam video stream
    succesful_frame_read, frame = webcam.read()

	#if there's an error, abort
    if not succesful_frame_read:
       break

    # Change to grayscale
    frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces first
    faces =  face_detector.detectMultiScale(frame_grayscale)

    # Run face detection within each of those faces
    # Iterate over the detected faces
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 200, 50), 4)

        # Get sub frame (using numpy N-dimensional array sclicing)
        the_face = frame[y:y+h, x:x+w]
    
        # Change to grayscale
        face_grayscale = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)

        smiles =  smile_detector.detectMultiScale(face_grayscale, scaleFactor=1.7, minNeighbors=20)

        # Find all smiles in the face
        #for (x_, y_, w_, h_) in smiles:
            # Draw a rectangle around the smile
        #    cv2.rectangle(the_face, (x_, y_), (x_ + w_, y_ + h_), (50, 50, 200), 3)

        # Label this face as smiling
        if len(smiles) > 0:
            cv2.putText(frame, 'Smiling', (x, y+h+40), fontScale=3, fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255, 255, 255))

    # Show the current frame
    cv2.imshow('Smile Detector', frame)

    #  Display
    cv2.waitKey(1)

# Cleanup
webcam.release()

cv2.destroyAllWindows()

# Code ran without errors 
print("Code Completed")
