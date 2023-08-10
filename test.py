import cv2
import tkinter as tk
from PIL import Image, ImageTk

# Face classifier
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# Smile classifier
smile_detector = cv2.CascadeClassifier('haarcascade_smile.xml')

# Create a window
window = tk.Tk()
window.title("E-Voting: Smile Verification")

# Create a label for e-voting information
info_label = tk.Label(window, text="Welcome to E-Voting!\nPlease verify that you are human by smiling.")
info_label.pack(pady=10)

# Create a frame for webcam display
frame = tk.Frame(window)
video_label = None
webcam = None
success_label = None

def show_camera():
    global video_label, webcam, success_label

    # Hide the e-voting information and show the webcam display
    info_label.pack_forget()
    frame.pack(pady=10)

    # Check if the previous camera window exists and destroy it
    if video_label is not None:
        video_label.pack_forget()
        video_label = None

        # Release the previous webcam instance
        webcam.release()

    # Clear the success message if it exists
    if success_label is not None:
        success_label.pack_forget()
        success_label = None

    # Grab webcam feed
    webcam = cv2.VideoCapture(0)

    # Create a label to display the video feed
    video_label = tk.Label(frame)
    video_label.pack()

    def verify_smile():
        # Read a frame from the webcam
        successful_frame_read, frame = webcam.read()

        if successful_frame_read:
            # Convert the frame to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces in the frame
            faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            # Iterate over detected faces
            for (x, y, w, h) in faces:
                # Draw a rectangle around the face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                # Extract the face region of interest
                face_roi = gray[y:y+h, x:x+w]

                # Detect smiles in the face region
                smiles = smile_detector.detectMultiScale(face_roi, scaleFactor=1.7, minNeighbors=43, minSize=(30, 30))

                # If at least one smile is detected, display a success message and close the camera window
                if len(smiles) > 0:
                    cv2.putText(frame, 'Smile Detected', (x-50, y+h+40), fontScale=1, fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, color=(0, 255, 0))
                    cv2.imshow("Video Feed", frame)
                    cv2.waitKey(2000)  # Display the frame for 2 seconds
                    cv2.destroyAllWindows()
                    webcam.release()
                    message_label = tk.Label(window, text="You have been verified to be human", fg="green", font=("Helvetica", 14))
                    message_label.pack(pady=10)
                    return

                else:
                    cv2.putText(frame, 'No Smile Detected, Show Some teeth', (x-200, y+h+40), fontScale=1, fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, color=(255, 255, 255))

            # Convert the frame to RGB for displaying in Tkinter
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            image = ImageTk.PhotoImage(image)

            # Update the video label with the new frame
            video_label.config(image=image)
            video_label.image = image

        # Schedule the next smile verification
        video_label.after(1, verify_smile)

    # Start smile verification
    verify_smile()

# Create a button for showing the camera
show_camera_button = tk.Button(window, text="Verify Human (Smile)", command=show_camera)
show_camera_button.pack(pady=10)

# Start the Tkinter event loop
window.mainloop()
