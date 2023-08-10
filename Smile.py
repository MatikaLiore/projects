import cv2
import tkinter as tk
from PIL import Image, ImageTk
from collections import deque

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
show_camera_button = None

# Define the maximum number of frames for smile detection
max_frames = 60

# Create a deque to store the frames for smile detection
frame_buffer = deque(maxlen=max_frames)

def show_camera():
    global video_label, webcam, success_label, show_camera_button

    # Hide the e-voting information and show the webcam display
    info_label.pack_forget()
    frame.pack(pady=10)

    # Store a reference to the "Verify Smile" button
    show_camera_button_ref = show_camera_button

    # Hide the "Verify Smile" button
    show_camera_button.pack_forget()

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

    def try_again():
        # Remove the success message
        success_label.pack_forget()

        # Start frame verification again
        verify_smile()

    def continue_voting():
        # Destroy the camera window and perform other actions
        window.destroy()
        # Additional actions for continuing with voting

    def verify_smile():
        # Read a frame from the webcam
        _, frame = webcam.read()

        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
#        faces = face_detector.detectMultiScale(gray)
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        # Iterate over detected faces
        for (x, y, w, h) in faces:
            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Extract the face region of interest
            face_roi = gray[y:y+h, x:x+w]

            # Detect smiles in the face region
            smiles = smile_detector.detectMultiScale(face_roi, scaleFactor=1.7, minNeighbors=43, minSize=(30, 30))

            # If at least one smile is detected, draw rectangles around the smiles
            if len(smiles) > 0:
                cv2.putText(frame, 'Detecting Smile, remain steady', (x-50, y+h+40), fontScale=1, fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, color=(0, 255, 0))
                for (sx, sy, sw, sh) in smiles:
                    cv2.rectangle(frame, (x+sx, y+sy), (x+sx+sw, y+sy+sh), (255, 0, 0), 2)
            else:
                cv2.putText(frame, 'Detecting No Smile!', (x-50, y+h+40), fontScale=1, fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, color=(255, 255, 255))

            # Append the frame's smile detection result to the frame buffer
            frame_buffer.append(len(smiles) > 0)

        # Convert the frame to RGB for displaying in Tkinter
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)

        # Update the video label with the new frame
        video_label.config(image=image)
        video_label.image = image

        # Check if the frame buffer is full
        if len(frame_buffer) == max_frames:
            # Count the number of frames with smiles
            smile_count = sum(frame_buffer)
            # Calculate the percentage of frames with smiles
            smile_percentage = (smile_count / max_frames) * 100

            # Display the message based on the smile percentage
            if smile_percentage >= 60:
                success_label = tk.Label(window, text="Smile verified!\nYou are human.", fg="green", font=("Helvetica", 14))
                success_label.pack(pady=10)

                # Show the "Continue with Voting" button
                continue_button.pack()

            else:
                success_label = tk.Label(window, text="Smiles in frames were below the threshold.", fg="red", font=("Helvetica", 14))
                success_label.pack(pady=10)

                # Show the "Try Again" button
                try_again_button.pack()

        else:
            # Schedule the next frame verification
            video_label.after(1, verify_smile)

    # Create the "Try Again" button
    try_again_button = tk.Button(window, text="Try Again", command=try_again)

    # Create the "Continue with Voting" button
    continue_button = tk.Button(window, text="Continue with Voting", command=continue_voting)

    # Start frame verification
    verify_smile()

# Create a button for showing the camera
show_camera_button = tk.Button(window, text="Verify Human (Smile)", command=show_camera)
show_camera_button.pack(pady=10)

# Start the Tkinter event loop
window.mainloop()
