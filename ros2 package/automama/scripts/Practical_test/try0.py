import cv2
import numpy as np
from vispy import scene
from vispy.app import Application

# Create a Vispy canvas
canvas1 = scene.SceneCanvas(keys='interactive', size=(640, 480), title="Camera Stream1", show=True)
view1 = canvas1.central_widget.add_view()
view1.camera.aspect = 1

canvas2 = scene.SceneCanvas(keys='interactive', size=(640, 480), title="Camera Stream2", show=True)
view2 = canvas2.central_widget.add_view()
view2.camera.aspect = 1

# Create an image visual to display frames
image_visual1 = scene.visuals.Image()
view1.add(image_visual1)

# Create an image visual to display frames
image_visual2 = scene.visuals.Image()
view2.add(image_visual2)

# Open the video stream (example for /dev/video0)
cap1 = cv2.VideoCapture('/dev/video0')
cap2 = cv2.VideoCapture('/dev/video2')
# print(cap2)
# app = Application()

if not cap1.isOpened() and not cap2.isOpened():
    print("Error: Could not open video stream.")
else:
    app = Application()

    while True:
        ret1, frame1 = cap1.read()
        print(frame1)
        ret2, frame2 = cap2.read()
        print(frame2)
        if not ret1 or not ret2:
            print('broken here')
            break

        # Convert the frame from BGR to RGB (Vispy uses RGB)
        frame_rgb1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        frame_rgb2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

        # Update the image visual with the current frame
        image_visual1.set_data(frame_rgb1)
        
        # Update the canvas
        canvas1.update()
        canvas2.update()

        app.process_events()

    # Release the video capture
    cap1.release()
    cap2.release()
