import cv2
import numpy as np
from vispy import scene, app
import cupy as cp


# Initialize video streams
cap1 = cv2.VideoCapture(0)  # First camera
cap2 = cv2.VideoCapture(2)  # Second camera, change index if needed

if not cap1.isOpened():
    print("Error: Could not open /dev/video2")
if not cap2.isOpened():
    print("Error: Could not open /dev/video0")
if not cap1.isOpened() or not cap2.isOpened():
    exit()

# Create Vispy canvases
canvas1 = scene.SceneCanvas(keys='interactive', size=(640,480), title="Camera 1", show=True)
canvas2 = scene.SceneCanvas(keys='interactive', size=(640,480), title="Camera 2", show=True)


view1 = canvas1.central_widget.add_view()
view2 = canvas2.central_widget.add_view()

view1.camera.aspect = 1


image_visual1 = scene.visuals.Image(np.zeros((480, 640, 3), dtype=np.uint8))
image_visual2 = scene.visuals.Image(np.zeros((480, 640, 3), dtype=np.uint8))

view1.add(image_visual1)
view2.add(image_visual2)


def update(event):
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret1:
        print("Error: Failed to read from /dev/video0")
    if not ret2:
        print("Error: Failed to read from /dev/video2")
    if not ret1 or not ret2:
        return

    # Convert BGR to RGB (Vispy requires RGB format)
    frame1_rgb = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
    frame2_rgb = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
    # Print frame shape to confirm data capture
    # print(f"Frame 1 Shape: {frame1.shape}, Frame 2 Shape: {frame2.shape}")

    # Update images
    image_visual1.set_data(frame1_rgb)
    image_visual2.set_data(frame2_rgb)


    canvas1.update()
    canvas2.update()


timer = app.Timer(interval=1/30)  # Run at 30 FPS
timer.connect(update)
timer.start()

app.run()

# Release resources
cap1.release()
cap2.release()
