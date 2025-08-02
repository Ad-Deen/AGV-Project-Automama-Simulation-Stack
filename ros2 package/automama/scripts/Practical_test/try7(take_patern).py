import cv2
import numpy as np
import os
from vispy import scene, app
import time

# Initialize video streams
cap1 = cv2.VideoCapture(0)  # First camera

if not cap1.isOpened():
    print("Error: Could not open /dev/video0")
    exit()

# Create folder to store calibration images in the home directory
os.makedirs(os.path.expanduser("~/callib_pics/"), exist_ok=True)

# Create Vispy canvas
canvas1 = scene.SceneCanvas(keys='interactive', size=(640, 480), title="Camera 1", show=True)
view1 = canvas1.central_widget.add_view()
view1.camera.aspect = 1
image_visual1 = scene.visuals.Image(np.zeros((480, 640, 3), dtype=np.uint8))
view1.add(image_visual1)

# Initialize last capture time
last_capture_time = time.time()

def update(event):
    global last_capture_time
    
    # Read the frame from the camera
    ret1, frame1 = cap1.read()
    if not ret1:
        print("Error: Failed to read from /dev/video0")
        return
    
    # Convert the BGR frame to RGB (Vispy requires RGB format)
    frame1_rgb = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
    
    # Update the display with the current frame
    image_visual1.set_data(frame1_rgb)
    canvas1.update()

    # Check if 1 second has passed since the last capture
    current_time = time.time()
    if current_time - last_capture_time >= 1.0:
        # Save the frame as an image in the '~/callib_pics/' directory
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        image_filename = os.path.expanduser(f"~/callib_pics/calib_{timestamp}.png")
        cv2.imwrite(image_filename, frame1)
        print(f"Image saved: {image_filename}")
        
        # Update the last capture time
        last_capture_time = current_time

# Create a timer to update the frames at 30 FPS
timer = app.Timer(interval=1/30)  # Run at 30 FPS
timer.connect(update)
timer.start()

# Run the app
app.run()

# Release the camera resource when done
cap1.release()
