import cv2
import numpy as np
from vispy import scene, app
import cupy as cp
import cvcuda
'''
This script is to extension of simple dual camera vision to finding the cost function of the pixel around its neighboring proximity
'''

def process_image_rgb(image):
    """
    Converts an image to grayscale, applies Canny edge detection to each channel (R, G, B),
    and returns the combined edge map.
    
    Parameters:
    - image: A 640x480 BGR image.
    
    Returns:
    - combined_edge_map: A 640x480 binary image (edges detected).
    """
    # Split the image into R, G, B channels
    (B, G, R) = cv2.split(image)
    
    # Apply Canny edge detection on each channel
    edges_r = cv2.Canny(R, 200, 250)
    edges_g = cv2.Canny(G, 200, 250)
    edges_b = cv2.Canny(B, 200, 250)
    
    # Combine the edge maps from all three channels
    combined_edge_map = cv2.bitwise_or(edges_r, edges_g)
    combined_edge_map = cv2.bitwise_or(combined_edge_map, edges_b)
    
    # Return the combined edge map
    return combined_edge_map

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
canvas1 = scene.SceneCanvas(keys='interactive', size=(640, 480), title="left", show=True)
canvas2 = scene.SceneCanvas(keys='interactive', size=(640, 480), title="right", show=True)

view1 = canvas1.central_widget.add_view()
view2 = canvas2.central_widget.add_view()

view1.camera.aspect = 1
view2.camera.aspect = 1

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

    # Resize frames to 640x480
    frame1_resized = cv2.resize(frame1, (640, 480))
    frame2_resized = cv2.resize(frame2, (640, 480))

    # Convert BGR to RGB (Vispy requires RGB format)
    left_rgb = cv2.cvtColor(frame1_resized, cv2.COLOR_BGR2RGB)
    right_rgb = cv2.cvtColor(frame2_resized, cv2.COLOR_BGR2RGB)
    edges1 = process_image_rgb(left_rgb)
    edges2 = process_image_rgb(right_rgb)
    # Update images
    image_visual1.set_data(edges1)
    image_visual2.set_data(edges2)

    canvas1.update()
    canvas2.update()

timer = app.Timer(interval=1/30)  # Run at 30 FPS
timer.connect(update)
timer.start()

app.run()

# Release resources
cap1.release()
cap2.release()
