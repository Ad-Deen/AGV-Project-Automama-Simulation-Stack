import numpy as np
import cv2 as cv
from vispy import scene, app

# Checkerboard parameters (adjust according to your checkerboard)
CHECKERBOARD = (10, 7)  # Internal corners (width, height)
SQUARE_SIZE = 9  # Size of a square in mm (only for calibration)
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Initialize video capture
cap = cv.VideoCapture(0)  # Change to the correct camera index if needed

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Create Vispy canvas for visualization
canvas = scene.SceneCanvas(keys='interactive', size=(640, 480), title="Checkerboard Detection", show=True)
view = canvas.central_widget.add_view()
view.camera.aspect = 1
image_visual = scene.visuals.Image(np.zeros((480, 640, 3), dtype=np.uint8))
view.add(image_visual)

def update(event):
    """Capture a frame, detect checkerboard, and update visualization."""
    ret, frame = cap.read()
    if not ret:
        print("Error: Couldn't capture a frame.")
        return

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Find chessboard corners
    found, corners = cv.findChessboardCorners(gray, CHECKERBOARD, cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_FAST_CHECK + cv.CALIB_CB_NORMALIZE_IMAGE)
    
    if found:
        corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        frame = cv.drawChessboardCorners(frame, CHECKERBOARD, corners2, found)
        print('corner found')
    print('nothing')
    # Update Vispy visualization
    image_visual.set_data(frame)
    canvas.update()

# Set up Vispy timer to refresh frames
timer = app.Timer(interval=0.03, connect=update, start=True)

app.run()

cap.release()
