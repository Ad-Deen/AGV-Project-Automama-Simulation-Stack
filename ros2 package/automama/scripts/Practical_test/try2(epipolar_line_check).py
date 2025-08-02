import cv2
import numpy as np
from vispy import scene, app
'''
This script is used to check if the epipolar lines in frames are on same horizontal epipolar line or not
'''
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
canvas1 = scene.SceneCanvas(keys='interactive', size=(640, 480), title="Camera 1", show=True)
canvas2 = scene.SceneCanvas(keys='interactive', size=(640, 480), title="Camera 2", show=True)

view1 = canvas1.central_widget.add_view()
view2 = canvas2.central_widget.add_view()

view1.camera.aspect = 1
view2.camera.aspect = 1

image_visual1 = scene.visuals.Image(np.zeros((480, 640, 3), dtype=np.uint8))
image_visual2 = scene.visuals.Image(np.zeros((480, 640, 3), dtype=np.uint8))

view1.add(image_visual1)
view2.add(image_visual2)

# Initialize the ORB detector and BFMatcher
orb = cv2.ORB_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

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

    # Convert BGR to grayscale
    gray1 = cv2.cvtColor(frame1_resized, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2_resized, cv2.COLOR_BGR2GRAY)

    # Detect ORB keypoints and descriptors
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    # Match descriptors using BFMatcher
    matches = bf.match(des1, des2)

    # Sort the matches based on their distances
    matches = sorted(matches, key = lambda x:x.distance)

    # Find the epipolar lines
    epipolar_deviations = []
    for match in matches[:10]:  # Use top 10 matches
        pt1 = kp1[match.queryIdx].pt
        pt2 = kp2[match.trainIdx].pt

        # Draw the corresponding points
        cv2.circle(frame1_resized, (int(pt1[0]), int(pt1[1])), 5, (0, 255, 0), 2)
        cv2.circle(frame2_resized, (int(pt2[0]), int(pt2[1])), 5, (0, 255, 0), 2)

        # Compute the epipolar line in the second frame
        # Here we assume a simple case where the epipolar line is the horizontal line from pt1
        line_equation = np.array([pt1[0], pt1[1], 1])  # [x, y, 1] for the line equation
        epipolar_line = np.array([line_equation[0], line_equation[1], line_equation[2]])

        # Calculate the deviation (vertical distance from the line)
        deviation = np.abs(pt2[1] - pt1[1])  # y-axis difference, deviation in the horizontal line
        epipolar_deviations.append(deviation)

        # Draw the epipolar line on the second frame
        cv2.line(frame2_resized, (int(pt2[0]), int(pt2[1])), (int(pt2[0] + 10), int(pt2[1])), (255, 0, 0), 2)

    # Calculate average deviation (for feedback)
    avg_deviation = np.mean(epipolar_deviations)
    print(f"Average Deviation: {avg_deviation} pixels")

    # Convert to RGB for Vispy
    frame1_rgb = cv2.cvtColor(frame1_resized, cv2.COLOR_BGR2RGB)
    frame2_rgb = cv2.cvtColor(frame2_resized, cv2.COLOR_BGR2RGB)

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
