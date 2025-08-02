import numpy as np
import cv2 as cv
import glob
import pickle
import os

# Define checkerboard properties
CHECKERBOARD = (8, 8)  # (Width, Height) of the internal corners
SQUARE_SIZE = 8  # Size of one square in mm (Adjust based on your checkerboard)
FRAME_SIZE = (640, 480)  # Frame resolution

# Define criteria for subpixel refinement
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points (3D points in real-world coordinates)
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * SQUARE_SIZE
print(objp)
# Lists to store object points and image points
objpoints = []  # 3D points
imgpoints = []  # 2D points

# Load all calibration images
image_dir = os.path.expanduser("~/callib_pics/")  # Directory containing checkerboard images
image_files = glob.glob(os.path.join(image_dir, "*.png"))

if not image_files:
    print("No calibration images found! Check the image directory path.")
    exit()

print(f"Found {len(image_files)} images for calibration.")

# Process each image
for img_file in image_files:
    img = cv.imread(img_file)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find chessboard corners
    ret, corners = cv.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret:
        objpoints.append(objp)
        
        # Refine corner positions
        corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)

        # Draw corners and display for verification
        cv.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        cv.imshow("Chessboard Detection", img)
        cv.waitKey(500)  # Show for 500ms
    else:
        print(f"Chessboard not found in {img_file}")

cv.destroyAllWindows()

# Perform camera calibration
if len(objpoints) > 0:
    print(f"Calibrating using {len(objpoints)} valid images...")
    
    ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(
        objpoints, imgpoints, FRAME_SIZE, None, None
    )

    print("\n=== Camera Calibration Results ===")
    print("Camera Matrix:\n", cameraMatrix)
    print("Distortion Coefficients:\n", dist)

    # Save calibration results
    with open("camera_calibration.pkl", "wb") as f:
        pickle.dump((cameraMatrix, dist), f)

    print("\nCalibration results saved as 'camera_calibration.pkl'")
else:
    print("Error: No valid chessboard detections. Check your images.")

# Test undistortion with an image
test_img = cv.imread(image_files[0])  # Use the first calibration image
h, w = test_img.shape[:2]
newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, dist, (w, h), 1, (w, h))

# Undistort using remapping
mapx, mapy = cv.initUndistortRectifyMap(cameraMatrix, dist, None, newCameraMatrix, (w, h), 5)
undistorted_img = cv.remap(test_img, mapx, mapy, cv.INTER_LINEAR)

# Crop and save the undistorted image
x, y, w, h = roi
undistorted_img = undistorted_img[y:y+h, x:x+w]
cv.imwrite("undistorted_result.png", undistorted_img)
cv.imshow("Undistorted Image", undistorted_img)
cv.waitKey(0)
cv.destroyAllWindows()
