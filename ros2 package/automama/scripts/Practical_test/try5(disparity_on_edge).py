import cv2
import numpy as np
from vispy import scene, app
import cupy as cp
import cvcuda
'''
This script is to extension of simple dual camera vision to finding the cost function of the pixel around its neighboring proximity
'''
stereo_matching_kernel = cp.RawKernel(r'''
extern "C" __global__
void compute_disparity(const float* left_image, const float* right_image, 
                       float* disparity_map, int width, int height, int max_disparity) {
    int row = blockIdx.x;  // Each block handles one row
    int col = threadIdx.x; // Each thread handles one pixel in the row

    const int WINDOW_SIZE = 10; // Number of pixels in the window

    if (row < height && col < width - WINDOW_SIZE) {
        int best_offset = 0;
        float min_sad = 1e6;  // Initialize to a large value

        // Search along epipolar line for the best disparity
        for (int d = 0; d < max_disparity; ++d) {
            int right_col = col - d;

            if (right_col >= 0 && right_col + WINDOW_SIZE < width) {  // Ensure within bounds
                float sad = 0.0f;

                // Compute sum of absolute differences (SAD) over the window
                for (int i = 0; i < WINDOW_SIZE; ++i) {
                    float left_pixel = left_image[row * width + (col + i)];
                    float right_pixel = right_image[row * width + (right_col + i)];
                    sad += fabs(left_pixel - right_pixel);
                }

                // Find the disparity with the lowest SAD
                if (sad < min_sad) {
                    min_sad = sad;
                    best_offset = d;
                }
            }
        }
        
        // Store disparity value
        disparity_map[row * width + col] = best_offset;
    }
}
''', 'compute_disparity')

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
    edges_r = cv2.Canny(R, 150, 200)
    edges_g = cv2.Canny(G, 150, 200)
    edges_b = cv2.Canny(B, 150, 200)
    
    # Combine the edge maps from all three channels
    combined_edge_map = cv2.bitwise_or(edges_r, edges_g)
    combined_edge_map = cv2.bitwise_or(combined_edge_map, edges_b)
    
    # Return the combined edge map
    return combined_edge_map

def compute_disparity_map(left_image, right_image, min_disp=0, num_disp=16, block_size=15):
    """
    Computes the disparity map from two images (left and right) based on edge detection.
    
    Parameters:
    - left_image: The edge-detected black & white image from the left camera (640x480).
    - right_image: The edge-detected black & white image from the right camera (640x480).
    - min_disp: Minimum disparity (should be multiple of 16).
    - num_disp: Number of disparities (should be multiple of 16).
    - block_size: Block size for StereoBM (usually an odd number).
    
    Returns:
    - disparity_map: Computed disparity map (640x480).
    """
    # Ensure both images are grayscale and have the same size (edge-detected black and white images)
    if len(left_image.shape) > 2:
        left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
    if len(right_image.shape) > 2:
        right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)
    
    # Apply the Canny edge detector if not already done (ensure they are edge maps)
    # left_edges = cv2.Canny(left_image, 100, 200)
    # right_edges = cv2.Canny(right_image, 100, 200)

    # Create StereoBM object for disparity calculation
    stereo = cv2.StereoBM_create(numDisparities=num_disp, blockSize=block_size)
    
    # Compute the disparity map
    disparity_map = stereo.compute(left_image, right_image).astype(np.float32) / 16.0
    
    # Normalize the disparity map for better visualization
    disparity_map = np.uint8(cv2.normalize(disparity_map, None, 0, 255, cv2.NORM_MINMAX))
    
    return disparity_map

def compute_disparity_gpu(left_image, right_image, max_disparity=64):
    # Convert images to GPU arrays
    left_image_gpu = cp.asarray(left_image, dtype=cp.float32)
    right_image_gpu = cp.asarray(right_image, dtype=cp.float32)
    disparity_map_gpu = cp.zeros_like(left_image_gpu, dtype=cp.float32)

    # Get image dimensions
    height, width = left_image.shape

    # Launch kernel
    block_size = 640  # One thread per pixel in a row
    grid_size = 480   # One block per row
    stereo_matching_kernel((grid_size,), (block_size,),
                        (left_image_gpu, right_image_gpu, disparity_map_gpu, 
                            width, height, max_disparity))
    # Transfer result back to CPU
    return cp.asnumpy(disparity_map_gpu)

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
canvas3 = scene.SceneCanvas(keys='interactive', size=(640, 480), title="Disparity", show=True)

view1 = canvas1.central_widget.add_view()
view2 = canvas2.central_widget.add_view()
view3 = canvas3.central_widget.add_view()

view1.camera.aspect = 1
view2.camera.aspect = 1
view3.camera.aspect = 1

image_visual1 = scene.visuals.Image(np.zeros((480, 640, 3), dtype=np.uint8))
image_visual2 = scene.visuals.Image(np.zeros((480, 640, 3), dtype=np.uint8))
image_visual3 = scene.visuals.Image(np.zeros((480, 640, 3), dtype=np.uint8))

view1.add(image_visual1)
view2.add(image_visual2)
view3.add(image_visual3)

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

    disparity_map = compute_disparity_gpu(edges1, edges2)
    disp_vis = cv2.normalize(disparity_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    disp_vis = disp_vis.astype(np.uint8)

    # Update images
    image_visual1.set_data(edges1)
    image_visual2.set_data(edges2)
    image_visual3.set_data(disp_vis)

    canvas1.update()
    canvas2.update()
    canvas3.update()

timer = app.Timer(interval=1/30)  # Run at 30 FPS
timer.connect(update)
timer.start()

app.run()

# Release resources
cap1.release()
cap2.release()
