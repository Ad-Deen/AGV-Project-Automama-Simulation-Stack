import numpy as np
import cv2

def generate_radian_map(frame, hfov=1.074, vfov=0.817):
    """
    Generate a radian map where each pixel in the frame corresponds to [hfov_angle, vfov_angle],
    mapped as [x, y] = [hfov, vfov].
    
    Parameters:
    - frame: Input frame (e.g., image) of shape (height, width).
    - hfov: Horizontal field of view in radians.
    - vfov: Vertical field of view in radians.
    
    Returns:
    - radian_map: A numpy array of shape (width, height, 2), where each element
                  is [hfov_angle, vfov_angle] for the corresponding pixel.
    """
    height, width = frame.shape
    
    # Calculate step sizes for HFOV and VFOV per pixel
    hfov_step = hfov / width
    vfov_step = vfov / height
    
    # Initialize an empty array for the radian map with swapped dimensions
    radian_map = np.zeros((width, height, 2), dtype=np.float32)
    
    # Calculate the center pixel
    center_x = width // 2
    center_y = height // 2
    
    # Populate the radian map with the corresponding angles for each pixel
    for y in range(height):
        for x in range(width):
            # Map the pixel position to the radian values
            hfov_angle = (x - center_x) * hfov_step  # Horizontal angle (left-right)
            vfov_angle = (y - center_y) * vfov_step  # Vertical angle (top-bottom)
            radian_map[x, y] = [hfov_angle, vfov_angle]  # Store the result with [x, y] mapping
    
    return radian_map

def display_radian_map(radian_map):
    """
    Function to visualize the radian map (only displaying the first channel for clarity).
    
    Parameters:
    - radian_map: The 3D numpy array of radian values (indexed as [x, y]).
    """
    # Transpose the HFOV channel for visualization as the image is typically indexed [y, x]
    hfov_map = radian_map[:, :, 0].T  # Transpose to [y, x]
    
    # Normalize the values for visualization (scale between 0 and 255)
    hfov_map_vis = cv2.normalize(hfov_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Display the HFOV map
    cv2.imshow('HFOV Radian Map', hfov_map_vis)


# Create an empty frame (480x640)
frame = np.zeros((480, 640), dtype=np.uint8)  # Example: An empty frame

# Generate the radian map for the frame
radian_map = generate_radian_map(frame)
print(radian_map)

# Example: Accessing radian values for the pixel at position (600, 470)
hfov_angle, vfov_angle = radian_map[620, 475]
print(f"HFOV angle at (600, 470): {hfov_angle:.3f} rad, VFOV angle: {vfov_angle:.3f} rad")

# Display the radian map (HFOV angles)
display_radian_map(radian_map)

# Wait for a key press and close the display window
cv2.waitKey(0)
cv2.destroyAllWindows()
