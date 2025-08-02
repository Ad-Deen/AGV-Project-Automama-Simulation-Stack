import numpy as np
import open3d as o3d
import cv2

def generate_radian_map(frame, hfov=1.074, vfov=0.817):
    """
    Generate a radian map where each pixel in the frame corresponds to [hfov_angle, vfov_angle],
    mapped as [x, y] = [hfov, vfov].
    """
    height, width = frame.shape
    
    hfov_step = hfov / width
    vfov_step = vfov / height
    
    radian_map = np.zeros((width, height, 2), dtype=np.float32)
    center_x = width // 2
    center_y = height // 2
    
    for y in range(height):
        for x in range(width):
            hfov_angle = (x - center_x) * hfov_step
            vfov_angle = (y - center_y) * vfov_step
            radian_map[x, y] = [hfov_angle, vfov_angle]
    
    return radian_map

def compute_ray_equation(position, orientation, radian_map, pixel_positions):
    """
    Compute the ray equations for a list of pixel positions.
    
    Parameters:
    - position: The origin point of the ray, e.g., [-0.3, 0, 0].
    - orientation: The principal axis orientation vector, e.g., [1, 0, 0].
    - radian_map: The radian map defining the direction of each pixel.
    - pixel_positions: A list of (x, y) pixel coordinates.
    
    Returns:
    - rays: A list of tuples (ray_origin, ray_direction) for each pixel position.
    """
    rays = []
    
    for x, y in pixel_positions[::50]:
        hfov_angle, vfov_angle = radian_map[x, y]
        
        # Compute the direction vector
        direction = np.array([
            orientation[0],  # Principal axis: x is unaffected
            np.tan(hfov_angle),  # Apply the hfov angle
            np.tan(vfov_angle),  # Apply the vfov angle
        ])
        
        # Normalize the direction vector
        direction = direction / np.linalg.norm(direction)
        
        # Append the ray (origin, direction)
        rays.append((position, direction))
    
    return rays

def get_pixels_in_intensity_range(image, min_intensity=100, max_intensity=200):
    """
    Get the pixel positions in the image where the intensity is within the specified range.
    
    Parameters:
    - image: Input grayscale image.
    - min_intensity: Minimum intensity value (inclusive).
    - max_intensity: Maximum intensity value (inclusive).
    
    Returns:
    - pixel_positions: A list of (x, y) pixel coordinates where intensity is within range.
    """
    # Find pixel positions that satisfy the intensity range
    positions = np.argwhere((image >= min_intensity) & (image <= max_intensity))
    return [(x, y) for y, x in positions]  # Switch order for correct indexing

def plot_rays_from_pixels(position, rays):
    """
    Plot rays using Open3D for their origin and direction vectors for specific pixels.
    
    Parameters:
    - position: The origin point of the rays (numpy array of shape (3,)).
    - rays: List of tuples (ray_origin, ray_direction) for the rays.
    """
    # Create an Open3D visualization scene
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # Create a sphere to represent the camera position (origin)
    origin_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
    origin_sphere.paint_uniform_color([1, 0, 0])  # Red color for the origin
    origin_sphere.translate(position)
    vis.add_geometry(origin_sphere)

    # Create rays and add them as lines to the visualizer
    lines = []
    points = [position]  # Include the origin for lines
    for ray_origin, ray_direction in rays:
        ray_end = ray_origin + ray_direction
        points.append(ray_end)
        lines.append([0, len(points) - 1])  # Line from origin to the endpoint

    # Create Open3D LineSet for visualization
    points = np.array(points)
    lines = np.array(lines)
    colors = [[0, 0, 1] for _ in range(len(lines))]  # Blue color for rays

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    vis.add_geometry(line_set)

    # Run the visualizer
    vis.run()
    vis.destroy_window()
# Example Usage

# Create an empty frame (480x640)
frame = np.zeros((480, 640), dtype=np.uint8)

# Generate a random grayscale image of the same size as the frame
random_greyscale_image = np.random.randint(0, 256, (480, 640), dtype=np.uint8)
cv2.imshow('rand grey scale',random_greyscale_image)

# Find pixel positions where intensity is within the range 100 to 200
pixel_positions = get_pixels_in_intensity_range(random_greyscale_image, 100, 200)
print(f'pixel position number {len(pixel_positions)}')
# print(pixel_positions)
# Generate the radian map
radian_map = generate_radian_map(frame)

# Define the ray origin and principal axis orientation
origin = np.array([-0.3, 0, 0])  # Starting point of rays
principal_axis = np.array([1, 0, 0])  # Orientation of principal axis

# Compute ray equations for the specific pixel positions
rays = compute_ray_equation(origin, principal_axis, radian_map, pixel_positions)
print(f'Ray position number {len(rays)}')
# # Plot the rays
plot_rays_from_pixels(origin, rays)
# # Wait for a key press and close the display window
# cv2.waitKey(0)
# cv2.destroyAllWindows()