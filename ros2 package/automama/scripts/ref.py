import numpy as np
from vispy import app, scene

def compute_3d_points_from_rays_and_disparities(rays, disparities, focal_length=554.38, base_length=0.6):
    """
    Compute the 3D point cloud from the rays and disparity values.

    Parameters:
    - rays: A list of direction vectors in the format [[origin], [direction]]
    - disparities: A list of disparity values corresponding to each ray.
    - focal_length: The focal length of the camera (in pixels).
    - base_length: The base length of the stereo camera (in meters).

    Returns:
    - points: A list of 3D coordinates [x, y, z].
    """
    points = []

    # Loop through each ray and disparity value
    for ray, disparity in zip(rays, disparities):
        # Extract the direction vector (ignoring the origin)
        direction = np.array(ray[1])  # Convert direction to numpy array
        
        # Compute the depth using the disparity
        depth = (focal_length * base_length) / disparity
        
        # Compute the 3D coordinates
        point = direction * depth  # Multiply direction by depth
        
        points.append(point)

    return np.array(points)


# Example rays and disparities
rays = [
    [[0., -0.3, 0.], [0.85783556, -0.35530918, -0.37131326]],
    [[0., -0.3, 0.], [0.85834849, -0.3538353, -0.37153528]],
    [[0., -0.3, 0.], [0.85885919, -0.35236079, -0.37175633]],
    [[0., -0.3, 0.], [0.85936765, -0.35088571, -0.37197642]],
    [[0., -0.3, 0.], [0.8598739, -0.34941, -0.37219555]]
]

disparities = [15., 16., 57., 58., 59.]

# Compute the 3D points
points = compute_3d_points_from_rays_and_disparities(rays, disparities)
print(points)
