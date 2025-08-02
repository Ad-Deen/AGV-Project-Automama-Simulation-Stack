import numpy as np
from vispy import app, scene

def visualize_3d_points(points):
    """
    Visualizes 3D points using VisPy.

    Parameters:
    - points: A NumPy array of shape (N, 3) where each row is a 3D point [x, y, z].
    """
    # Create the VisPy canvas
    canvas = scene.SceneCanvas(keys='interactive', show=True)
    view = canvas.central_widget.add_view()

    # Create a Markers visual for the points in 3D
    markers = scene.visuals.Markers()

    # Set data for the markers (points)
    markers.set_data(
        pos=points,  # Points data
        size=10,  # Point size
        face_color='blue',  # Points color
        edge_color='blue',  # Border color
    )

    # Set the camera for 3D view
    view.camera = scene.cameras.TurntableCamera(fov=60, azimuth=90, elevation=30, distance=10)

    # Start the VisPy app loop
    app.run()

# Example: Pass your points to visualize (e.g., from rays and disparity calculation)
points = np.array([
    [1.0, 2.0, 3.0],
    [2.0, 3.0, 4.0],
    [3.0, 4.0, 5.0],
    [4.0, 5.0, 6.0],
    [5.0, 6.0, 7.0]
])

# Call the function with the points
visualize_3d_points(points)
