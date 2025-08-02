import numpy as np
from vispy import scene
from vispy.scene import visuals
from vispy.app import Timer

# Create a Vispy canvas for rendering
canvas = scene.SceneCanvas(keys='interactive', size=(800, 600), show=True)
view = canvas.central_widget.add_view()

# Number of rays and ray length
num_rays = 10
ray_length = 1  # Length of each ray

# Create the scatter plot for the origin points (all at [0,0,0])
# scatter = visuals.Markers()
# scatter.set_data(np.zeros((num_rays, 3)), face_color='red', size=10)  # All at origin
# view.add(scatter)

# Create a line visual for the rays
lines = visuals.Line()
view.add(lines)

# Set the camera for 3D view
view.camera = scene.cameras.TurntableCamera()

# Function to generate random rays with fixed origin and random directions
def generate_rays(num_rays, ray_length):
    rays = np.zeros((num_rays, 2, 3))  # Two points for each ray: origin and direction
    rays[:, 1, :] = np.random.rand(num_rays, 3) * 2 - 1  # Random directions in [-1, 1] for each axis
    # rays[:, 1, :] = rays[:, 1, :] / np.linalg.norm(rays[:, 1, :], axis=1)[:, None]  # Normalize direction
    # rays[:, 1, :] *= ray_length  # Define the endpoint based on ray length
    # print(type(rays))
    return rays

# Timer function to update the rays in real-time (just to simulate)
def update_rays(event):
    # Get the new rays from the generate_rays function
    rays = generate_rays(num_rays, ray_length)
    
    # Update the lines visual with the new ray data
    lines.set_data(rays, connect='segments', color='blue', width=2)

# Create a timer that calls the update_rays function every 100ms
timer = Timer(interval=0.1, connect=update_rays, start=True)
# update_rays()
# Run the event loop
canvas.app.run()

