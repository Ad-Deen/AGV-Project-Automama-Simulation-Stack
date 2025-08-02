import numpy as np
import time
from vispy import scene
from vispy.scene import visuals
from vispy.app import Application

# Create a Vispy canvas for rendering
canvas = scene.SceneCanvas(keys='interactive', size=(800, 600), show=True)
view = canvas.central_widget.add_view()

# Number of rays and ray length
num_rays = 10
ray_length = 1  # Length of each ray

# Create a line visual for the rays
lines = visuals.Line()
view.add(lines)

# Set the camera for 3D view
view.camera = scene.cameras.TurntableCamera()

# Function to generate random rays with fixed origin and random directions
def generate_rays(num_rays, ray_length):
    rays = np.zeros((num_rays, 2, 3))  # Two points for each ray: origin and direction
    rays[:, 1, :] = np.random.rand(num_rays, 3) * 2 - 1  # Random directions in [-1, 1] for each axis
    print(rays)
    return rays

app = Application()



# Run the update function manually in an infinite loop with a delay
while True:
    # Get the new rays from the generate_rays function
    rays = generate_rays(num_rays, ray_length)
    
    # Update the lines visual with the new ray data
    lines.set_data(rays, connect='segments', color='blue', width=2)
    
    # Redraw the canvas
    canvas.update()
    
    # Process events for Vispy
    app.process_events()  # Allows Vispy to process events like user input

    # Add a delay to control the update rate (e.g., 0.1 seconds)
    time.sleep(0.1)  # 100ms delay for updating rays
