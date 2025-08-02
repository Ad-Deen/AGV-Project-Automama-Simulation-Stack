import numpy as np
import time
from vispy import scene
from vispy.scene import visuals

# Create a Vispy canvas for rendering
canvas = scene.SceneCanvas(keys='interactive', size=(800, 600), show=True)
view = canvas.central_widget.add_view()

# Initialize points with some data (this prevents the camera from failing)
initial_points = np.random.rand(50, 3)
scatter = visuals.Markers()
scatter.set_data(initial_points, face_color='green', size=10)
view.add(scatter)

# Set the camera for 3D view
view.camera = scene.cameras.TurntableCamera()

# Start the VisPy application event loop in a separate thread
app = canvas.app

# Infinite loop for updating points
while True:
    # Generate new random points
    new_points = np.random.rand(50, 3)
    print(new_points[:5])

    # Update the scatter visual with the new points
    scatter.set_data(new_points, face_color='green', size=10)

    # Process events to keep the canvas interactive
    app.process_events()

    # Add a delay for the update frequency (e.g., 0.1 seconds)
    time.sleep(0.1)
