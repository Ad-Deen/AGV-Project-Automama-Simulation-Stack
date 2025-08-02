import numpy as np
from vispy import scene
from vispy.scene import visuals
from vispy.app import Timer

# Create a Vispy canvas for rendering
canvas = scene.SceneCanvas(keys='interactive', size=(800, 600), show=True)
view = canvas.central_widget.add_view()

# Initialize points
initial_points = np.random.rand(50, 3)

# Create the scatter plot for the point cloud and initialize with random points
scatter = visuals.Markers()
scatter.set_data(initial_points, face_color='green', size=10)

# Add the scatter plot to the view
view.add(scatter)

# Set the camera for 3D view
view.camera = scene.cameras.TurntableCamera()

# Timer function to update the points in real-time
def update_points(event):
    # Generate new random points
    new_points = np.random.rand(50, 3)
    print(new_points[:10])
    
    # Update the scatter plot data with the new points
    scatter.set_data(new_points, face_color='green', size=10)

# Create a timer that calls the update_points function every 100ms
timer = Timer(interval=0.1, connect=update_points, start=True)

# Run the event loop
canvas.app.run()
