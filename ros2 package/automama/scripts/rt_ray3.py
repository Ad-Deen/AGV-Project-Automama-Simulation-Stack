import numpy as np
import rclpy
from vispy import scene
from vispy.scene import visuals
from vispy.app import Application
from rclpy.node import Node
import time



class VispyVisualizer:
    def __init__(self):
        # Initialize Vispy canvas
        self.canvas = scene.SceneCanvas(keys='interactive', size=(800, 600), show=True)
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.cameras.TurntableCamera()

        # Create a line visual for the rays
        self.lines = visuals.Line()
        self.view.add(self.lines)

        # Vispy application
        self.app = Application()

    def process_events(self,rays):
        """Process Vispy events."""
        self.lines.set_data(rays, connect='segments', color='blue', width=2)
        self.canvas.update()
        self.app.process_events()


class ROSNode(Node):
    def __init__(self, num_rays=10, ray_length=1):
        super().__init__('ray_visualizer')
        
        self.num_rays = num_rays
        self.ray_length = ray_length

    def process(self):
        """Custom processing for ROS + Visualization."""
        # Generate rays
        rays = np.zeros((self.num_rays, 2, 3))  # Two points for each ray: origin and direction
        rays[:, 1, :] = np.random.rand(self.num_rays, 3) * 2 - 1  # Random directions in [-1, 1]
        # Update the visualizer with the new rays
        return rays


def main():
    # Initialize ROS 2
    rclpy.init()

    # Create the Vispy visualizer
    visualizer = VispyVisualizer()

    # Create the ROS node
    ros_node = ROSNode(num_rays=10, ray_length=1)

    # Main loop
    try:
        while rclpy.ok():
            # Process ROS events
            rclpy.spin_once(ros_node, timeout_sec=0.1)

            # Update the visualization
            rays = ros_node.process()

            # Process Vispy events
            visualizer.process_events(rays)

            # Allow a slight delay to control the update rate
            # time.sleep(0.1)

    except KeyboardInterrupt:
        print("Shutting down...")

    finally:
        # Cleanup
        rclpy.shutdown()


if __name__ == '__main__':
    main()
