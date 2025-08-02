import numpy as np
import rclpy
from rclpy.node import Node
import open3d as o3d


class Open3DVisualizer:
    def __init__(self):
        # Initialize Open3D visualizer
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name="Open3D Ray Visualizer", width=800, height=600)
        self.line_set = o3d.geometry.LineSet()  # Create a LineSet object
        self.vis.add_geometry(self.line_set)

    def process_events(self, rays):
        """Update Open3D visualization with ray data."""
        # Convert rays to Open3D LineSet
        points = rays.reshape(-1, 3)  # Flatten rays into individual points
        lines = [[i * 2, i * 2 + 1] for i in range(len(rays))]  # Connect origin to direction
        colors = [[0, 0, 1] for _ in lines]  # Set color to blue

        # Update LineSet geometry
        self.line_set.clear()
        self.line_set.points = o3d.utility.Vector3dVector(points)
        self.line_set.lines = o3d.utility.Vector2iVector(lines)
        self.line_set.colors = o3d.utility.Vector3dVector(colors)

        # Render the scene
        self.vis.update_geometry(self.line_set)
        self.vis.poll_events()
        self.vis.update_renderer()


class ROSNode(Node):
    def __init__(self, visualizer, num_rays=10, ray_length=1.0):
        super().__init__('ray_visualizer')
        self.visualizer = visualizer
        self.num_rays = num_rays
        self.ray_length = ray_length

    def process(self):
        """Generate rays with origin and direction."""
        # Rays with origin at (0, 0, 0) and random directions
        rays = np.zeros((self.num_rays, 2, 3))  # Two points per ray: origin and direction
        directions = np.random.rand(self.num_rays, 3) * 2 - 1  # Random directions in [-1, 1]
        directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)  # Normalize
        rays[:, 1, :] = directions * self.ray_length  # Scale by ray_length
        return rays


def main():
    # Initialize ROS 2
    rclpy.init()

    # Create the Open3D visualizer
    visualizer = Open3DVisualizer()

    # Create the ROS node
    ros_node = ROSNode(visualizer, num_rays=10, ray_length=1.0)

    # Main loop
    try:
        while rclpy.ok():
            # Process ROS events
            rclpy.spin_once(ros_node, timeout_sec=0.1)

            # Update the visualization
            rays = ros_node.process()

            # Update Open3D visualization
            visualizer.process_events(rays)

    except KeyboardInterrupt:
        print("Shutting down...")

    finally:
        # Cleanup
        rclpy.shutdown()
        visualizer.vis.destroy_window()


if __name__ == '__main__':
    main()
