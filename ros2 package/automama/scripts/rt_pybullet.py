import numpy as np
import rclpy
import pybullet as p
import pybullet_data
from rclpy.node import Node
import time


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
        rays[:, 1, :] *= self.ray_length  # Scale rays by ray_length
        rays[:, 1, :] += rays[:, 0, :]  # Offset by origin to get the end points
        return rays


class PyBulletVisualizer:
    def __init__(self):
        # Connect to PyBullet
        self.client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # Load default data path
        p.loadURDF("plane.urdf")  # Load plane

    def visualize_rays(self, rays):
        """Visualize rays in PyBullet."""
        for ray in rays:
            p.addUserDebugLine(ray[0], ray[1], lineColorRGB=[0, 0, 1], lineWidth=2)


def main():
    # Initialize ROS 2
    rclpy.init()

    # Create the ROS node
    ros_node = ROSNode(num_rays=10, ray_length=1)

    # Create the PyBullet visualizer
    pybullet_visualizer = PyBulletVisualizer()

    try:
        while rclpy.ok():
            # Process ROS events
            rclpy.spin_once(ros_node, timeout_sec=0.1)

            # Generate rays
            rays = ros_node.process()

            # Visualize rays in PyBullet
            pybullet_visualizer.visualize_rays(rays)

            # Let the PyBullet GUI update
            p.stepSimulation()

            # Optional: You can add a sleep time to control frame rate
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("Shutting down...")

    finally:
        # Cleanup PyBullet connection
        p.disconnect()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
