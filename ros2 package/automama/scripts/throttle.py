#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


class ForwardThrottlePublisher(Node):
    def __init__(self):
        super().__init__('forward_throttle_publisher')

        # Create a publisher for the /automama/cmd_vel topic
        self.publisher = self.create_publisher(Twist, '/automama/cmd_vel', 10)

        # Set the throttle value
        self.throttle_value = -1.0

        # Create a timer to publish messages at a fixed rate (10 Hz)
        self.timer = self.create_timer(0.1, self.publish_throttle_command)

    def publish_throttle_command(self):
        # Create a Twist message
        cmd = Twist()
        cmd.linear.x = self.throttle_value  # Forward throttle
        cmd.linear.y = 0.0                 # No lateral movement
        cmd.linear.z = 0.0                 # No vertical movement
        cmd.angular.x = 0.0                # No roll
        cmd.angular.y = 0.0                # No pitch
        cmd.angular.z = 0.0                # No yaw

        # Publish the message
        self.publisher.publish(cmd)
        self.get_logger().info(f'Publishing forward throttle command: {cmd.linear.x}')


def main(args=None):
    rclpy.init(args=args)
    forward_throttle_publisher = ForwardThrottlePublisher()

    try:
        rclpy.spin(forward_throttle_publisher)
    except KeyboardInterrupt:
        pass
    finally:
        forward_throttle_publisher.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
