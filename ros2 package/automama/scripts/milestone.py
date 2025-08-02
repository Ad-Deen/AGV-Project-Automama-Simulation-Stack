#!/usr/bin/env python3

import cupy as cp
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
# import open3d as o3d
from vispy import scene
from vispy.scene import visuals
from vispy.app import Application
import time
## Info 
# B = 0.6 m x= -0.3m to +0.3m baselength between camera frames
# size 640(W) * 480(H)
# Hfov 1.047 rad
# vfov 0.817
# f = 554.38 pixels  focal length
# Zmin= 1.039 Minimum possible depth perception
# Dmax = 319.7 Maximum possible disparity for our system

# Define CUDA kernel for stereo matching
stereo_matching_kernel = cp.RawKernel(r'''
extern "C" __global__
void compute_disparity(const float* left_image, const float* right_image, 
                       float* disparity_map, int width, int height, int max_disparity) {
    int row = blockIdx.x;  // Each block handles one row
    int col = threadIdx.x; // Each thread handles one pixel in the row

    if (row < height && col < width) {
        int best_offset = 0;
        float min_diff = 1e6;  // Initialize to a large value
        
        // Match pixel intensity along the epipolar line
        for (int d = 0; d < max_disparity; ++d) {
            int right_col = col - d;
            if (right_col >= 0) {
                float diff = fabs(left_image[row * width + col] - right_image[row * width + right_col]);
                if (diff < min_diff) {
                    min_diff = diff;
                    best_offset = d;
                }
            }
        }
        disparity_map[row * width + col] = best_offset;
    }
}
''', 'compute_disparity')

class StereoCameraViewer(Node):
    def __init__(self):
        
        super().__init__('stereo_camera_viewer')
        
        # Create subscriptions for both camera topics
        self.subscription_camera1 = self.create_subscription(
            Image,
            '/automama/camera1',
            self.camera1_callback,
            10
        )
        
        self.subscription_camera2 = self.create_subscription(
            Image,
            '/automama/camera2',
            self.camera2_callback,
            10
        )
        
        # Initialize CV Bridge
        self.bridge = CvBridge()

        # Initialize image containers
        self.frame_camera1 = None
        self.frame_camera2 = None


    def camera1_callback(self, msg):
        # Convert ROS Image message to OpenCV image
        self.frame_camera1 = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def camera2_callback(self, msg):
        # Convert ROS Image message to OpenCV image
        self.frame_camera2 = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def compute_disparity_gpu(self,left_image, right_image, max_disparity=64):
        # Convert images to GPU arrays
        left_image_gpu = cp.asarray(left_image, dtype=cp.float32)
        right_image_gpu = cp.asarray(right_image, dtype=cp.float32)
        disparity_map_gpu = cp.zeros_like(left_image_gpu, dtype=cp.float32)

        # Get image dimensions
        height, width = left_image.shape

        # Launch kernel
        block_size = 640  # One thread per pixel in a row
        grid_size = 480   # One block per row
        stereo_matching_kernel((grid_size,), (block_size,),
                            (left_image_gpu, right_image_gpu, disparity_map_gpu, 
                                width, height, max_disparity))
        
        # Transfer result back to CPU
        return cp.asnumpy(disparity_map_gpu)
    

    def get_pixels_with_disparity(self, disparity_map, row_step=20, col_step=1):
        """
        Get the pixel positions and their corresponding disparity values from the disparity map
        where the disparity value is greater than 0, while skipping rows and columns based on the given step intervals.

        Parameters:
        - disparity_map: Input disparity map as a 2D NumPy array.
        - row_step: Interval (number of rows) to skip when collecting pixel positions.
        - col_step: Interval (number of columns) to skip when collecting pixel positions in each row.

        Returns:
        - pixel_positions: A NumPy array (list of lists) of (x, y) pixel coordinates where disparity > 0
                        and only from rows and columns at the specified intervals.
        - disparity_values: A 1D NumPy array of disparity values corresponding to the pixel positions.
        """
        # Get the indices of the rows to process (with the given row step)
        rows_to_process = np.arange(0, disparity_map.shape[0], row_step)
        
        # Initialize lists to collect valid pixel positions and their disparity values
        valid_positions = []
        disparity_values = []

        # Loop through the selected rows
        for row in rows_to_process:
            # Get valid pixel positions where disparity > 0 in the selected row
            valid_cols = np.argwhere(disparity_map[row, :] > 0)
            
            # Apply column step to filter out pixels in the selected row
            cols_to_process = valid_cols[::col_step]
            
            # Collect pixel positions and disparity values
            for col in cols_to_process:
                valid_positions.append((row, col[0]))
                disparity_values.append(disparity_map[row, col[0]])
        
        return np.array(valid_positions), np.array(disparity_values)

    def generate_radian_map(self,frame_width,frame_height, hfov=1.074, vfov=0.817):
        """
        Generate a radian map where each pixel in the frame corresponds to [hfov_angle, vfov_angle],
        mapped as [x, y] = [hfov, vfov].
        """
        height = frame_height
        width = frame_width
        
        hfov_step = hfov / width
        vfov_step = vfov / height
        
        radian_map = np.zeros((width, height, 2), dtype=np.float32)
        center_x = width // 2
        center_y = height // 2
        
        for y in range(height):
            for x in range(width):
                hfov_angle = (x - center_x) * hfov_step
                vfov_angle = (y - center_y) * vfov_step
                radian_map[x, y] = [hfov_angle, vfov_angle]
        
        return radian_map
    
    def compute_ray_equation(self, position, orientation, radian_map, pixel_positions):
        """
        Compute the ray equations for a list of pixel positions.

        Parameters:
        - position: The origin point of the ray, e.g., [-0.3, 0, 0].
        - orientation: The principal axis orientation vector, e.g., [1, 0, 0].
        - radian_map: The radian map defining the direction of each pixel.
        - pixel_positions: A list of (x, y) pixel coordinates.

        Returns:
        - rays: A list of arrays where each element contains [origin, direction].
        """
        rays = []

        for x, y in pixel_positions:
            hfov_angle, vfov_angle = radian_map[y, x]

            # Compute the direction vector
            direction = np.array([
                orientation[0],  # Principal axis: x is unaffected
                np.tan(hfov_angle),  # Apply the hfov angle
                np.tan(vfov_angle),  # Apply the vfov angle
            ])

            # Normalize the direction vector
            direction = direction / np.linalg.norm(direction)

            # Stack origin and direction as a pair
            ray = np.vstack((position, direction))  # Shape (2, 3)

            rays.append(ray)

        # Convert the list of rays into a numpy array with the desired shape (n, 2, 3)
        return np.array(rays)
    
    def compute_3d_points_from_rays_and_disparities(self,rays, disparities, focal_length=554.38, base_length=0.6):
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

    def show_images(self):
        # Ensure radian_map is computed once
        radian_map = self.generate_radian_map(640, 480)
        print(f"Radian map generated: {radian_map.shape} entries")

        # Define the ray origin and principal axis orientation
        origin = np.array([0, -0.3, 0])  # Starting point of rays
        principal_axis = np.array([1, 0, 0])  # Orientation of principal axis

        # Create two separate Vispy canvases for rendering
        ray_canvas = scene.SceneCanvas(keys='interactive', size=(800, 600), title="Ray Visualization", show=True)
        ray_view = ray_canvas.central_widget.add_view()
        ray_view.camera = scene.cameras.TurntableCamera()

        disp_canvas = scene.SceneCanvas(keys='interactive', size=(640, 480), title="Disparity Map", show=True)
        disp_view = disp_canvas.central_widget.add_view()
        disp_view.camera.aspect = 1  # Maintain aspect ratio for 2D images

        # Create visuals
        ray_lines = visuals.Line()
        ray_view.add(ray_lines)

        disp_image_visual = visuals.Image()
        disp_view.add(disp_image_visual)

        app = Application()

        while rclpy.ok():
            # Spin once to handle incoming messages
            rclpy.spin_once(self, timeout_sec=0.1)

            # Check if frames are available
            if self.frame_camera1 is not None and self.frame_camera2 is not None:
                # Convert frames to grayscale and ensure float32 for GPU processing
                gray_camera1 = cv2.cvtColor(self.frame_camera1, cv2.COLOR_BGR2GRAY).astype(np.float32)
                gray_camera2 = cv2.cvtColor(self.frame_camera2, cv2.COLOR_BGR2GRAY).astype(np.float32)

                try:
                    # Compute disparity map using GPU
                    disparity_map = self.compute_disparity_gpu(gray_camera1, gray_camera2)

                    # Get pixel positions where disparity > 0
                    pixel_positions , disparity_values = self.get_pixels_with_disparity(disparity_map)

                    # Compute ray equations for the specific pixel positions
                    rays = self.compute_ray_equation(origin, principal_axis, radian_map, pixel_positions)

                    pcd = self.compute_3d_points_from_rays_and_disparities(rays,disparity_values)
                    # Set print options to show all elements
                    # np.set_printoptions(threshold=np.inf)
                    # print(f'Disparity Map: {disparity_map[:10:2]}')
                    # print(f'Pixel positions: {len(pixel_positions)}')
                    print(f'Rays length: {len(rays)}')
                    # print(f'Rays: {rays[:5]}')
                    print(f'disparities: {disparity_values[:5]}')
                    print(f'points: {pcd[:5]}')

                    # Update the rays visual
                    ray_lines.set_data(rays, connect='segments', color='blue', width=0.1)
                    ray_canvas.update()

                    # Normalize disparity map for visualization
                    disp_vis = cv2.normalize(disparity_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
                    disp_vis = disp_vis.astype(np.uint8)

                    # Update the disparity map visual
                    disp_image_visual.set_data(disp_vis)
                    disp_canvas.update()

                    # Process events for Vispy
                    app.process_events()

                except cp.cuda.memory.OutOfMemoryError as e:
                    self.get_logger().error(f"CUDA Out of Memory: {e}")
                finally:
                    # Ensure GPU memory is cleared
                    cp._default_memory_pool.free_all_blocks()
                    cp._default_pinned_memory_pool.free_all_blocks()




def main(args=None):
    rclpy.init(args=args)
    # self.Application.run()
    stereo_viewer = StereoCameraViewer()

    try:
        stereo_viewer.show_images()
    except KeyboardInterrupt:
        pass
    finally:
        stereo_viewer.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    
    main()
