"""
Client script to demonstrate the grasp prediction pipeline with all paths execution.

This script demonstrates a complete end-to-end grasp prediction workflow:
1. Initialize simulation environment with objects to grasp
2. Render camera view and generate initial point cloud
3. Allow user to select target object via mouse click
4. Call Point Cloud Cropping service to isolate the target object
5. Call Grasp Prediction service on the cropped point cloud
6. Transform predicted grasps from camera frame to robot frame
7. Call Grasp Filtering service to get kinematically valid grasps
8. Visualize the valid grasps in 3D
9. Execute ALL valid grasps in sequence, resetting state between each
10. Drop the grasped object into a tray after each grasp attempt

The pipeline integrates computer vision, machine learning, and robotics
to demonstrate autonomous object manipulation.
"""


import numpy as np
from typing import List
from scipy.spatial.transform import Rotation as R
from vis_grasps import vis_grasps_meshcat
from transform import transform_pcd_cam_to_rob
import open3d as o3d
from client import GeneralBionixClient, PointCloudData, Grasp
from img_click import ImgClick
from vis_grasps import launch_visualizer
from utils import downsample_pcd, upsample_pcd
from sim import (
    SimGrasp, 
    ObjectInfo, 
    CUBE_ORIENTATION, 
    CUBE_SCALING, 
    SPHERE_ORIENTATION, 
    SPHERE_SCALING, 
    TRAY_ORIENTATION, 
    TRAY_SCALING, 
    TRAY_POS, 
    CUBE_RGBA, 
    SPHERE_RGBA, 
    SPHERE_MASS,
    DUCK_ORIENTATION,
    DUCK_SCALING,
)
import pybullet as pb
import time

# User TODO
API_KEY = "" # Use your API key here
OS = "LINUX" # "MAC" or "LINUX"

# Define simulation objects
SIMULATION_OBJECTS = [
    ObjectInfo(
        urdf_path="cube_small.urdf",
        position=[0.35, 0.0, 0.025],
        orientation=CUBE_ORIENTATION,
        scaling=CUBE_SCALING,
        color=CUBE_RGBA
    ),
    ObjectInfo(
        urdf_path="cube_small.urdf",
        position=[0.25, 0., 0.025],
        orientation=CUBE_ORIENTATION,
        scaling=CUBE_SCALING,
        color=CUBE_RGBA
    ),
    ObjectInfo(
        urdf_path="tray/traybox.urdf",
        position=TRAY_POS,
        orientation=TRAY_ORIENTATION,
        scaling=TRAY_SCALING
    ),
    ObjectInfo(
        urdf_path="sphere2.urdf",
        position=[0.2, 0.1, 0.025],
        orientation=SPHERE_ORIENTATION,
        scaling=SPHERE_SCALING,
        color=SPHERE_RGBA,
        mass=SPHERE_MASS
    ),
    ObjectInfo(
        urdf_path="duck_vhacd.urdf",
        position=[0.3, 0.05, 0.],
        orientation=DUCK_ORIENTATION,
        scaling=DUCK_SCALING
    )
]

FREQUENCY = 30
URDF_PATH = "piper_description/urdf/piper_description_virtual_eef_free_gripper.urdf"
DOWN_SAMPLE = 4 # Don't change this

def reset_simulation(env: SimGrasp):
    """Reset the simulation environment to its initial state."""
    # Reset robot to initial position
    initial_joint_angles = [0.0] * 9  # 9 joints for the robot
    env.joint_pos(initial_joint_angles)
    
    # Wait until all joint positions match initial angles
    max_wait_time = 5.0  # Maximum time to wait in seconds
    start_time = time.time()
    while time.time() - start_time < max_wait_time:
        current_angles = [pb.getJointState(env.robot_id, i)[0] for i in range(9)]  # Get current position for each joint
        if all(abs(curr - init) < 0.01 for curr, init in zip(current_angles, initial_joint_angles)):
            break
        env.step_simulation()
    
    # Remove all objects
    for obj_id in list(env.object_ids.values()):
        env.remove_object(obj_id)
    
    # Clear the object_ids dictionary
    env.object_ids.clear()
    
    # Re-add all objects
    for obj in SIMULATION_OBJECTS:
        obj_id = env.add_object(
            obj.urdf_path, 
            obj.position, 
            obj.orientation, 
            globalScaling=obj.scaling
        )
        if obj.color is not None:
            pb.changeVisualShape(obj_id, -1, rgbaColor=obj.color)
        if obj.mass is not None:
            pb.changeDynamics(obj_id, -1, mass=obj.mass)
        
        # Store object ID with a key based on the object type
        obj_type = obj.urdf_path.split('/')[-1].split('.')[0]
        if obj_type in env.object_ids:
            # If we already have this type, add index
            env.object_ids[f"{obj_type}_{len(env.object_ids)}"] = obj_id
        else:
            env.object_ids[obj_type] = obj_id
    
    # Wait again for everything to settle
    start_time = time.time()
    while time.time() - start_time < max_wait_time:
        current_angles = [pb.getJointState(env.robot_id, i)[0] for i in range(9)]  # Get current position for each joint
        if all(abs(curr - init) < 0.01 for curr, init in zip(current_angles, initial_joint_angles)):
            break
        env.step_simulation()

def main():
    """
    Main execution function for the grasp prediction pipeline.
    
    This function orchestrates the complete workflow from scene setup
    to grasp execution, integrating multiple services and components.
    """
    
    # -------------------------------------------------------------------------
    # Step 1: Initialize Simulation Environment
    # -------------------------------------------------------------------------
    print("Initializing simulation environment...")
    env = SimGrasp(urdf_path=URDF_PATH, frequency=FREQUENCY, objects=SIMULATION_OBJECTS)
    
    # Initialize API client for grasp prediction services
    client = GeneralBionixClient(api_key=API_KEY)
    
    # Launch 3D visualizer for displaying grasps
    vis = launch_visualizer()

    # -------------------------------------------------------------------------
    # Step 2: Capture Scene and Generate Point Cloud
    # -------------------------------------------------------------------------
    print("Capturing camera view and generating point cloud...")
    # Render RGB-D image from robot's camera perspective
    color, depth, _ = env.render_camera()
    
    # Convert RGB-D image to 3D point cloud in camera coordinate frame
    pcd = env.create_pointcloud(color, depth)

    # -------------------------------------------------------------------------
    # Step 3: Interactive Object Selection
    # -------------------------------------------------------------------------
    print("\n" + "="*60)
    print("OBJECT SELECTION")
    print("="*60)
    print("Click on the object you want to grasp, then close the image window")
    
    # Launch interactive image viewer for user to select target object
    img_click = ImgClick(np.asarray(pcd.colors), os=OS)
    x, y = img_click.run()

    # Validate that user made a selection
    assert x is not None and y is not None, "No object clicked - please run again and click on an object"

    # -------------------------------------------------------------------------
    # Step 4: Point Cloud Preprocessing
    # -------------------------------------------------------------------------
    print("\nPreprocessing point cloud...")
    # Downsample point cloud to reduce computational load for cropping service
    pcd_ds = downsample_pcd(pcd, DOWN_SAMPLE)

    # -------------------------------------------------------------------------
    # Step 5: Point Cloud Cropping Service
    # -------------------------------------------------------------------------
    print("Requesting Point Cloud Cropping service...")
    
    # Call external service to crop point cloud around user-selected object
    cropped_pcd_data = client.crop_point_cloud(pcd_ds, int(x/DOWN_SAMPLE), y)

    # Convert service response back to Open3D point cloud format
    cropped_pcd_cam_frame = o3d.geometry.PointCloud()
    cropped_pcd_cam_frame.points = o3d.utility.Vector3dVector(np.array(cropped_pcd_data.points))
    cropped_pcd_cam_frame.colors = o3d.utility.Vector3dVector(np.array(cropped_pcd_data.colors))

    # Upsample cropped point cloud back to original resolution
    cropped_pcd_crop_full_cam_frame = upsample_pcd(cropped_pcd_cam_frame, pcd, DOWN_SAMPLE)

    # -------------------------------------------------------------------------
    # Step 6: Coordinate Frame Transformations
    # -------------------------------------------------------------------------
    print("Transforming point clouds to robot coordinate frame...")
    cropped_pcd_robot_frame = transform_pcd_cam_to_rob(cropped_pcd_crop_full_cam_frame)
    pcd_robot_frame = transform_pcd_cam_to_rob(pcd)
    
    cropped_pcd_data_robot_frame = PointCloudData(
        points=np.array(cropped_pcd_robot_frame.points).tolist(),
        colors=np.array(cropped_pcd_robot_frame.colors).tolist()
    )

    # -------------------------------------------------------------------------
    # Step 7: Grasp Prediction Service
    # -------------------------------------------------------------------------
    print("Requesting Grasp Prediction service...")
    grasps_response = client.predict_grasps(cropped_pcd_data_robot_frame)
    predicted_grasps_robot_frame = grasps_response.grasps

    print(f"Generated {len(predicted_grasps_robot_frame)} potential grasp candidates")

    # -------------------------------------------------------------------------
    # Step 8: Grasp Filtering Service
    # -------------------------------------------------------------------------
    print("Requesting Grasp Filtering service...")
    filter_response = client.filter_grasps(predicted_grasps_robot_frame)
    valid_grasp_idxs = filter_response.valid_grasp_idxs
    valid_grasp_joint_angles = filter_response.valid_grasp_joint_angles

    # Check if any valid grasps were found
    if not valid_grasp_idxs:
        print("No valid grasps found after filtering.")
        return

    # Extract valid grasps from the full set of predictions
    valid_grasps: List[Grasp] = [predicted_grasps_robot_frame[i] for i in valid_grasp_idxs]
    
    print(f"✅ Found {len(valid_grasps)} kinematically valid grasps")

    # -------------------------------------------------------------------------
    # Step 9: Grasp Visualization
    # -------------------------------------------------------------------------
    print("Launching 3D visualization of valid grasps...")
    print("Check the MeshCat visualizer to see the grasp poses")
    vis_grasps_meshcat(vis, valid_grasps, pcd_robot_frame)

    # -------------------------------------------------------------------------
    # Step 10: Execute All Valid Grasps
    # -------------------------------------------------------------------------
    print("\nExecuting all valid grasps in sequence...")
    
    for grasp_idx, (grasp, joint_angles) in enumerate(zip(valid_grasps, valid_grasp_joint_angles)):
        print(f"\nExecuting grasp {grasp_idx + 1} out of {len(valid_grasps)}")
        print(f"Grasp position: [{grasp.translation[0]:.3f}, {grasp.translation[1]:.3f}, {grasp.translation[2]:.3f}]")
        
        # Add visual debug marker at grasp location
        env.add_debug_point(grasp.translation)
        
        # Execute the grasp
        print("Moving robot to grasp pose and closing gripper...")
        env.grasp(joint_angles)
        
        # Transport grasped object to tray
        print("Moving grasped object to tray...")
        env.drop_object_in_tray()
        
        # Reset simulation for next grasp attempt
        print("Resetting simulation state...")
        reset_simulation(env)
        
        # Wait a short moment to observe the result before continuing
        time.sleep(10.0)  # 10 second delay between grasps

if __name__ == "__main__":
    main() 