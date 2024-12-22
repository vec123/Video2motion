import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import glob
import os
import shutil
import utils


if __name__ == "__main__":
    # Path to the .npy file
    npy_folder = glob.glob("extracted_features_video/backflip/*.npy")
    for npy_file in npy_folder:
        #npy_file = "extracted_features_video/slack_long_landmarkPositions.npy"  # Replace with actual path
        npy_name = npy_file.split("\\")[-1].split(".")[0]
        video_name = npy_name.split('_landmarkPositions')[0]

        output_dir = "skeleton_data/" + video_name
        os.makedirs(output_dir, exist_ok=True)

        utils.find_and_copy_file("../Kinetics_dataset/backflip (human)", output_dir, video_name)
        utils.find_and_copy_file("extracted_features_video", output_dir, npy_name + ".npy")
        joint_positions = np.load(npy_file)
        utils.graficasEnCarpetas(joint_positions,output_dir + "/timeseries_plots")
        # Visualize skeleton motion
        #visualize_skeleton_motion(joint_positions, skeleton_edges)

        # Convert 3D joint positions to 2D
        joint_positions_2d = joint_positions[:, :, :2]

        # Path to a video frame for background (optional)
        # frame_path = "frame.jpg"  # Replace with the path to the extracted frame

        # Visualize skeleton motion in 2D
        utils.visualize_skeleton_motion_2d(joint_positions_2d, utils.skeleton_edges, 
                                     image_path=None, output_location=output_dir + "/skeleton.gif" )