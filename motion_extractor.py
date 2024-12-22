import os
import mediapipe as mp
import cv2
import numpy as np
import matplotlib.pyplot as plt
from time import time
import imageio
import glob
from tqdm import tqdm
import os
import video2motion_helpers
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
import utils.general_helper as general_helpers

if __name__ == "__main__":
    inicio = time()
    
    # Get list of video filenames from the "videos_trickline" directory
    #video_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'videos_trickline'))

   # import ipdb;ipdb.set_trace()
    video_folder = "../Data_sets/Kinetics_video_movement/backflip (human)"
    #video_folder = "motion_extractions_2/backflip"
    #extracted_features_folder = "extracted_features/backflip"
    output_folder_1 = "motion_extractions_2/backflip_nominal_mediapipe"
    output_folder_2 = "motion_extractions_2/backflip_adapted_mediapipe"
    output_folder_3 = "motion_extractions_2/backflip_openpose"
    output_folders = [output_folder_1, output_folder_2, output_folder_3]
    for output_folder in output_folders:
        general_helpers.recursive_mkdir(output_folder)
    
    video_files = glob.glob(os.path.join(video_folder, "**/*.mp4"), recursive=True)
   # import ipdb;ipdb.set_trace()

    for index,video_file in enumerate( tqdm(video_files) ):
        if index >= 20:
            break
    #for i in range(1):
        #video_file = "slack_long.mp4"
        print("Procesando video:", video_file)
        video_name = os.path.splitext(os.path.basename(video_file))[0]
        joint_positions_1, fpss, ancho, alto = video2motion_helpers.coorVideoMoveNet(video_file)
        joint_positions_2, fpss, ancho, alto = video2motion_helpers.coorVideoExtremeMotion(video_file)
        joint_positions_3, fpss, ancho, alto = video2motion_helpers.coorVideoExtremeMotion(video_file)
        joint_positions = [joint_positions_1, joint_positions_2, joint_positions_3]
        for index, joint_positions in enumerate(joint_positions):
            output_folder = output_folders[index]
            joint_positions, joint_edges = video2motion_helpers.filter_joint_data(joint_positions, video2motion_helpers.movenet_edges)

            #------Save the landmark positions for each video separately
            output_folder_this_feature = os.path.join(output_folder, video_name)
            general_helpers.recursive_mkdir(output_folder_this_feature)
            np.save(os.path.join(output_folder_this_feature, "landmarkPositions.npy"), joint_positions)

            #------Plot timeseries of each joint
            time_series_folder = os.path.join(output_folder_this_feature, "timeseries_plots")
            general_helpers.recursive_mkdir(time_series_folder)
            video2motion_helpers.graficasEnCarpetas(joint_positions, time_series_folder)

            #------Find and copy the original video to the output folder
            video2motion_helpers.find_and_copy_file(video_folder, output_folder_this_feature, video_name)

            #------Create and save a GIF of the skeleton
            joint_positions_2d = joint_positions[:, :, :2]
            print(joint_positions_2d.shape)
            video2motion_helpers.visualize_skeleton_motion_2d(joint_positions_2d, joint_edges, 
                                        image_path=None, output_location=output_folder_this_feature + "/skeleton.gif" )
    fin = time()
    print(f"Ejecución finalizada en: {fin-inicio} segundos")