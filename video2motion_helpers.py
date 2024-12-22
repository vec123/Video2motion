
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import shutil
import mediapipe as mp
import cv2
import numpy as np
import matplotlib.pyplot as plt
import imageio
import os
import torch
from pykalman import KalmanFilter
import tensorflow as tf
import tensorflow_hub as hub

# Define the joint names and their connections
partes_cuerpo = {
    0: "Nariz", 1: "Ojo Izquierdo Interno", 2: "Ojo Izquierdo", 3: "Ojo Izquierdo Externo",
    4: "Ojo Derecho Interno", 5: "Ojo Derecho", 6: "Ojo Derecho Externo", 7: "Oreja Izquierda",
    8: "Oreja Derecha", 9: "Boca Izquierda", 10: "Boca Derecha", 11: "Hombro Izquierdo",
    12: "Hombro Derecho", 13: "Codo Izquierdo", 14: "Codo Derecho", 15: "Muñeca Izquierda",
    16: "Muñeca Derecha", 17: "Meñique Izquierdo", 18: "Meñique Derecho", 19: "Índice Izquierdo",
    20: "Índice Derecho", 21: "Pulgar Izquierdo", 22: "Pulgar Derecho", 23: "Cadera Izquierda",
    24: "Cadera Derecha", 25: "Rodilla Izquierda", 26: "Rodilla Derecha", 27: "Tobillo Izquierdo",
    28: "Tobillo Derecho", 29: "Talón Izquierdo", 30: "Talón Derecho", 31: "Índice del Pie Izquierdo",
    32: "Índice del Pie Derecho"
}
(['root', 
  'lhipjoint', 
  'lfemur', 
  'ltibia', 
  'lfoot', 
  'ltoes', 
  'rhipjoint',
    'rfemur', 
    'rtibia', 'rfoot', 'rtoes', 'lowerback', 'upperback', 'thorax', 'lowerneck', 'upperneck', 'head', 'lclavicle', 'lhumerus', 'lradius', 'lwrist', 'lhand', 'lfingers', 'lthumb', 'rclavicle', 'rhumerus', 'rradius', 'rwrist', 'rhand', 'rfingers',
   'rthumb'])
skeleton_edges = [
    (30, 26), (29, 25), #Talon - rodilla
    (26,24), (25,23),  # rodilla - cadera
    (24,25),  # cadera - cadera
    (24,12), (23,11),  # cadera - hombros
    (12,11),  # hombros -hombros
    (12,14), (11,13),  # hombros -codo
    (14,16), (13,15)  # codo-muneca
]

# MoveNet keypoints
movenet_keypoints = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# MoveNet edges
movenet_edges = [
    (0, 1),  # Nose ↔ Left Eye
    (0, 2),  # Nose ↔ Right Eye
    (1, 3),  # Left Eye ↔ Left Ear
    (2, 4),  # Right Eye ↔ Right Ear
    (5, 6),  # Left Shoulder ↔ Right Shoulder
    (5, 7),  # Left Shoulder ↔ Left Elbow
    (6, 8),  # Right Shoulder ↔ Right Elbow
    (7, 9),  # Left Elbow ↔ Left Wrist
    (8, 10), # Right Elbow ↔ Right Wrist
    (5, 11), # Left Shoulder ↔ Left Hip
    (6, 12), # Right Shoulder ↔ Right Hip
    (11, 12),# Left Hip ↔ Right Hip
    (11, 13),# Left Hip ↔ Left Knee
    (12, 14),# Right Hip ↔ Right Knee
    (13, 15),# Left Knee ↔ Left Ankle
    (14, 16) # Right Knee ↔ Right Ankle
]


# Define skeleton connections (edges between joints)
#skeleton_edges = [
#    (11, 13), (13, 15),  # Left arm
#    (12, 14), (14, 16),  # Right arm
#    (23, 25), (25, 27),  # Left leg
#    (24, 26), (26, 28),  # Right leg
#    (11, 12), (23, 24),  # Shoulders and hips
#    (11, 23), (12, 24),  # Torso connections
#    (0, 11), (0, 12)     # Head to shoulders
#]

def filter_joint_data(joint_data, skeleton_edges):
    unique_joints = set([joint for edge in skeleton_edges for joint in edge])
    unique_joints = sorted(unique_joints)
    joint_mapping = {original: new for new, original in enumerate(unique_joints)}
    filtered_joint_data = joint_data[:, unique_joints, :]  # Shape: (time, len(unique_joints), 3)
    remapped_skeleton_edges = [(joint_mapping[edge[0]], joint_mapping[edge[1]]) for edge in skeleton_edges]
    return filtered_joint_data, remapped_skeleton_edges

def compute_torso_center(joint_data):
    # Extract the joint positions for the torso joints
    torso_joints = [0, 11, 12, 23, 24]
    torso_joint_data = joint_data[:, torso_joints, :]  # Shape: (time, len(torso_joints), 3)

    # Compute the center of the torso as the mean position of the torso joints
    center = np.mean(torso_joint_data, axis=1)  # Shape: (time, 3)
    return center

def compute_relative_positions(joint_data, center):
    # Subtract the center from all joint positions to make the torso center the origin
    relative_positions = joint_data - center[:, np.newaxis, :]  # Shape: (time, num_joints, 3)
    return relative_positions

def combine_data_torsocenter(joint_data, center):
    data = np.concatenate([joint_data, center[:, np.newaxis, :]], axis=1)
    return data
          
def coorVideoConVideoChatGpt(video_path, show_video=False):
    """Devuelve un tensor con los fotogramas y en cada fotograma, la matriz con las coordenadas de los puntos

    Args:
        nombre_video (_type_): Video en formato .mp4

    Returns:
        (np.array, int, int, int): Tensor de dimension (len_fotogramas_video, 33, 3) y los fps del video, ancho y alto
    """
    print("-"*100)
    print("-"*100)

    # Inicializar MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7)

    # Cargar el video
    cap = cv2.VideoCapture(video_path)

    # Obtiene el fps del video. Redondea hacia abajo
    fps_video = int(cap.get(cv2.CAP_PROP_FPS))
    print(f"El vídeo tiene {fps_video} FPSs.")
    if fps_video < 30:
        print("El video tiene menos de 30 FPSs. Se ajustará a 30 FPSs.")
    new_fps = 60  # Target FPS
    cap.set(cv2.CAP_PROP_FPS, new_fps)
    
    # Obtener las dimensiones del video
    ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    tensor = []
    # Iterar sobre cada fotograma del video
    while cap.isOpened():
        ret, frame = cap.read()
      
        if not ret:
            break

        frame = cv2.GaussianBlur(frame, (5, 5), 0) #denoise
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        frame = cv2.filter2D(frame, -1, kernel) #sharpen edges

        # Convertir la imagen a RGB y procesarla con MediaPipe Pose
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        matriz = []
        # Dibujar los landmarks si están disponibles
        if results.pose_landmarks:
            for landmark in results.pose_landmarks.landmark:
                # Convertir las coordenadas normalizadas a coordenadas de píxeles
                height, width, _ = frame.shape
                cx, cy = int(landmark.x * width), int(landmark.y * height)
                cv2.circle(frame, (cx, cy), 1, (0, 255, 0), -1)
                # Sacamos las cositas
                x = landmark.x
                y = landmark.y
                z = landmark.z if landmark.HasField('z') else None
                fila = [x, y, z]
                if not fila:  # Si la fila está vacía, usamos la última entrada registrada
                    fila = matriz[-1] if matriz else [0, 0, 0]  # Usar la última entrada si existe, sino usar [0,0,0]
                matriz.append(fila)

        if len(matriz) < 33:
            print('Inside second if')
            matriz = []
            for _ in range(33):
                matriz.append([0,0,0])
        tensor.append(matriz)
        
        # Mostrar el fotograma procesado
        if False:
            cv2.imshow('Pose Estimation', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar los recursos
    cap.release()
    cv2.destroyAllWindows()
    return (np.array(tensor), fps_video, ancho, alto)

def coorVideoMoveNet(video_path, show_video=False, target_fps=30, model="movenet_thunder"):
    """
    Extract joint coordinates from a video using MoveNet.

    Args:
        video_path (str): Path to the input video file.
        show_video (bool): If True, displays the video with pose landmarks.
        target_fps (int): Target FPS for processing frames.
        model (str): MoveNet model to use ("movenet_thunder" or "movenet_lightning").

    Returns:
        (np.array, int, int, int): Tensor of shape (num_frames, 17, 3), FPS of the video, width, height.
    """
    # Load the MoveNet model from TensorFlow Hub
    if model == "movenet_thunder":
        model_url = "https://tfhub.dev/google/movenet/singlepose/thunder/4"
    elif model == "movenet_lightning":
        model_url = "https://tfhub.dev/google/movenet/singlepose/lightning/4"
    else:
        raise ValueError("Invalid model. Choose 'movenet_thunder' or 'movenet_lightning'.")

    movenet = hub.load(model_url)

    # Load the video
    cap = cv2.VideoCapture(video_path)
    original_fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_skip = max(1, original_fps // target_fps)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Original FPS: {original_fps}, Target FPS: {target_fps}, Frame Skip: {frame_skip}")
    print(f"Video dimensions: {width}x{height}")

    tensor = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_skip != 0:
            frame_count += 1
            continue
        frame_count += 1

        # Preprocess the frame for MoveNet
        input_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_frame = tf.image.resize_with_pad(tf.convert_to_tensor(input_frame), 256, 256)
        input_frame = tf.cast(input_frame, dtype=tf.int32)  # Explicitly cast to int32
        input_frame = tf.expand_dims(input_frame, axis=0)

        # Run inference
        outputs = movenet.signatures['serving_default'](input_frame)
        keypoints = outputs['output_0'].numpy()[0, 0, :, :]  # Shape: (17, 3)

        # Store the keypoints
        tensor.append(keypoints)

        # Optionally display the video with landmarks
        if show_video:
            for kp in keypoints:
                x, y, confidence = int(kp[1] * width), int(kp[0] * height), kp[2]
                if confidence > 0.5:  # Only draw keypoints with high confidence
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

            cv2.imshow("Pose Estimation", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

    return np.array(tensor), target_fps, width, height

def coorVideoExtremeMotion(video_path, show_video=False, target_fps=30):
    """
    Processes a video to extract joint coordinates for extreme motions using MediaPipe Pose and Kalman filtering.

    Args:
        video_path (str): Path to the input video file.
        show_video (bool): If True, displays the video with pose landmarks.
        target_fps (int): Target FPS for processing frames.

    Returns:
        (np.array, int, int, int): Tensor of dimensions (num_frames, 33, 3), FPS of the video, width, and height.
    """
    print("-" * 100)
    
    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    
    # Load the video
    cap = cv2.VideoCapture(video_path)
    original_fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_skip = max(1, original_fps // target_fps)
    ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Original FPS: {original_fps}, Target FPS: {target_fps}, Frame Skip: {frame_skip}")
    print(f"Video dimensions: {ancho}x{alto}")

    tensor = []
    prev_landmarks = None

    # Initialize Kalman filters for each joint
    kalman_filters = [
        KalmanFilter(
            transition_matrices=np.eye(3),  # Identity matrix for transitions
            observation_matrices=np.eye(3),  # Observations directly map to state
            initial_state_mean=np.zeros(3),
            initial_state_covariance=np.eye(3),
            observation_covariance=0.1 * np.eye(3),  # Noise in measurements
            transition_covariance=0.01 * np.eye(3)  # Noise in state transitions
        ) for _ in range(33)
    ]

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_skip != 0:
            frame_count += 1
            continue
        frame_count += 1

        # Preprocess the frame
        frame = cv2.GaussianBlur(frame, (3, 3), 0)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process with MediaPipe Pose
        results = pose.process(frame_rgb)

        landmarks = []
        if results.pose_landmarks:
            for landmark in results.pose_landmarks.landmark:
                landmarks.append([landmark.x, landmark.y, landmark.z])
        else:
            # If no landmarks detected, use previous or fill with zeros
            landmarks = prev_landmarks if prev_landmarks is not None else [[0, 0, 0] for _ in range(33)]

        # Fill missing landmarks
        if len(landmarks) < 33:
            for _ in range(33 - len(landmarks)):
                landmarks.append([0, 0, 0])

        # Apply Kalman filter smoothing
        smoothed_landmarks = []
        for i, landmark in enumerate(landmarks):
            # Update the Kalman filter with the current landmark
            smoothed_landmark = kalman_filters[i].filter_update(
                kalman_filters[i].initial_state_mean,
                kalman_filters[i].initial_state_covariance,
                np.array(landmark)  # Pass as 1D array
            )
            smoothed_landmarks.append(smoothed_landmark[0])  # Append smoothed result

        tensor.append(smoothed_landmarks)
        prev_landmarks = landmarks

        # Optionally display the frame
        if show_video:
            for smoothed_landmark in smoothed_landmarks:
                cx, cy = int(smoothed_landmark[0] * ancho), int(smoothed_landmark[1] * alto)
                cv2.circle(frame, (cx, cy), 2, (0, 255, 0), -1)
            cv2.imshow("Pose Estimation", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

    return np.array(tensor), target_fps, ancho, alto

def coorVideoExtremeMotionOpenPose(video_path, show_video=False, output_fps=30, model_folder="models/"):
    from openpose import pyopenpose as op
    """
    Processes a video to extract joint coordinates for extreme motions using OpenPose.

    Args:
        video_path (str): Path to the input video file.
        show_video (bool): If True, displays the video with pose landmarks.
        output_fps (int): Target FPS for processing frames.
        model_folder (str): Path to OpenPose model folder.

    Returns:
        (np.array, int, int, int): Tensor of dimensions (num_frames, 25, 3), FPS of the video, width, and height.
    """
  
    # OpenPose Parameters
    params = {
        "model_folder": model_folder,
        "net_resolution": "320x176",  # Reduce resolution for faster processing
        "tracking": 1,               # Use tracking to improve performance in fast motions
        "render_threshold": 0.5,     # Only render detections above this confidence
    }
    op_wrapper = op.WrapperPython()
    op_wrapper.configure(params)
    op_wrapper.start()

    # Load the video
    cap = cv2.VideoCapture(video_path)
    original_fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_skip = max(1, original_fps // output_fps)
    ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Original FPS: {original_fps}, Target FPS: {output_fps}, Frame Skip: {frame_skip}")
    print(f"Video dimensions: {ancho}x{alto}")

    tensor = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames dynamically to achieve target FPS
        if frame_count % frame_skip != 0:
            frame_count += 1
            continue
        frame_count += 1

        # Process frame with OpenPose
        datum = op.Datum()
        datum.cvInputData = frame
        op_wrapper.emplaceAndPop([datum])

        # Extract landmarks
        if datum.poseKeypoints is not None:
            # Ensure single-person keypoints (reshape if multi-person detected)
            keypoints = datum.poseKeypoints[0] if len(datum.poseKeypoints.shape) > 2 else datum.poseKeypoints
            tensor.append(keypoints)
        else:
            # Handle missing detections by appending empty keypoints
            tensor.append(np.zeros((25, 3)))

        # Optionally show the video with landmarks
        if show_video:
            cv2.imshow("Pose Estimation", datum.cvOutputData)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

    return np.array(tensor), output_fps, ancho, alto

def coordenadasPunto(punto:int, tensor:np.array):
    """Devuelve un np.array (len_video, 3) que representa la secuencia del punto dado

    Args:
        punto (int): el punto que representa la parte del cuerpo. Está entre 0 y 32.
        tensor (np.array): El array de donde sacar los puntos

    Returns:
        np.array: La serie temporal del punto.
    """
    if punto not in range(33):
        raise Exception("No metiste un número entre 0 y 32")
    return tensor[:,punto,:]

def graficasEnCarpetas(tensor:np.array, carpeta_guardado:str):
    """Guarda las series temporales de cada punto en la carpeta indicada.

    Args:
        tensor (np.array): El tensor de donde se van a guardar los puntos.
        carpeta_guardado (str): La dirección dentro de la carpeta raíz donde se guardarán las 33 gráficas.
    """
    # Si no existe la carpeta indicada, se crea.
    if not os.path.exists(carpeta_guardado):
        os.makedirs(carpeta_guardado)
    
    for punto in range(tensor.shape[1]):
        coord_punto = coordenadasPunto(punto, tensor)
        coords_x = coord_punto[:,0]
        coords_y = coord_punto[:,1]
        coords_z = coord_punto[:,2]

        plt.plot(coords_x, color="blue", label="x")
        plt.plot(coords_y, color="red", label="y")
        plt.plot(coords_z, color="green", label="z")
        plt.xlabel("fotogramas")
        plt.legend()
        plt.title(f"Evolución {partes_cuerpo[punto]}")
        #plt.show()

        # Guardar la gráfica en la carpeta especificada
        nombre_archivo = f"{partes_cuerpo[punto]}.png"
        ruta_guardado = os.path.join(carpeta_guardado, nombre_archivo)
        plt.savefig(ruta_guardado)

        plt.close()  # Cerrar la figura para liberar recursos

def ploteaPunto(punto:int, tensor:np.array, carpeta_guardado:str = None):
        plt.show()

def crear_video_puntos(lista_matrices, nombre_video_salida, fps, ancho, alto):
    # Dimensiones del video
    #height, width = 720, 1280  # Puedes ajustar estas dimensiones según sea necesario
    # Ajustar las dimensiones del video para que sean divisibles por 16
    alto = alto + (16 - alto % 16) % 16
    ancho = ancho + (16 - ancho % 16) % 16

    # Lista para almacenar los fotogramas del video
    frames = []

    # Iterar sobre cada matriz de coordenadas (cada fotograma)
    for matriz in lista_matrices:
        # Crear un fondo blanco
        frame = np.ones((alto, ancho, 3), dtype=np.uint8) * 255

        # Dibujar los puntos en el fotograma
        for punto in matriz:
            x, y, _ = punto  # Ignoramos la tercera dimensión (z)

            # Escalar las coordenadas al tamaño del video
            x = int(x * ancho)
            y = int(y * alto)

            # Asegurar que las coordenadas estén dentro de los límites del video
            x = max(0, min(ancho - 1, x))
            y = max(0, min(alto - 1, y))


            cv2.circle(frame, (int(x), int(y)), 1, (0, 0, 255), -1)  # Dibujar un círculo rojo en cada punto

        # Agregar el fotograma a la lista de fotogramas
        frames.append(frame)

    # Guardar los fotogramas como un video usando imageio
    imageio.mimwrite(nombre_video_salida + '.mp4', frames, fps=fps)

def find_and_copy_file(source_folder, dest_folder, filename):
    # Iterate through all files in the source folder
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            # Check if the file matches the desired filename
            if filename in file:
                # Construct the full path of the file
                source_file = os.path.join(root, file)
                
                # Construct the destination file path
                dest_file = os.path.join(dest_folder, file)
                
                # Copy the file to the destination folder
                shutil.copy(source_file, dest_file)
                print(f"Copied {file} to {dest_folder}")
                return  # Stop after the first match (if you only want one match)
    
    print(f"No file named '{filename}' found in {source_folder}.")

def load_landmark_data(npy_file):
    """
    Load landmark data from a .npy file.
    Args:
        npy_file (str): Path to the .npy file.
    Returns:
        np.ndarray: Array of shape (num_frames, 33, 3).
    """
    return np.load(npy_file)

def visualize_skeleton_motion(joint_positions, edges, interval=50):
    """
    Visualizes the motion of the skeleton in 3D.
    Args:
        joint_positions (np.ndarray): Array of shape (num_frames, num_joints, 3).
        edges (list): List of tuples representing skeleton edges (connections between joints).
        interval (int): Animation interval in milliseconds.
    """
    num_frames, num_joints, _ = joint_positions.shape

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Set axis limits
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_zlim([0, 1])

    # Plot for joints and edges
    joint_scatter, = ax.plot([], [], [], 'o', markersize=5, label='Joints')
    edge_lines = [ax.plot([], [], [], '-', color='black')[0] for _ in edges]

    def update(frame_idx):
        # Update joint positions
        joints = joint_positions[frame_idx]
        joint_scatter.set_data(joints[:, 0], joints[:, 1])
        joint_scatter.set_3d_properties(joints[:, 2])

        # Update edge lines
        for edge_idx, (start, end) in enumerate(edges):
            x = [joints[start, 0], joints[end, 0]]
            y = [joints[start, 1], joints[end, 1]]
            z = [joints[start, 2], joints[end, 2]]
            edge_lines[edge_idx].set_data(x, y)
            edge_lines[edge_idx].set_3d_properties(z)

        ax.set_title(f"Frame {frame_idx}")
        return joint_scatter, *edge_lines

    ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=interval, blit=False)
    plt.legend()
    plt.show()

def visualize_skeleton_motion_2d(joint_positions, edges, interval=50, image_path=None, output_location=None):
    """
    Visualizes the motion of the skeleton in 2D.
    Args:
        joint_positions (np.ndarray): Array of shape (num_frames, num_joints, 2).
        edges (list): List of tuples representing skeleton edges (connections between joints).
        interval (int): Animation interval in milliseconds.
        image_path (str): Path to the background image (e.g., frame of the video).
    """
    num_frames, num_joints, _ = joint_positions.shape

    fig, ax = plt.subplots()

    # Load the background image if provided
    if image_path:
        img = plt.imread(image_path)
        ax.imshow(img, extent=[0, img.shape[1], img.shape[0], 0])

    # Set axis limits based on joint positions or image dimensions
    if image_path:
        ax.set_xlim([0, img.shape[1]])
        ax.set_ylim([img.shape[0], 0])  # Flip y-axis for image coordinate system
    else:
        ax.set_xlim([0, 1])
        ax.set_ylim([1, 0])  # Flip y-axis for consistency

    # Plot for joints and edges
    joint_scatter, = ax.plot([], [], 'o', markersize=5, label='Joints')
    edge_lines = [ax.plot([], [], '-', color='black')[0] for _ in edges]

    def update(frame_idx):
        # Update joint positions
        joints = joint_positions[frame_idx]
        joint_scatter.set_data(joints[:, 0], joints[:, 1])

        # Update edge lines
        for edge_idx, (start, end) in enumerate(edges):
            x = [joints[start, 0], joints[end, 0]]
            y = [joints[start, 1], joints[end, 1]]
            edge_lines[edge_idx].set_data(x, y)

        ax.set_title(f"Frame {frame_idx}")
        return joint_scatter, *edge_lines

    ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=interval, blit=False)
    plt.legend()
    if output_location is not None:
            ani.save(output_location, writer='ffmpeg', fps=1000 / interval)
        #plt.show()

def plot_gp_predictions(mean: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor, npy_file: str, output_folder: str):
    """
    Generates time-series plots of Gaussian Process predictions with confidence intervals,
    with x, y, z plotted in the same figure for each joint.

    Args:
        mean (torch.Tensor): Predicted mean (shape: [n_samples, 33*3]).
        lower (torch.Tensor): Lower bound of confidence intervals (shape: [n_samples, 33*3]).
        upper (torch.Tensor): Upper bound of confidence intervals (shape: [n_samples, 33*3]).
        npy_file (str): Path to the original .npy file for joint names and time reference.
        output_folder (str): Directory to save the plots.
    """
    # Load the original data for time reference
    data = np.load(npy_file)  # Shape: [n_samples, 33, 3]
    _, n_joints, n_coords = data.shape
    time = np.arange(data.shape[0])  # Time indices

    # Reshape predictions to match joint dimensions
    #mean = mean.view(-1, n_joints, n_coords).cpu().numpy()
    #lower = lower.view(-1, n_joints, n_coords).cpu().numpy()
    #upper = upper.view(-1, n_joints, n_coords).cpu().numpy()

    # Normalize predictions for plotting
    mean = mean - np.nanmean(mean, axis=0)
    lower = lower - np.nanmean(lower, axis=0)
    upper = upper - np.nanmean(upper, axis=0)

    # Create output folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate through each joint and plot x, y, z coordinates in the same figure
    for joint_idx in range(n_joints):
        joint_name = partes_cuerpo.get(joint_idx, f"Joint_{joint_idx}")  # Use dictionary to get joint name

        plt.figure(figsize=(10, 6))
        
        # Plot x, y, z with their respective colors
        plt.plot(time, mean[:, joint_idx, 0], label='X (Blue)', color='blue')
        plt.fill_between(time, lower[:, joint_idx, 0], upper[:, joint_idx, 0],
                         color='blue', alpha=0.2)

        plt.plot(time, mean[:, joint_idx, 1], label='Y (Red)', color='red')
        plt.fill_between(time, lower[:, joint_idx, 1], upper[:, joint_idx, 1],
                         color='red', alpha=0.2)

        plt.plot(time, mean[:, joint_idx, 2], label='Z (Green)', color='green')
        plt.fill_between(time, lower[:, joint_idx, 2], upper[:, joint_idx, 2],
                         color='green', alpha=0.2)

        # Formatting the plot
        plt.title(f"{joint_name} (Gaussian Process Prediction)")
        plt.xlabel("Time (Frames)")
        plt.ylabel("Normalized Position")
        plt.legend()

        # Save the plot directly to the output folder
        plot_path = os.path.join(output_folder, f"{joint_name}.png")
        plt.savefig(plot_path)
        plt.close()

    print(f"Plots saved in {output_folder}")

def save_predictions_to_npy(mean: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor, output_file: str):
    """
    Saves Gaussian Process predictions (mean, lower, upper) to a .npy file.

    Args:
        mean (torch.Tensor): Predicted mean (shape: [n_samples, 33*3]).
        lower (torch.Tensor): Lower bound of confidence intervals (shape: [n_samples, 33*3]).
        upper (torch.Tensor): Upper bound of confidence intervals (shape: [n_samples, 33*3]).
        output_file (str): Path to save the .npy file.
    """
    # Convert tensors to numpy arrays
    predictions = {
        "mean": mean.cpu().numpy(),
        "lower": lower.cpu().numpy(),
        "upper": upper.cpu().numpy(),
    }

    # Save to .npy file
    np.save(output_file, predictions)
    print(f"Predictions saved to {output_file}")