'''
ArUco Marker Detection and Area Drawing Controller
Cybersterilizer Project
Università degli Studi di Palermo
'''

from controller import Supervisor
import cv2
import cv2.aruco as aruco
import numpy as np

# Define the supervisor and the time step
supervisor = Supervisor()
TIME_STEP = int(supervisor.getBasicTimeStep())

# Define the robot node
robot = supervisor.getFromDef("END_EFFECTOR")

# Initialize the camera
camera = supervisor.getDevice('camera')
camera.enable(TIME_STEP)
width = camera.getWidth()
height = camera.getHeight()
fov = camera.getFov()

# Calculate the focal length
fx = (width / 2) / np.tan(fov / 2)
fy = fx

# Calculate the central point
cx = width / 2
cy = height / 2

# Calculate the camera matrix
camera_matrix = np.array(
    [
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float32
)

# No distortion coefficients
distortion_coeffs = np.zeros((4, 1), dtype=np.float32)

# Initialize the display
display = supervisor.getDevice('display')

# Define the ArUco dictionary and detector parameters
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
parameters = aruco.DetectorParameters()

# Physical length of the marker in meters
marker_length = 0.05

# Create a dictionary with the marker IDs and their corresponding definitions
marker_definitions = {
    0: "ARUCO_1",
    1: "ARUCO_2",
    2: "ARUCO_3",
    3: "ARUCO_4",
}

# Create a dictionary to store the marker positions according to the camera frame
camera_marker_positions = {
    0: np.array([0, 0, 0], dtype=np.float32),
    1: np.array([0, 0, 0], dtype=np.float32),
    2: np.array([0, 0, 0], dtype=np.float32),
    3: np.array([0, 0, 0], dtype=np.float32),
}

# Create a dictionary to store the marker positions according to the world frame
world_marker_positions = {
    0: np.array([0, 0, 0], dtype=np.float32),
    1: np.array([0, 0, 0], dtype=np.float32),
    2: np.array([0, 0, 0], dtype=np.float32),
    3: np.array([0, 0, 0], dtype=np.float32),
}

# Function to get marker positions in camera coordinates
def update_camera_marker_positions(corners, ids):
    if ids is None:
        return

    # Estimate pose
    rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
        corners, marker_length, camera_matrix, distortion_coeffs
    )

    # Get the coordinates of the markers
    for i in range(len(ids)):
        tvec = tvecs[i][0]
        rvec = rvecs[i][0]

        # Store the coordinates according to the camera frame
        camera_marker_positions[ids[i][0]] = tvec

def update_world_marker_positions():

    for marker in marker_definitions.keys():

        # Get the node
        marker_node = supervisor.getFromDef(marker_definitions[marker])

        # Get the position of the marker in world coordinates
        position = marker_node.getPosition()

        # Store the position in the dictionary
        world_marker_positions[marker] = np.array([position[0], position[1], position[2]], dtype=np.float32)

# Function to print marker positions in camera/world coordinates
def print_marker_positions(corners, ids):
    if ids is None:
        return

    # Print the coordinates of the markers in camera coordinates
    for marker in marker_definitions.keys():
        print(f"Marker {marker_definitions[marker]} (Camera position): {camera_marker_positions[marker]}")
        print(f"Marker {marker_definitions[marker]} (World position): {world_marker_positions[marker]}")

# Function to detect ArUco markers in a raw image buffer
def detect_markers(raw, width, height):
    
    # Convert the raw buffer to a BGRA image
    image_bgra = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 4))
    
    # Convert BGRA to BGR for OpenCV processing
    image_bgr = cv2.cvtColor(image_bgra, cv2.COLOR_BGRA2BGR)
    
    # Detect ArUco markers
    corners, ids, _ = aruco.detectMarkers(image_bgr, aruco_dict, parameters=parameters)

    # Estimate pose and draw axis if markers are detected
    if ids is not None:
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
            corners, marker_length, camera_matrix, distortion_coeffs
        )
        for i in range(len(ids)):
            cv2.drawFrameAxes(image_bgr, camera_matrix, distortion_coeffs, rvecs[i], tvecs[i], marker_length * 0.5)

    # Draw detected markers
    image_bgr_marked = aruco.drawDetectedMarkers(image_bgr.copy(), corners, ids)

    # Convert back to BGRA for Webots display
    image_bgra_marked = cv2.cvtColor(image_bgr_marked, cv2.COLOR_BGR2BGRA)
    return image_bgra_marked, corners, ids

# Function to draw a quadrilateral area based on the first 4 detected markers
def draw_area(image, corners, ids):
    
    if ids is None or len(ids) < 4:
        return None
    centers = []
    
    # Reshape corner points and calculate the center of each marker
    for c in corners[:4]:
        pts = c.reshape((4, 2))
        center = pts.mean(axis=0)
        centers.append(center)
    
    centers = np.array(centers, dtype=np.float32)
    centroid = centers.mean(axis=0)
    angles = np.arctan2(centers[:, 1] - centroid[1], centers[:, 0] - centroid[0])
    order = np.argsort(angles)
    pts4 = centers[order].astype(int)
    cv2.polylines(image, [pts4], isClosed=True, color=(0, 255, 0), thickness=3)
    return image

# Frame processing control variables
frame_counter = 0
last_raw = None
image_to_display = np.zeros((height, width, 4), dtype=np.uint8)

# Main loop
while supervisor.step(TIME_STEP) != -1:
    raw = camera.getImage()
    frame_counter += 1

    if frame_counter % 3 == 0 and raw != last_raw:
        
        image_markers, corners, ids = detect_markers(raw, width, height)

        if ids is not None and len(ids) >= 4:
            image_area = draw_area(image_markers.copy(), corners, ids)
            image_to_display = image_area if image_area is not None else image_markers
        else:
            image_to_display = image_markers
        
        # Update the marker positions in world coordinates (FOR SIMULATION PURPOSES ONLY)
        update_world_marker_positions()

        # Update the marker positions in camera coordinates
        update_camera_marker_positions(corners, ids)

        # Print the marker positions (camera/world coordinates)
        print_marker_positions(corners, ids)

        # Update display in Webots
        ir = display.imageNew(image_to_display.tobytes(), display.BGRA, width, height)
        display.imagePaste(ir, 0, 0, False)
        display.imageDelete(ir)

        last_raw = raw