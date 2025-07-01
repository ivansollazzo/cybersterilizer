'''
University of Palermo - Medical Robotics Course
ABB IRB4600 Controller
Developed by: Ivan Sollazzo, Gabriele Burgio, Luca Masi and Federico Ennio Ambrogi
'''

# Import Webots libraries
from controller import Supervisor

# Import base libraries
import tempfile
import math
import numpy as np
import random

# Import OpenCV libraries
import cv2
import cv2.aruco as aruco

# Import robotics libraries. We use ikpy for inverse kinematics and Robotics Toolbox by Peter Corke for trajectory generation.
import ikpy
from ikpy.chain import Chain

# Check if Numpy has the disp function, if not, define it (not all versions of Numpy need this)
if not hasattr(np, 'disp'):
    np.disp = lambda x, *args, **kwargs: print(x, *args, **kwargs)

# Import trajectory generation tools from Robotics Toolbox
from roboticstoolbox.tools.trajectory import jtraj

# Define max iterations for IKPY library. Higher values can lead to better accuracy but may increase computation time.
IKPY_MAX_ITERATIONS = 100

# Target orientation for the end-effector (pointing straight down)
# Assumes robot base Z is up, end-effector Z is tool direction.
# This matrix represents a 180-degree rotation around the base X-axis.
target_orientation_matrix_down = np.array([
    [1,  0,  0],  # End-effector X-axis aligned with robot base X-axis
    [0, -1,  0],  # End-effector Y-axis aligned with robot base -Y-axis
    [0,  0, -1]   # End-effector Z-axis aligned with robot base -Z-axis (pointing down)
])

# Initialize the Webots supervisor and calculate the time step.
supervisor = Supervisor()
timeStep = int(4 * supervisor.getBasicTimeStep())

# Enable keyboard input
keyboard = supervisor.getKeyboard()
keyboard.enable(timeStep)

# Create the arm chain from the URDF
filename = None
with tempfile.NamedTemporaryFile(suffix='.urdf', delete=False) as file:
    filename = file.name
    file.write(supervisor.getUrdf().encode('utf-8'))

# Define the end effector offset. It will be used to adjust as the last link vector in the chain.
end_effector_offset = [0.0, 0.0, 0.3]

# Initialize the arm chain using the URDF file
chain = Chain.from_urdf_file(filename, last_link_vector=end_effector_offset, active_links_mask=[False, True, True, True, True, True, True, False, False, False])

# Initialize the arm motors and encoders.
motors = []

for link in chain.links:
    
    # If there's a motor in the link, initialize it with its position sensor
    if 'motor' in link.name:
        motor = supervisor.getDevice(link.name)
        motor.setVelocity(1.0)
        position_sensor = motor.getPositionSensor()
        position_sensor.enable(timeStep)
        motors.append(motor)

# Get the arm and target nodes.
arm = supervisor.getSelf()

# Initialize the camera
camera = supervisor.getDevice('camera')
camera.enable(timeStep)

# Initialize the UV Detector and light killer
uv_detector = supervisor.getFromDef("UVDetector")
light_killer = supervisor.getFromDef("killer")

# Initialize bacteria group
bacteria_group = supervisor.getFromDef("BacteriaGroup")

# Calculate camera parameters
img_width, img_height = camera.getWidth(), camera.getHeight()
fov = camera.getFov()

# Calculate the camera intrinsic matrix. Camera is assumed to be the pinhole model.
cx, cy = img_width / 2, img_height / 2
fx = cx / np.tan(fov / 2)
camera_matrix = np.array([[fx, 0, cx], [0, fx, cy], [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros((4, 1), dtype=np.float32)

# Initialize a display for the camera image
display = supervisor.getDevice('display')

# Initialize the ArUco marker dictionary and parameters
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
detector_params = aruco.DetectorParameters()

# Set the marker size in meters and target cell area in cm^2
MARKER_SIZE = 0.05  # meters
CELL_AREA_TARGET = 30.0  # cm^2

# Map marker IDs to DEF names in Webots
marker_defs = {
    0: 'ARUCO_1',
    1: 'ARUCO_2',
    2: 'ARUCO_3',
    3: 'ARUCO_4'
}

# Initialize dictionaries to store camera and world positions of markers
cam_positions = {i: np.zeros(3, np.float32) for i in marker_defs}
world_positions = {i: np.zeros(3, np.float32) for i in marker_defs}

# Function to update camera positions of markers
def update_cam_positions(corners, ids):

    # If there are no markers, return
    if ids is None:
        return

    # Do pose estimation for each marker
    _, translations, _ = aruco.estimatePoseSingleMarkers(
        corners, MARKER_SIZE, camera_matrix, dist_coeffs)

    # Store camera positions in the dictionary
    for idx, marker_id in enumerate(ids.flatten()):
        cam_positions[marker_id] = translations[idx][0]

# Function to update world positions of markers
def update_world_positions():
    
    # Update world positions of markers from Webots nodes
    for m_id, def_name in marker_defs.items():
        node = supervisor.getFromDef(def_name)
        if node:
            world_positions[m_id] = np.array(node.getPosition(), np.float32)

# Function to compute the centroid of the four markers in world space
def compute_world_center():
    pts = np.stack([world_positions[i] for i in sorted(marker_defs)])
    return pts.mean(axis=0)

# Function to print camera vs world positions of markers as a table
def print_marker_table():
    header = "Marker      |   Cam X   Cam Y   Cam Z ||   Wld X   Wld Y   Wld Z"
    print(header)
    print('-' * len(header))
    for m_id, def_name in marker_defs.items():
        cam = cam_positions[m_id]
        wld = world_positions[m_id]
        print(f"{def_name:<10} | {cam[0]:8.3f}{cam[1]:8.3f}{cam[2]:8.3f} || {wld[0]:8.3f}{wld[1]:8.3f}{wld[2]:8.3f}")

# Function to compute grid dimensions based on target cell area
def compute_grid_dims():

    # Compute the area of the quadrilateral formed by the four markers
    corners = [world_positions[i] for i in range(4)]

    # Calculate the area of the quadrilateral by dividing it into two triangles and summing their areas.
    a1 = 0.5 * np.linalg.norm(np.cross(corners[1]-corners[0], corners[2]-corners[0]))
    a2 = 0.5 * np.linalg.norm(np.cross(corners[2]-corners[0], corners[3]-corners[0]))

    # Convert total area from m^2 to cm^2
    total_area_cm2 = (a1 + a2) * 1e4

    # Calculate the number of cells based on the target cell area
    num_cells = max(1, int(round(total_area_cm2 / CELL_AREA_TARGET)))

    # Calculate the number of rows and columns for the grid
    rows = int(np.floor(np.sqrt(num_cells)))
    cols = int(np.ceil(num_cells / rows))

    return rows, cols

# Function to detect markers and draw axes on the image
def detect_and_draw_markers(frame_raw):

    # Convert the raw image to a numpy array and reshape it
    img_bgra = np.frombuffer(frame_raw, np.uint8).reshape(img_height, img_width, 4)
    img_bgr = cv2.cvtColor(img_bgra, cv2.COLOR_BGRA2BGR)

    # Detect the markers in the image
    corners, ids, _ = aruco.detectMarkers(img_bgr, aruco_dict, parameters=detector_params)
    
    # If there are detected markers, draw them on the image
    if ids is not None:
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
            corners, MARKER_SIZE, camera_matrix, dist_coeffs)
        for rv, tv in zip(rvecs, tvecs):
            cv2.drawFrameAxes(img_bgr, camera_matrix, dist_coeffs, rv, tv, MARKER_SIZE/2)
    img_marked = aruco.drawDetectedMarkers(img_bgr.copy(), corners, ids)
    
    return img_marked, corners, ids

# Function to sort the corners of the detected markers
def sort_quad(corners, ids):

    # Sort the corners based on their IDs
    pts = np.array([c.reshape(-1,2).mean(axis=0) for c in corners[:4]], np.float32)
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:,1]-center[1], pts[:,0]-center[0])
    order = np.argsort(angles)
    
    return pts[order], ids.flatten()[order]

# Function to draw grid overlay on the image and compute 2D centers
def draw_grid_overlay(img, quad_pts, rows, cols):

    # Draw the grid overlay on the image
    tl, tr, br, bl = quad_pts
    grid = np.zeros((rows+1, cols+1, 2), np.float32)

    for i in range(rows+1):
        alpha = i / rows
        left  = tl*(1-alpha) + bl*alpha
        right = tr*(1-alpha) + br*alpha
        
        for j in range(cols+1):
            grid[i,j] = left*(1-j/cols) + right*(j/cols)
    
    # Draw grid lines
    for i in range(rows+1): cv2.polylines(img, [grid[i].astype(int)], False, (0,0,255), 3)
    for j in range(cols+1): cv2.polylines(img, [grid[:,j].astype(int)], False, (0,0,255), 3)
    
    # Compute and mark cell centers (2D)
    centers2d = {}
    
    for i in range(rows):
        for j in range(cols):
            pts = grid[[i,i+1,i+1,i],[j,j,j+1,j+1]]
            center = tuple(pts.mean(axis=0).astype(int))
            centers2d[(i,j)] = center
            cv2.circle(img, center, 3, (255,0,0), -1)
    
    return img, centers2d

# Function to interpolate camera centers from 2D marker positions
def interpolate_cam_centers(centers2d, rows, cols):

    centers3d = {}
    
    for (i, j), _ in centers2d.items():
        u, v = (j + 0.5) / cols, (i + 0.5)/ rows
        top = cam_positions[0] * (1-u) + cam_positions[1] * u
        bot = cam_positions[3] * (1-u) + cam_positions[2] * u
        centers3d[(i,j)] = top * (1-v) + bot * v
    
    return centers3d

# Function to interpolate world centers from sorted marker IDs
def interpolate_world_centers(sorted_ids, rows, cols):
    
    p0, p1, p2, p3 = [world_positions[id_] for id_ in sorted_ids]
    
    centers3d = {}
    
    for i in range(rows):
        for j in range(cols):
            u, v = (j + 0.5) / cols, (i + 0.5) / rows
            centers3d[(i,j)] = (
                (1-u) * (1-v) * p0 + u *(1-v) * p1 + u * v * p2 + (1-u) * v * p3
            )
    
    return centers3d

# Function to print the cell table with camera and world coordinates
def print_cell_table(cam_centers, world_centers):
    header = "Cell (i,j) |   Cam X   Cam Y   Cam Z ||   Wld X   Wld Y   Wld Z"
    print(header)
    print('-'*len(header))
    for key in sorted(cam_centers):
        c = cam_centers[key]
        w = world_centers[key]
        print(f"({key[0]},{key[1]})        |{c[0]:8.3f}{c[1]:8.3f}{c[2]:8.3f} ||{w[0]:8.3f}{w[1]:8.3f}{w[2]:8.3f}")

def convert_webots_to_robot_coordinates(targetPosition):

    # Get the arm position in Webots coordinates.
    armPosition = arm.getPosition()

    # Compute the position of the target relatively to the arm.
    # x and y axis are inverted because the arm is not aligned with the Webots global axes.
    x = -(targetPosition[1] - armPosition[1])
    y = targetPosition[0] - armPosition[0]
    z = targetPosition[2] - armPosition[2]

    return [x, y, z]

# Function to spawn bacteria in the grid cells
def spawn_bacteria_in_cells(world_centers, num_bacteria=5):

    # If there are no world centers, return an empty dictionary
    if not world_centers:
        print("No cell centers available for bacteria spawning!")
        return {}
    
    # Get the bacteria group from the supervisor
    bacteria_group = supervisor.getFromDef("BacteriaGroup")
    if not bacteria_group:
        print("BacteriaGroup not found!")
        return {}
    
    # Get the field for children nodes in the bacteria group
    group_field = bacteria_group.getField("children")

    # Define the proto for the bacteria to be spawned
    bacteria_proto = """
    Solid {
      translation 0 0 0
      children [
        Shape {
          appearance Appearance {
            material Material {
              diffuseColor 0.8 0.1 0.1
              transparency 1.0
            }
          }
          geometry Sphere {
            radius 0.01
          }
        }
      ]
      name "bacteria"
      boundingObject Sphere {
        radius 0.005
      }
    }
    """
    
    # Initialize a dictionary to map bacteria to cells
    batterio_to_cella = {}

    # Get the positions of the cells in the world
    cell_positions = list(world_centers.values())

    # Spawn the specified number of bacteria
    for i in range(num_bacteria):
        
        # Choose a random cell index to spawn the bacteria
        cell_index = random.randint(0, len(cell_positions) - 1)
        base_position = cell_positions[cell_index]

        bacteria_position = base_position

        # Spawn the bacteria at the chosen cell position
        group_field.importMFNodeFromString(-1, bacteria_proto)
        node = group_field.getMFNode(group_field.getCount() - 1)
        node.getField("translation").setSFVec3f(bacteria_position.tolist())

        batterio_to_cella[cell_index] = i    

    print(f"Spawned {num_bacteria} bacteria in the grid cells")
    print("Mapping bacteria to cell:", batterio_to_cella)

    return batterio_to_cella

# Function to get the positions of all bacteria in the world
def get_bacteria_positions():
    positions = []
    bacteria_group = supervisor.getFromDef("BacteriaGroup")
    if not bacteria_group:
        return positions
    
    children_field = bacteria_group.getField("children")
    count = children_field.getCount()

    for i in range(count):
        node = children_field.getMFNode(i)
        if node and node.getField("translation"):
            positions.append((node, np.array(node.getField("translation").getSFVec3f(), dtype=np.float32)))

    return positions

# Function to find contours of red bacteria in the raw camera image and draw them
def findContours(raw_image_buffer):

    # Convert the raw image buffer to a BGR image
    img_bgra = np.frombuffer(raw_image_buffer, np.uint8).reshape(img_height, img_width, 4)
    img_bgr = cv2.cvtColor(img_bgra, cv2.COLOR_BGRA2BGR)
    
    # Detect red bacteria using color thresholding
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([179, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Find contours in the mask by using OpenCV
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on the original image
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Draw a rectangle around the contour
        for thickness in range(1, 4):
            cv2.rectangle(img_bgr, 
                         (max(0, x - thickness), max(0, y - thickness)), 
                         (min(img_width, x + w + thickness), min(img_height, y + h + thickness)), 
                         (0, 255, 0),  # Verde brillante in formato BGR
                         2)
        
        # Draw a circle at the center of the contour
        center_x = x + w // 2
        center_y = y + h // 2
        radius = max(5, max(w, h) // 8)
        cv2.circle(img_bgr, (center_x, center_y), radius, (0, 255, 0), -1)
        cv2.circle(img_bgr, (center_x, center_y), radius + 2, (0, 0, 255), 2)
    
    return img_bgr

# Function to update the Webots display with a given BGR image
def update_display(image_bgr_to_show):
    
    # Convert the BGR image to BGRA format for Webots display
    img_bgra = cv2.cvtColor(image_bgr_to_show, cv2.COLOR_BGR2BGRA)
    img_buffer = img_bgra.tobytes()
    
    # Basic check to ensure display is somewhat valid before using
    if display.getWidth() > 0 and display.getHeight() > 0:
        ir = display.imageNew(img_buffer, display.BGRA, img_width, img_height)
        if ir:
            display.imagePaste(ir, 0, 0, False)
            display.imageDelete(ir)

# Function to process the raw camera image, detect markers, and draw them
def run_detection_and_get_annotated_image(raw_camera_image_buffer):
    
    # Get BGR image with basic ArUco markers drawn from the raw camera buffer
    img_bgr_with_aruco, corners, ids = detect_and_draw_markers(raw_camera_image_buffer)
    
    # Initialize return values
    final_annotated_image_bgr = img_bgr_with_aruco.copy() # Start with ArUco-drawn image
    calculated_cam_centers = {}
    calculated_world_centers = {}
    
    if ids is not None and len(ids) >= 4:
        
        # Update global marker positions (these are used by interpolation functions)
        update_world_positions()
        update_cam_positions(corners, ids)
        
        # Sort quad corners and IDs from the current detection
        quad_pts, quad_ids = sort_quad(corners, ids)
        
        # Compute grid layout (uses global world_positions which were just updated)
        rows, cols = compute_grid_dims()
        
        # Overlay grid on the image that already has ArUco markers
        final_annotated_image_bgr, centers2d = draw_grid_overlay(final_annotated_image_bgr, quad_pts, rows, cols)
        
        # Interpolate 3D cell centers (these use global cam_positions and world_positions)
        calculated_cam_centers = interpolate_cam_centers(centers2d, rows, cols)
        calculated_world_centers = interpolate_world_centers(quad_ids, rows, cols)

    return final_annotated_image_bgr, calculated_cam_centers, calculated_world_centers

# Function to check if the robot has moved from a target position
def check_if_moved(targetPosition):

    # Get the current joint angles
    current_joint_angles = [0] + [m.getPositionSensor().getValue() for m in motors] + [0, 0, 0]

    # Calculate the forward kinematics and get the position vector
    fk = chain.forward_kinematics(current_joint_angles)
    current_position = fk[0:3, 3]

    # Convert the target position to robot coordinates
    target_position_robot = convert_webots_to_robot_coordinates(targetPosition)

    # Calculate the squared distance between the current position and the target position
    squared_distance = (current_position[0] - target_position_robot[0])**2 + \
                       (current_position[1] - target_position_robot[1])**2 + \
                       (current_position[2] - target_position_robot[2])**2
    # Check if the distance is greater than a threshold (e.g., 0.01)
    if math.sqrt(squared_distance) < 0.01:
        return True
    
    return False

# Function to get current joint angles
def get_current_joint_angles():
    return [0] + [m.getPositionSensor().getValue() for m in motors] + [0, 0, 0]


# Definition of the state machine of the robot
STANDBY, MOVING_TO_DETECT_SPOT, DETECTING_AT_SPOT, AWAITING_CONFIRMATION, MONITORING_SELECT_CELL, MOVING_TO_CELL, STERILIZING_CELL = range(7)
current_state = STANDBY

# Dictionaries to store camera and world centers of cells
cam_centers = {}
world_centers = {}

# Variables to store the current cell index and timestep sum
current_cell_index = 0
timestep_sum = 0

# Variable to store the current state of the bacteria spawning
bacteria_spawned = False

# Dictionary to map bacteria to cells
batteri_dict = {}

# Variables for trajectory management
active_trajectory = None
trajectory_step_index = 0

# Stores the world coordinate target for check_if_moved
g_target_for_check_if_moved = None

# Flags to control one-time console messages
standby_prompt_shown = False
confirmation_prompt_shown = False

# Flag to control if the robot is coming from sterilization
coming_from_process = False

# Flag to control if there's a force reset
force_reset = False

# To store the last image with grid for display in STANDBY
last_known_annotated_image = None

# Main loop
while supervisor.step(timeStep) != -1:
    raw_camera_image_buffer = camera.getImage() # Get raw image data once per loop
    key = keyboard.getKey()

    # Global stop command ('x' key)
    if key == ord('X') or key == ord('x') or force_reset:
        
        print("Stop command received ('x'). Returning to STANDBY.")
        current_state = STANDBY

        # Reset all state variables
        standby_prompt_shown = False
        confirmation_prompt_shown = False
        bacteria_spawned = False
        force_reset = False
        
        # Reset trajectory and step index
        active_trajectory = None
        trajectory_step_index = 0
        g_target_for_check_if_moved = None
        
        # Reset camera and world centers
        cam_centers = {}
        world_centers = {}

        # Reset current cell index and bacteria dictionary
        current_cell_index = 0
        batteri_dict = {}

        # Resetting flag to indicate the robot is coming from a process
        coming_from_process = False

        # Clear the last known annotated image for the display
        last_known_annotated_image = None

        # Reset all spawned bacteria in the world
        bacteria_group = supervisor.getFromDef("BacteriaGroup")
        if bacteria_group:
            children_field = bacteria_group.getField("children")
            # Rimuovi tutti i batteri esistenti
            while children_field.getCount() > 0:
                children_field.removeMF(0)
            print("All existing bacteria removed from world.")

        # Power off all spotlights
        killer_spotlight = supervisor.getFromDef("killerspotlight")
        if killer_spotlight:
            intensity_field = killer_spotlight.getField("intensity")
            if intensity_field:
                intensity_field.setSFFloat(0.0)

        detector_spotlight = supervisor.getFromDef("detectorspotlight")
        if detector_spotlight:
            intensity_field = detector_spotlight.getField("intensity")
            if intensity_field:
                intensity_field.setSFFloat(0.0)
        
        # Print a message to indicate the robot has stopped
        print("Robot stopped and reset to STANDBY.")
        
        # Clear remaining key presses for this step
        while key != -1: key = keyboard.getKey()
        
        # Continue to the next iteration to process the STANDBY state immediately
        continue

    # Start with an image that has ArUco markers drawn if any are visible.
    image_to_display_bgr, _, _ = detect_and_draw_markers(raw_camera_image_buffer)

    # In current state STANDBY, the controller is ready to start the detection at any time
    if current_state == STANDBY:

        # If coming from sterilization, process the trajectory
        if coming_from_process:

            # Set the flag bacteria spawned to False
            bacteria_spawned = False

            # Reset the bacteria dictionary
            batteri_dict = {}

            # Reset cam and world centers
            cam_centers = {}
            world_centers = {}
            current_cell_index = 0
            last_known_annotated_image = None

            # Reset all spawned bacteria in the world
            bacteria_group = supervisor.getFromDef("BacteriaGroup")
            if bacteria_group:
                children_field = bacteria_group.getField("children")
                
                # Rimuovi tutti i batteri esistenti
                while children_field.getCount() > 0:
                    children_field.removeMF(0)          

            # If there's an active trajectory, move the robot along it
            if active_trajectory is not None and trajectory_step_index < len(active_trajectory):

                # Get the current joint targets from the active trajectory
                current_joint_targets = active_trajectory[trajectory_step_index]

                # Set the motors to the current joint targets
                for i in range(len(motors)):
                    motors[i].setPosition(current_joint_targets[i + 1])

                # Increase the trajectory step index
                trajectory_step_index += 1

            # If the trajectory is complete, or if no trajectory is set, check if the robot has arrived at the target position
            else:

                # If there's an active trajectory but we reached the end, ensure the final joint targets are set
                if active_trajectory is not None and trajectory_step_index >= len(active_trajectory):

                     # Get the final joint targets from the active trajectory
                     final_joint_targets = active_trajectory[-1]

                     # Set the motors to the final joint targets
                     for i in range(len(motors)):
                        motors[i].setPosition(final_joint_targets[i + 1])

                # Check if the robot has arrived at the target position.
                # If it has, change state to DETECTING_AT_SPOT. Then clear the trajectory.
                if g_target_for_check_if_moved is not None and check_if_moved(g_target_for_check_if_moved):
                    print("Arrived at standby position.")
                    active_trajectory = None
                    coming_from_process = False

                # If there isn't any active trajectory, change state to STANDBY
                elif g_target_for_check_if_moved is None and active_trajectory is None:
                    print("Error: No target to move to for detection. Returning to STANDBY.")
                    current_state = STANDBY

                # If the robot has not arrived yet, continue moving
                else:
                    print("Waiting to arrive at standby position...")

        # Get the killer spotlight from supervisor. If it is not found, print an error. Otherwise, get the intensity field and set it to 0.
        killer_spotlight = supervisor.getFromDef("killerspotlight")

        if killer_spotlight is None:
            print("Error: 'killerspotlight' node not found.")
        else:
            intensitykiller_field = killer_spotlight.getField("intensity")
            if intensitykiller_field is None:
                print("Error: 'intensity' field not found in 'killerspotlight'.")
            else:
                intensitykiller_field.setSFFloat(0.0)
        
        # Get the detector spotlight from supervisor. If it is not found, print an error. Otherwise, get the intensity field and set it to 0.
        detector_spotlight = supervisor.getFromDef("detectorspotlight")

        if detector_spotlight is None:
            print(f"Error: node 'detectorspotlight' not found.")
        else:
            intensitydetector_field = detector_spotlight.getField("intensity")

            if intensitydetector_field is None:
                print(f"Error: field 'intensity' not found in 'detectorspotlight'.")
            else:
                intensitydetector_field.setSFFloat(0.0)
        
        # If no active trajectory, show the standby prompt only once
        if active_trajectory is None:
            if not standby_prompt_shown:
                print("State: STANDBY. Press 's' to start detection sequence, or 'x' to stop (if applicable).")
                standby_prompt_shown = True

        # If start command ('s') is received, transition to MOVING_TO_DETECT_SPOT
        if key == ord('S') or key == ord('s'):
            print("Start command ('s') received. Initializing detection sequence.")
            standby_prompt_shown = False
            
            # Prepare for movement to initial detection spot
            target_position_world = [0, 0.150, 1]
            
            x_robot, y_robot, z_robot = convert_webots_to_robot_coordinates(target_position_world)
            ik_target_robot = np.array([x_robot, y_robot, z_robot])
            
            initial_joint_positions = get_current_joint_angles()
            
            try:
                target_joint_positions = chain.inverse_kinematics(
                    ik_target_robot,
                    target_orientation=target_orientation_matrix_down,
                    orientation_mode="all",
                    max_iter=IKPY_MAX_ITERATIONS,
                    initial_position=initial_joint_positions
                )
            
            except Exception as e:
                
                # If IK fails, print an error and reset the trajectory.
                print(f"IK failed for initial detection spot {target_position_world} with orientation: {e}")
                active_trajectory = None
                continue

            # Set the actve trajectory to move to the target position
            active_trajectory = jtraj(initial_joint_positions, target_joint_positions, 10).q
            trajectory_step_index = 0
            
            # Set the target for check_if_moved to the target position
            g_target_for_check_if_moved = target_position_world

            # Change the state to MOVING_TO_DETECT_SPOT
            current_state = MOVING_TO_DETECT_SPOT

    # If on state MOVING_TO_DETECT_SPOT, move the robot to the detection spot
    elif current_state == MOVING_TO_DETECT_SPOT:
        
        # If there's an active trajectory, move the robot along it
        if active_trajectory is not None and trajectory_step_index < len(active_trajectory):

            # Get the current joint targets from the active trajectory
            current_joint_targets = active_trajectory[trajectory_step_index]
            
            # Set the motors to the current joint targets
            for i in range(len(motors)):
                motors[i].setPosition(current_joint_targets[i + 1])
            
            # Increase the trajectory step index
            trajectory_step_index += 1
        
        # If the trajectory is complete, or if no trajectory is set, check if the robot has arrived at the target position
        else:

            # If there's an active trajectory but we reached the end, ensure the final joint targets are set
            if active_trajectory is not None and trajectory_step_index >= len(active_trajectory):
                 
                 # Get the final joint targets from the active trajectory
                 final_joint_targets = active_trajectory[-1]

                 # Set the motors to the final joint targets
                 for i in range(len(motors)):
                    motors[i].setPosition(final_joint_targets[i + 1])
            
            # Check if the robot has arrived at the target position.
            # If it has, change state to DETECTING_AT_SPOT. Then clear the trajectory.
            if g_target_for_check_if_moved is not None and check_if_moved(g_target_for_check_if_moved):
                print("Arrived at detection spot.")
                current_state = DETECTING_AT_SPOT
                active_trajectory = None
            
            # If there isn't any active trajectory, change state to STANDBY
            elif g_target_for_check_if_moved is None and active_trajectory is None:
                print("Error: No target to move to for detection. Returning to STANDBY.")
                current_state = STANDBY
            
            # If the robot has not arrived yet, continue moving
            else:
                print("Waiting to arrive at detection spot...")

    # If on state DETECTING_AT_SPOT, perform detection and prepare for confirmation
    elif current_state == DETECTING_AT_SPOT:
        
        # Perform detection and get the annotated image with cell centers
        annotated_image, detected_cam_centers, detected_world_centers = run_detection_and_get_annotated_image(raw_camera_image_buffer)
        
        # Show the detailed image for this state
        image_to_display_bgr = annotated_image
        
        # Update global cam_centers and world_centers for use in this state and potentially others
        cam_centers = detected_cam_centers
        world_centers = detected_world_centers

        # If there are world centers detected, update the last known annotated image. Else, go back to standby. Also reset the active trajectory.
        if world_centers:
            last_known_annotated_image = image_to_display_bgr.copy()
        else:
            last_known_annotated_image = None

        if not world_centers:
            print("Error: No cell centers found at detection spot. Check camera view/marker setup. Returning to STANDBY.")
            current_state = STANDBY 
            active_trajectory = None
            continue

        # If there are not bacteria spawned yet, spawn them in the detected cells
        if not bacteria_spawned:
            batteri_dict = spawn_bacteria_in_cells(world_centers, num_bacteria=5) 
            bacteria_spawned = True
        
        # The controller is now ready to confirm the detection and proceed with sterilization. So, change the state to AWAITING_CONFIRMATION.
        current_state = AWAITING_CONFIRMATION

        # Ensure prompt shows on entry to state AWAITING_CONFIRMATION
        confirmation_prompt_shown = False

    # If on state AWAITING_CONFIRMATION, wait for user input to confirm sterilization
    elif current_state == AWAITING_CONFIRMATION:
        
        # If a previously annotated image with a grid exists and world_centers is populated, use it for display. Otherwise, use the last known annotated image.
        if last_known_annotated_image is not None and world_centers:
            image_to_display_bgr = last_known_annotated_image

        # If confirmation prompt has not been shown yet, print the prompt
        if not confirmation_prompt_shown:
            print("State: AWAITING_CONFIRMATION. Markers detected. Initiate sterilization? (y/n)")
            confirmation_prompt_shown = True
        
        # Wait for user input.
        # If 'y' or 'Y' are pressed, proceed to sterilization.
        if key == ord('Y') or key == ord('y'):
            print("'y' received. Starting sterilization process.")

            # Reset the confirmation prompt for next entry to this state
            confirmation_prompt_shown = False

            # Set the current cell index to 0 to start monitoring cells
            current_cell_index = 0

            # Change the state to MONITORING_SELECT_CELL
            current_state = MONITORING_SELECT_CELL
        
        # If 'n' or 'N' are pressed, go back to state STANDBY.
        elif key == ord('N') or key == ord('n'):
            print("'n' received. Returning to STANDBY.")

            # Reset the confirmation prompt and other relevant variables
            confirmation_prompt_shown = False

            # Reset the state to STANDBY
            current_state = STANDBY

            # Reset the trajectory and step index
            standby_prompt_shown = False
            active_trajectory = None
            
            # Reset dictionaries
            cam_centers = {}
            world_centers = {}
    
    # If on state MONITORING_SELECT_CELL, select the next cell to move to
    elif current_state == MONITORING_SELECT_CELL:
        
        # Print the current cell index and check if there are world centers available
        print(f"Current cell index: {current_cell_index}")

        # If there are world centers available, proceed to move to the next cell
        if world_centers and current_cell_index < len(world_centers):

            # Get the target world position for the current cell index
            next_target_world = list(world_centers.values())[current_cell_index]
            
            # Convert the target world position to robot coordinates
            x_cell_robot, y_cell_robot, z_cell_robot = convert_webots_to_robot_coordinates(next_target_world)
            
            # Prepare the target position for inverse kinematics
            ik_target_robot = np.array([x_cell_robot, y_cell_robot, z_cell_robot])

            # Get the current joint angles to use as the initial guess for IK
            initial_joint_positions = get_current_joint_angles()
            
            # Solve the inverse kinematics to find the joint positions for the target position
            try:
                target_joint_positions = chain.inverse_kinematics(
                    ik_target_robot,
                    target_orientation=target_orientation_matrix_down,
                    orientation_mode="all",
                    max_iter=IKPY_MAX_ITERATIONS,
                    initial_position=initial_joint_positions
                )
            except Exception as e:

                # If IK fails, print an error message and skip to the next cell
                print(f"IK failed for cell {current_cell_index} at {next_target_world} and orientation: {e}")
                current_cell_index += 1

                # Reset the active trajectory
                active_trajectory = None

                # If we have processed all cells, go to state STANDBY
                if current_cell_index >= len(world_centers):
                    print("Finished all cells or remaining cells failed IK. Returning to STANDBY.")
                    
                    # Prepare for movement to standby position
                    target_position_world = [0, 0.150, 1]

                    x_robot, y_robot, z_robot = convert_webots_to_robot_coordinates(target_position_world)
                    ik_target_robot = np.array([x_robot, y_robot, z_robot])

                    initial_joint_positions = get_current_joint_angles()

                    try:
                        target_joint_positions = chain.inverse_kinematics(
                            ik_target_robot,
                            target_orientation=target_orientation_matrix_down,
                            orientation_mode="all",
                            max_iter=IKPY_MAX_ITERATIONS,
                            initial_position=initial_joint_positions
                        )

                    except Exception as e:

                        # If IK fails, print an error and reset the trajectory.
                        print(f"IK failed for initial detection spot {target_position_world} with orientation: {e}")
                        active_trajectory = None
                        continue
                    
                    # Set the actve trajectory to move to the target position
                    active_trajectory = jtraj(initial_joint_positions, target_joint_positions, 10).q
                    trajectory_step_index = 0

                    # Set the target for check_if_moved to the target position
                    g_target_for_check_if_moved = target_position_world

                    # Set the flag to indicate we are coming from a process
                    coming_from_process = True

                    # Change the state to STANDBY              
                    current_state = STANDBY
                continue 
            
            # Plan the trajectory to move to the target joint positions
            active_trajectory = jtraj(initial_joint_positions, target_joint_positions, 10).q
            
            # Set the trajectory step index to 0
            trajectory_step_index = 0
            
            # Set the target for check_if_moved to the next target world position
            g_target_for_check_if_moved = next_target_world

            # Change the state to MOVING_TO_CELL
            current_state = MOVING_TO_CELL
            print(f"Moving to cell {current_cell_index} (target: {next_target_world})")
        
        else:
            # If there are no more cells to process, print a message and go back to STANDBY. Also, reset the active trajectory.
            print("All cells processed or no cells to process. Going back to STANDBY.")

            # Prepare for movement to standby position
            target_position_world = [0, 0.150, 1]

            x_robot, y_robot, z_robot = convert_webots_to_robot_coordinates(target_position_world)
            ik_target_robot = np.array([x_robot, y_robot, z_robot])

            initial_joint_positions = get_current_joint_angles()

            try:
                target_joint_positions = chain.inverse_kinematics(
                    ik_target_robot,
                    target_orientation=target_orientation_matrix_down,
                    orientation_mode="all",
                    max_iter=IKPY_MAX_ITERATIONS,
                    initial_position=initial_joint_positions
                )

            except Exception as e:
                # If IK fails, print an error and reset the trajectory.
                print(f"IK failed for initial detection spot {target_position_world} with orientation: {e}")
                active_trajectory = None
                continue
                    
            # Set the actve trajectory to move to the target position
            active_trajectory = jtraj(initial_joint_positions, target_joint_positions, 10).q
            trajectory_step_index = 0

            # Set the target for check_if_moved to the target position
            g_target_for_check_if_moved = target_position_world

            # Set the flag to indicate we are coming from a process
            coming_from_process = True

            # Change the state to STANDBY              
            current_state = STANDBY
    
    # If on state MOVING_TO_CELL, move the robot to the target cell
    elif current_state == MOVING_TO_CELL:
        
        # If there's an active trajectory, move the robot along it
        if active_trajectory is not None and trajectory_step_index < len(active_trajectory):
            
            # Get the current joint targets from the active trajectory, according to the trajectory step index
            current_joint_targets = active_trajectory[trajectory_step_index]
            
            # Set the joint positions for the motors
            for i in range(len(motors)):
                motors[i].setPosition(current_joint_targets[i + 1])
            
            # Increase the trajectory step index
            trajectory_step_index += 1
        
        # If the trajectory is complete, check if the robot has arrived at the target position
        else:

            # If there's an active trajectory but we reached the end, ensure the final joint targets are set
            if active_trajectory is not None and trajectory_step_index >= len(active_trajectory):
                 final_joint_targets = active_trajectory[-1]
                 for i in range(len(motors)):
                    motors[i].setPosition(final_joint_targets[i + 1])
            
            # If the robot has arrived at the target position, change state to STERILIZING_CELL
            if g_target_for_check_if_moved is not None and check_if_moved(g_target_for_check_if_moved):
                
                # Print a message indicating arrival at the cell
                print(f"Arrived at cell {current_cell_index}.")

                # Change the state to STERILIZING_CELL
                current_state = STERILIZING_CELL
                
                # Reset the trajectory and step index
                timestep_sum = 0 
                active_trajectory = None
            
            # If there isn't any active trajectory, go back to MONITORING_SELECT_CELL
            elif g_target_for_check_if_moved is None and active_trajectory is None:
                
                # Print a message indicating no active trajectory
                print(f"Error: No target to move to for cell {current_cell_index}. Returning to MONITORING_SELECT_CELL.")

                # Go back to MONITORING_SELECT_CELL to try the next cell
                current_state = MONITORING_SELECT_CELL
                current_cell_index +=1
            
            # If the robot has not arrived yet, continue moving
            else:
                print(f"Waiting to arrive at cell {current_cell_index}...")

    # If on state STERILIZING_CELL, inspect the current cell and sterilize if bacteria are present
    elif current_state == STERILIZING_CELL:
        
        # Print the current cell index and its position
        print(f"Inspecting cell {current_cell_index} at position {list(world_centers.values())  [current_cell_index]}")

        # Get the spotlight nodes for detector and killer lights
        detector_spotlight = supervisor.getFromDef("detectorspotlight")
        killer_spotlight = supervisor.getFromDef("killerspotlight")

        # If the nodes are found, get their intensity fields
        if detector_spotlight:
            intensitydetector_field = detector_spotlight.getField("intensity")
        if killer_spotlight:
            intensitykiller_field = killer_spotlight.getField("intensity")

        # Get the children field from the bacteria group
        children_field = bacteria_group.getField("children")

        # Get the available bacteria in the current cell index
        valore = batteri_dict.get(current_cell_index, [])
        
        if isinstance(valore, int):
            batteri_presenti = [valore]
        else:
            batteri_presenti = valore

        # WARNING: Here we start the sterilization process for the current cell.
        # Step 1 - detection
        print(f"Detecting bacteria into cell by index {current_cell_index}")

        # Turn on the detector spotlight
        if detector_spotlight and intensitydetector_field:
            intensitydetector_field.setSFFloat(20.0)

        # Update the image on the display to show the detector light
        current_raw = camera.getImage()
        image_to_display_bgr = findContours(current_raw)
        update_display(image_to_display_bgr)

        # If there are bacteria in the current cell, print their indices. Also make them visible gradually (simulation of giving them light).
        if len(batteri_presenti) > 0:
            print(f"Found bacteria in cell index {current_cell_index}: {batteri_presenti}")

            # Make bacteria visible gradually
            for i in batteri_presenti:

                # If force reset is triggered, break the loop
                if force_reset:
                    break

                if i < children_field.getCount():
                    bacterium = children_field.getMFNode(i)
                    if bacterium:
                        shape_node = bacterium.getField("children").getMFNode(0)
                        appearance_node = shape_node.getField("appearance").getSFNode()
                        material_node = appearance_node.getField("material").getSFNode()
                        transparency_field = material_node.getField("transparency")

                        for step in range(101):
                            transparency = 1.0 - (step / 100)
                            transparency_field.setSFFloat(transparency)
                            supervisor.step(timeStep)

                            # Make sure we can force reset at any time
                            key = keyboard.getKey()
                            if key == ord('X') or key == ord('x'):
                                print("Force reset command received ('x'). Returning to STANDBY.")
                                force_reset = True
                                break

                            # Update the image every 10 steps to show the visibility change
                            if step % 10 == 0:
                                current_raw = camera.getImage()
                                image_to_display_bgr = findContours(current_raw)
                                update_display(image_to_display_bgr)
        
        # If no bacteria are present, print a message and wait for 1 second with continuous updates
        else:
            print(f"No bacteria found on cell by index {current_cell_index}")

            # Wait for 1 second with continuous updates
            wait_steps = int(10000 / timeStep)
            
            for step in range(wait_steps):
                supervisor.step(timeStep)

                # Make sure we can force reset at any time
                key = keyboard.getKey()
                if key == ord('X') or key == ord('x'):
                    print("Force reset command received ('x'). Returning to STANDBY.")
                    force_reset = True
                    break
                
                # Update the image every 20 steps to show the detector light
                if step % 20 == 0:
                    current_raw = camera.getImage()
                    image_to_display_bgr = findContours(current_raw)
                    update_display(image_to_display_bgr)

        # Step 2 - sterilization (if bacteria are present)
        if len(batteri_presenti) > 0:
            print(f"Sterilizing cell by index {current_cell_index}")

            # Turn off the detection spotlight and turn on the killer spotlight
            if detector_spotlight and intensitydetector_field:
                intensitydetector_field.setSFFloat(0.0)
            if killer_spotlight and intensitykiller_field:
                intensitykiller_field.setSFFloat(20.0)

            # Update the image on the display to show the killer light
            current_raw = camera.getImage()
            image_to_display_bgr = findContours(current_raw)
            update_display(image_to_display_bgr)

            # Wait for 1 second with continuous updates to show the killer light
            wait_steps = int(1000 / timeStep)
            for step in range(wait_steps):
                supervisor.step(timeStep)

                # Make sure we can force reset at any time
                key = keyboard.getKey()
                if key == ord('X') or key == ord('x'):
                    print("Force reset command received ('x'). Returning to STANDBY.")
                    force_reset = True
                    break
                
                # Update the image every 20 steps to show the killer light
                if step % 20 == 0:
                    current_raw = camera.getImage()
                    image_to_display_bgr = findContours(current_raw)
                    update_display(image_to_display_bgr)

            # Make bacteria invisible gradually (simulation of sterilization)
            for i in batteri_presenti:

                # If force reset is triggered, break the loop
                if force_reset:
                    break

                if i < children_field.getCount():
                    bacterium = children_field.getMFNode(i)
                    if bacterium:
                        shape_node = bacterium.getField("children").getMFNode(0)
                        appearance_node = shape_node.getField("appearance").getSFNode()
                        material_node = appearance_node.getField("material").getSFNode()
                        transparency_field = material_node.getField("transparency")

                        for step in range(101):
                            transparency = step / 100.0
                            transparency_field.setSFFloat(transparency)
                            supervisor.step(timeStep)

                            # Make sure we can force reset at any time
                            key = keyboard.getKey()
                            if key == ord('X') or key == ord('x'):
                                print("Force reset command received ('x'). Returning to STANDBY.")
                                force_reset = True
                                break

                            # Update the image every 10 steps to show the visibility change
                            if step % 10 == 0:
                                current_raw = camera.getImage()
                                image_to_display_bgr = findContours(current_raw)
                                update_display(image_to_display_bgr)

            # Turn off the killer spotlight after sterilization
            if killer_spotlight and intensitykiller_field:
                intensitykiller_field.setSFFloat(0.0)

            # Update the image on the display to show the final state after sterilization
            current_raw = camera.getImage()
            image_to_display_bgr = findContours(current_raw)
            update_display(image_to_display_bgr)

            print(f"Cell by index {current_cell_index} has been sterilized successfully.")

            # Remove the bacteria from the world
            del batteri_dict[current_cell_index]
            print(f"Bacteria in cell by index {current_cell_index} removed from the world.")
        
        else:
            # Just turn off the detector spotlight if no bacteria were found
            if detector_spotlight and intensitydetector_field:
                intensitydetector_field.setSFFloat(0.0)

            # Update the image on the display to show the final state after inspection
            current_raw = camera.getImage()
            image_to_display_bgr = findContours(current_raw)
            update_display(image_to_display_bgr)

            print(f"No need to sterilize cell by index {current_cell_index}, process done.")

        # Go to the next cell by incrementing the current cell index
        current_cell_index += 1
        timestep_sum = 0

        # Change the state back to MONITORING_SELECT_CELL to check for the next cell
        current_state = MONITORING_SELECT_CELL
        print(f"Inspection/sterilization of cell {current_cell_index-1} complete. Checking for next cell.")
    
    # At the end of each loop iteration, update the display with the chosen image_to_display_bgr
    update_display(image_to_display_bgr)