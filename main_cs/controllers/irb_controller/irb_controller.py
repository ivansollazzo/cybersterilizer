# Import needed libraries
import numpy as np
import ikpy
from ikpy.chain import Chain
from controller import Supervisor
import tempfile
import math
import cv2
import cv2.aruco as aruco
from scipy.spatial.transform import Rotation as R
import random

# Check if numpy has the disp function, if not, define it
if not hasattr(np, 'disp'):
    np.disp = lambda x, *args, **kwargs: print(x, *args, **kwargs)

#Now we can import the Robotics Toolbox for Python
from roboticstoolbox.tools.trajectory import jtraj

# Constants
IKPY_MAX_ITERATIONS = 100 # Increased for better accuracy

# Target orientation for the end-effector (pointing straight down)
# Assumes robot base Z is up, end-effector Z is tool direction.
# This matrix represents a 180-degree rotation around the base X-axis.
target_orientation_matrix_down = np.array([
    [1,  0,  0],  # End-effector X-axis aligned with robot base X-axis
    [0, -1,  0],  # End-effector Y-axis aligned with robot base -Y-axis
    [0,  0, -1]   # End-effector Z-axis aligned with robot base -Z-axis (pointing down)
])

# Initialize the Webots Supervisor and calculate the time step.
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

# Initialize the arm chain using the URDF file
end_effector_offset = [0.0, 0.0, 0.5]
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
uv_detector = supervisor.getFromDef("UVDetector")
light_killer = supervisor.getFromDef("killer")
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

    # Calculate the area using the cross product of two vectors
    a1 = 0.5 * np.linalg.norm(np.cross(corners[1]-corners[0], corners[2]-corners[0]))
    a2 = 0.5 * np.linalg.norm(np.cross(corners[2]-corners[0], corners[3]-corners[0]))
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
        alpha = i/rows
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
    # x and y axis are inverted because the arm is not aligned with theWebots global axes.
    x = -(targetPosition[1] - armPosition[1])
    y = targetPosition[0] - armPosition[0]
    z = targetPosition[2] - armPosition[2]

    return [x, y, z]

# Function to move the arm to a target position with an optional offset
def move_to_target(targetPosition):

    # Convert Webots global target to robot base frame
    new_target = convert_webots_to_robot_coordinates(targetPosition)

    # Get current joint positions as initial guess
    initial_position = [0] + [m.getPositionSensor().getValue() for m in motors] + [0, 0, 0]

    # Inverse kinematics to reach the new target with offset
    ikResults = chain.inverse_kinematics(new_target, max_iter=IKPY_MAX_ITERATIONS, initial_position=initial_position)

    # Plan the joint-space trajectory
    trajectory = jtraj(initial_position, ikResults, 100)

    # Execute the trajectory
    for pos in trajectory.q:
        for i in range(len(motors)):
            motors[i].setPosition(pos[i + 1])

def spawn_bacteria_in_cells(world_centers, num_bacteria=5):

    if not world_centers:
        print("No cell centers available for bacteria spawning!")
        return
        
    bacteria_group = supervisor.getFromDef("BacteriaGroup")
    if not bacteria_group:
        print("BacteriaGroup not found!")
        return
        
    group_field = bacteria_group.getField("children")

    bacteria_proto = """
    Solid {
      translation 0 0 0
      children [
        Shape {
          appearance Appearance {
            material Material {
              diffuseColor 0.8 0.1 0.1
              transparency 0.0
            }
          }
          geometry Sphere {
            radius 0.005
          }
        }
      ]
      name "bacteria"
      boundingObject Sphere {
        radius 0.005
      }
    }
    """
    
    # Spawna batteri in posizioni casuali vicino ai centri delle celle
    cell_positions = list(world_centers.values())
    
    for _ in range(num_bacteria):
        # Scegli una cella casuale
        base_position = random.choice(cell_positions)
        
        # Aggiungi un piccolo offset casuale
        offset = np.array([
            random.uniform(-0.02, 0.02),  # ±2cm in x
            random.uniform(-0.02, 0.02),  # ±2cm in y
            random.uniform(0, 0.01)       # 0-1cm in z (sopra la superficie)
        ])
        
        bacteria_position = base_position + offset
        
        # Spawna il batterio
        group_field.importMFNodeFromString(-1, bacteria_proto)
        node = group_field.getMFNode(group_field.getCount() - 1)
        node.getField("translation").setSFVec3f(bacteria_position.tolist())
        
    print(f"Spawned {num_bacteria} bacteria in the grid cells")

def get_bacteria_positions():
    """Restituisce una lista delle posizioni dei batteri nel mondo."""
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

# Function to update the Webots display with a given BGR image
def _update_display(image_bgr_to_show):
    img_bgra = cv2.cvtColor(image_bgr_to_show, cv2.COLOR_BGR2BGRA)
    img_buffer = img_bgra.tobytes()
    # Basic check to ensure display is somewhat valid before using
    if display.getWidth() > 0 and display.getHeight() > 0:
        ir = display.imageNew(img_buffer, display.BGRA, img_width, img_height)
        if ir: # Check if imageNew was successful
            display.imagePaste(ir, 0, 0, False)
            display.imageDelete(ir)
        # else:
            # print("Debug: display.imageNew failed") # Optional for debugging
    # else:
        # print("Debug: display not properly initialized or has zero dimensions") # Optional for debugging

# This function processes the raw camera image to detect markers, draw them,
# calculate grid and cell centers, and returns the annotated image and centers.
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
        # draw_grid_overlay returns (image_with_grid_bgr, centers2d)
        final_annotated_image_bgr, centers2d = draw_grid_overlay(final_annotated_image_bgr, quad_pts, rows, cols)
        
        # Interpolate 3D cell centers (these use global cam_positions and world_positions)
        calculated_cam_centers = interpolate_cam_centers(centers2d, rows, cols)
        calculated_world_centers = interpolate_world_centers(quad_ids, rows, cols)

    return final_annotated_image_bgr, calculated_cam_centers, calculated_world_centers

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

# Store the current state of the arm
# STANDBY, DETECTING, MONITORING, STERILIZING = range(4) # Old states
# New states for more granular control over movement and actions
STANDBY, MOVING_TO_DETECT_SPOT, DETECTING_AT_SPOT, AWAITING_CONFIRMATION, MONITORING_SELECT_CELL, MOVING_TO_CELL, STERILIZING_CELL = range(7)
current_state = STANDBY

cam_centers = {}
world_centers = {}
current_cell_index = 0
timestep_sum = 0

# Store the current state of the world
bacteria_spawned = False

# Variables for trajectory management
active_trajectory = None
trajectory_step_index = 0
g_target_for_check_if_moved = None # Stores the world coordinate target for check_if_moved

# Flags to control one-time console messages
standby_prompt_shown = False
confirmation_prompt_shown = False

# Main loop
while supervisor.step(timeStep) != -1:
    raw_camera_image_buffer = camera.getImage() # Get raw image data once per loop
    key = keyboard.getKey()

    # Global stop command ('x' key)
    if key == ord('X') or key == ord('x'):
        print("Stop command received ('x'). Returning to STANDBY.")
        current_state = STANDBY
        standby_prompt_shown = False # Ensure prompt shows on next STANDBY entry
        confirmation_prompt_shown = False # Reset this flag as well
        active_trajectory = None
        trajectory_step_index = 0
        g_target_for_check_if_moved = None
        cam_centers = {}
        world_centers = {}
        current_cell_index = 0
        bacteria_spawned = False # Reset bacteria spawning status
        # Optionally, turn off light_killer if it could be on
        # light_killer.set(0) 
        print("Robot stopped and reset to STANDBY.")
        # Clear remaining key presses for this step
        while key != -1: key = keyboard.getKey()
        # Continue to the next iteration to process the STANDBY state immediately
        continue


    # Start with an image that has ArUco markers drawn if any are visible.
    # detect_and_draw_markers returns (img_marked_bgr, corners, ids)
    image_to_display_bgr, _, _ = detect_and_draw_markers(raw_camera_image_buffer)
    # Now image_to_display_bgr has basic markers.
    # If in DETECTING_AT_SPOT, it will be further processed and replaced.

    if current_state == STANDBY:
        # print("State: STANDBY -> MOVING_TO_DETECT_SPOT") # Original line
        if not active_trajectory: # Only print waiting message if not already in a (aborted) movement
            if not standby_prompt_shown:
                print("State: STANDBY. Press 's' to start detection sequence, or 'x' to stop (if applicable).")
                standby_prompt_shown = True

        if key == ord('S') or key == ord('s'):
            print("Start command ('s') received. Initializing detection sequence.")
            standby_prompt_shown = False # Reset for next STANDBY entry
            # Prepare for movement to initial detection spot
            target_position_world = [0, 0.150, 1] # Initial detection pose in world coords
            
            x_robot, y_robot, z_robot = convert_webots_to_robot_coordinates(target_position_world)
            ik_target_robot = np.array([x_robot, y_robot, z_robot])
            
            initial_joint_positions = [0] + [m.getPositionSensor().getValue() for m in motors] + [0, 0, 0]
            try:
                target_joint_positions = chain.inverse_kinematics(
                    ik_target_robot,
                    target_orientation=target_orientation_matrix_down,
                    orientation_mode="all",
                    max_iter=IKPY_MAX_ITERATIONS,
                    initial_position=initial_joint_positions
                )
            except Exception as e:
                print(f"IK failed for initial detection spot {target_position_world} with orientation: {e}")
                # supervisor.simulationQuit(1) # Or handle error appropriately
                # current_state = STANDBY # Already in STANDBY, or will be due to stop command
                active_trajectory = None # Ensure no active trajectory
                continue # Stay in standby or go to an error state

            active_trajectory = jtraj(initial_joint_positions, target_joint_positions, 30).q # Reduced from 50 to 30 steps
            trajectory_step_index = 0
            
            g_target_for_check_if_moved = target_position_world
            current_state = MOVING_TO_DETECT_SPOT
        # else: remain in STANDBY, waiting for 's'

    elif current_state == MOVING_TO_DETECT_SPOT:
        if active_trajectory is not None and trajectory_step_index < len(active_trajectory):
            current_joint_targets = active_trajectory[trajectory_step_index]
            for i in range(len(motors)):
                motors[i].setPosition(current_joint_targets[i + 1])
            trajectory_step_index += 1
        else: # Trajectory complete or no trajectory
            if active_trajectory is not None and trajectory_step_index >= len(active_trajectory): # Ensure final point is commanded
                 final_joint_targets = active_trajectory[-1]
                 for i in range(len(motors)):
                    motors[i].setPosition(final_joint_targets[i + 1])
            
            # Add a small delay or check for actual movement if check_if_moved is too sensitive
            # For now, directly check if moved
            if g_target_for_check_if_moved is not None and check_if_moved(g_target_for_check_if_moved):
                print("Arrived at detection spot.")
                current_state = DETECTING_AT_SPOT
                active_trajectory = None # Clear trajectory
            elif g_target_for_check_if_moved is None and active_trajectory is None:
                # This case might happen if IK failed in STANDBY and we jumped here senza un target
                print("Error: No target to move to for detection. Returning to STANDBY.")
                current_state = STANDBY
            else:
                print("Waiting to arrive at detection spot...")


    elif current_state == DETECTING_AT_SPOT:
        print("State: DETECTING_AT_SPOT")
        # Perform full detection, get annotated image and cell centers
        # This uses the raw_camera_image_buffer obtained at the start of the loop
        annotated_image, detected_cam_centers, detected_world_centers = run_detection_and_get_annotated_image(raw_camera_image_buffer)
        
        image_to_display_bgr = annotated_image # Show the detailed image for this state
        
        # Update global cam_centers and world_centers for use in this state and potentially others
        cam_centers = detected_cam_centers
        world_centers = detected_world_centers

        if not world_centers: # Check the updated global world_centers
            print("Error: No cell centers found at detection spot. Check camera view/marker setup. Returning to STANDBY.")
            current_state = STANDBY 
            active_trajectory = None # Ensure no trajectory active
            continue

        # Bacteria spawning logic remains, but confirmation comes after
        if not bacteria_spawned:
            spawn_bacteria_in_cells(world_centers, num_bacteria=5) 
            bacteria_spawned = True
        
        # Transition to awaiting confirmation instead of directly to monitoring
        # print("Markers detected. Initiate sterilization? (y/n)") # Moved to AWAITING_CONFIRMATION state
        current_state = AWAITING_CONFIRMATION
        confirmation_prompt_shown = False # Ensure prompt shows on entry to AWAITING_CONFIRMATION
        # current_cell_index = 0 # Reset cell index for new scan # Moved to after 'y' confirmation
        # current_state = MONITORING_SELECT_CELL # Original line
        # print("Detection complete. Transitioning to MONITORING_SELECT_CELL.") # Original line

    elif current_state == AWAITING_CONFIRMATION:
        # It's important to print the prompt every time we are in this state,
        # as the key press might not have happened in the same simulation step.
        # print("State: AWAITING_CONFIRMATION. Markers detected. Initiate sterilization? (y/n)") # Old repetitive print
        if not confirmation_prompt_shown:
            print("State: AWAITING_CONFIRMATION. Markers detected. Initiate sterilization? (y/n)")
            confirmation_prompt_shown = True

        if key == ord('Y') or key == ord('y'):
            print("'y' received. Starting sterilization process.")
            confirmation_prompt_shown = False # Reset for next entry to this state
            current_cell_index = 0 # Reset cell index for new scan
            current_state = MONITORING_SELECT_CELL
            # standby_prompt_shown = False # Not needed here, will be handled if/when returning to STANDBY
            print("Transitioning to MONITORING_SELECT_CELL.")
        elif key == ord('N') or key == ord('n'):
            print("'n' received. Returning to STANDBY.")
            confirmation_prompt_shown = False # Reset for next entry to this state
            current_state = STANDBY
            standby_prompt_shown = False # Ensure prompt shows on entry to STANDBY
            active_trajectory = None
            # Reset relevant variables for a clean standby
            cam_centers = {}
            world_centers = {}
            # bacteria_spawned = False # Decide if bacteria should despawn or stay for next attempt
            # For now, let's keep bacteria if they were spawned, user might want to try again soon.
            # If a full reset is desired on 'n', uncomment the line above.
            print("Robot returning to STANDBY. Press 's' to try again.")

    elif current_state == MONITORING_SELECT_CELL:
        print(f"State: MONITORING_SELECT_CELL. Current cell index: {current_cell_index}")
        if world_centers and current_cell_index < len(world_centers):
            next_target_world = list(world_centers.values())[current_cell_index]
            
            x_cell_robot, y_cell_robot, z_cell_robot = convert_webots_to_robot_coordinates(next_target_world)
            ik_target_robot = np.array([x_cell_robot, y_cell_robot, z_cell_robot])

            initial_joint_positions = [0] + [m.getPositionSensor().getValue() for m in motors] + [0, 0, 0]
            try:
                target_joint_positions = chain.inverse_kinematics(
                    ik_target_robot,
                    target_orientation=target_orientation_matrix_down,
                    orientation_mode="all",
                    max_iter=IKPY_MAX_ITERATIONS,
                    initial_position=initial_joint_positions
                )
            except Exception as e:
                print(f"IK failed for cell {current_cell_index} at {next_target_world} and orientation: {e}")
                current_cell_index += 1 
                active_trajectory = None # Ensure no active trajectory
                if current_cell_index >= len(world_centers):
                    print("Finished all cells or remaining cells failed IK. Returning to STANDBY.")
                    current_state = STANDBY
                # else: continue to MONITORING_SELECT_CELL for the next cell
                continue 

            active_trajectory = jtraj(initial_joint_positions, target_joint_positions, 30).q # Reduced from 50 to 30 steps
            trajectory_step_index = 0
            
            g_target_for_check_if_moved = next_target_world

            current_state = MOVING_TO_CELL
            print(f"Moving to cell {current_cell_index} (target: {next_target_world})")
        else:
            print("All cells processed or no cells to process. Returning to STANDBY.")
            current_state = STANDBY
            active_trajectory = None # Clear trajectory

    elif current_state == MOVING_TO_CELL:
        if active_trajectory is not None and trajectory_step_index < len(active_trajectory):
            current_joint_targets = active_trajectory[trajectory_step_index]
            for i in range(len(motors)):
                motors[i].setPosition(current_joint_targets[i + 1])
            trajectory_step_index += 1
        else: # Trajectory complete
            if active_trajectory is not None and trajectory_step_index >= len(active_trajectory): # Ensure final point is commanded
                 final_joint_targets = active_trajectory[-1]
                 for i in range(len(motors)):
                    motors[i].setPosition(final_joint_targets[i + 1])

            if g_target_for_check_if_moved is not None and check_if_moved(g_target_for_check_if_moved):
                print(f"Arrived at cell {current_cell_index}.")
                current_state = STERILIZING_CELL
                timestep_sum = 0 
                active_trajectory = None # Clear trajectory
            elif g_target_for_check_if_moved is None and active_trajectory is None:
                # This case might happen if IK failed and we jumped here senza un target
                print(f"Error: No target to move to for cell {current_cell_index}. Returning to MONITORING_SELECT_CELL.")
                current_state = MONITORING_SELECT_CELL # Or STANDBY
                current_cell_index +=1 # Try next cell
            else:
                print(f"Waiting to arrive at cell {current_cell_index}...")

    elif current_state == STERILIZING_CELL:

        print(f"Sterilizing cell {current_cell_index} at position {list(world_centers.values())[current_cell_index]}")
        timestep_sum += timeStep
        
        if timestep_sum >= timeStep * 5: # Sterilization duration (5 timesteps)
            current_cell_index += 1
            timestep_sum = 0
            current_state = MONITORING_SELECT_CELL # Go to next cell
            print(f"Sterilization of cell {current_cell_index-1} complete. Checking for next cell.")
    
    # At the end of each loop iteration, update the display with the chosen image_to_display_bgr
    _update_display(image_to_display_bgr)
