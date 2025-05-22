# Import needed libraries
import numpy as np
import ikpy
from ikpy.chain import Chain
from controller import Supervisor
import tempfile
import math
import cv2
import cv2.aruco as aruco

# Constants
IKPY_MAX_ITERATIONS = 4

# Initialize the Webots Supervisor and calculate the time step.
supervisor = Supervisor()
timeStep = int(4 * supervisor.getBasicTimeStep())

# Create the arm chain from the URDF
filename = None
with tempfile.NamedTemporaryFile(suffix='.urdf', delete=False) as file:
    filename = file.name
    file.write(supervisor.getUrdf().encode('utf-8'))

# Initialize the arm chain using the URDF file
chain = Chain.from_urdf_file(filename, active_links_mask=[False, True, True, True, True, True, True, False, False])

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

# Store the current state of the arm
STANDBY, DETECTING, MONITORING, STERILIZING = range(4)
current_state = STANDBY

# Store the position of the cells
world_cells_centers = []

# Function to move the arm to a target position
def move_to_target(targetPosition):
    
    # Get the absolute postion of the target and the arm base.
    armPosition = arm.getPosition()

    # Compute the position of the target relatively to the arm.
    # x and y axis are inverted because the arm is not aligned with theWebots global axes.
    x = -(targetPosition[1] - armPosition[1])
    y = targetPosition[0] - armPosition[0]
    z = targetPosition[2] - armPosition[2]

    # Calculate the inverse kinematics of the arm.
    initial_position = [0] + [m.getPositionSensor().getValue() for m in motors] + [0,0]
    ikResults = chain.inverse_kinematics([x, y, z], max_iter=IKPY_MAX_ITERATIONS,   initial_position=initial_position)

    # Actuate the arm motors with the IK results.
    for i in range(len(motors)):
        motors[i].setPosition(ikResults[i + 1])
        
    # Check if the arm has reached the target position.
    if all(abs(m.getPositionSensor().getValue() - ikResults[i + 1]) < 0.01 for i, m in enumerate(motors)):
        return True
    
    return False

def detect_markers():

    # Get the camera image
    raw = camera.getImage()

    # Detect and draw ArUco
    img, corners, ids = detect_and_draw_markers(raw)
    
    if ids is not None and len(ids) >= 4:
        
        # Update marker positions
        update_world_positions()
        update_cam_positions(corners, ids)
        
        # Sort quad corners and IDs
        quad_pts, quad_ids = sort_quad(corners, ids)
        
        # Compute grid layout
        rows, cols = compute_grid_dims()
        
        # Overlay grid and get 2D centers
        img, centers2d = draw_grid_overlay(img, quad_pts, rows, cols)
        
        # Interpolate 3D centers
        cam_centers   = interpolate_cam_centers(centers2d, rows, cols)
        world_centers = interpolate_world_centers(quad_ids, rows, cols)
        
        # Store the world centers of the cells
        world_cells_centers.clear()
        
        for val in world_centers.values():
            world_cells_centers.append(val)

        '''
        # Print tables
        print_marker_table()
        print(f"Grid size: {rows}x{cols}  (target cell area{CELL_AREA_TARGET} cm^2)")
        print_cell_table(cam_centers, world_centers)
        
        # Print overall grid center
        wc = compute_world_center()
        print(f"World Grid Center: {wc[0]:.3f}, {wc[1]:.3f}, {wc[2]:.3f}")
        '''

    # Update display
    img_bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    img_buffer = img_bgra.tobytes()
    
    ir = display.imageNew(img_buffer, display.BGRA, img_width, img_height)
    display.imagePaste(ir, 0, 0, False)
    display.imageDelete(ir)

# Main loop
while supervisor.step(timeStep) != -1:

    if current_state == STANDBY:
        current_state = DETECTING
    
    elif current_state == DETECTING:
        
        # Move to target position
        moved = move_to_target([0, 0.12, 1])

        if moved:
            detect_markers()
            current_state = MONITORING

    elif current_state == MONITORING:
        #move_to_target([0.088, -0.21032351, 1])
        pass

    elif current_state == STERILIZING:
        pass