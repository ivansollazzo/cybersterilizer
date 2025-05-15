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

# Grid dimensions (number of cells)
GRID_ROWS = 6
GRID_COLS = 6

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

# Dictionaries to store cell centers
camera_cell_centers = {}  # (row, col) -> (x, y, z) in camera coordinates
world_cell_centers = {}   # (row, col) -> (x, y, z) in world coordinates

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

# Function to sort corners in clockwise order starting from top-left
def sort_corners(centers):
    # First find the centroid
    centroid = np.mean(centers, axis=0)
    
    # Calculate angles relative to centroid
    angles = np.arctan2(centers[:, 1] - centroid[1], centers[:, 0] - centroid[0])
    
    # Sort by angle
    sorted_indices = np.argsort(angles)
    return centers[sorted_indices]

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
    
    # Sort corners in clockwise order
    centroid = centers.mean(axis=0)
    angles = np.arctan2(centers[:, 1] - centroid[1], centers[:, 0] - centroid[0])
    order = np.argsort(angles)
    pts4 = centers[order].astype(int)
    
    # Draw the area outline
    cv2.polylines(image, [pts4], isClosed=True, color=(0, 255, 0), thickness=3)
    
    return image, pts4

# Function to generate grid inside the quadrilateral
def create_grid(image, corners, grid_rows=GRID_ROWS, grid_cols=GRID_COLS):
    """
    Create a grid inside the quadrilateral defined by the four corners.
    Returns the image with grid lines and centers of cells in both image and 3D coordinates.
    """
    # Compute the intersection points of the grid lines
    grid_points_2d = np.zeros((grid_rows + 1, grid_cols + 1, 2), dtype=np.float32)
    
    # Compute the four sides of the quadrilateral
    for i in range(grid_rows + 1):
        # Interpolate along the left and right sides
        alpha = i / grid_rows
        left_point = corners[0] * (1 - alpha) + corners[3] * alpha
        right_point = corners[1] * (1 - alpha) + corners[2] * alpha
        
        # Interpolate between left and right sides to get grid points
        for j in range(grid_cols + 1):
            beta = j / grid_cols
            grid_points_2d[i, j] = left_point * (1 - beta) + right_point * beta
    
    # Draw the grid lines
    # Horizontal lines
    for i in range(grid_rows + 1):
        pts = grid_points_2d[i, :].reshape((-1, 1, 2)).astype(np.int32)
        cv2.polylines(image, [pts], False, (0, 0, 255), thickness=3)
    
    # Vertical lines
    for j in range(grid_cols + 1):
        pts = grid_points_2d[:, j].reshape((-1, 1, 2)).astype(np.int32)
        cv2.polylines(image, [pts], False, (0, 0, 255), thickness=3)
    
    # Calculate and draw cell centers
    cell_centers_2d = {}
    for i in range(grid_rows):
        for j in range(grid_cols):
            # Calculate center of the cell as average of its four corners
            center_x = (grid_points_2d[i, j][0] + grid_points_2d[i, j+1][0] + 
                     grid_points_2d[i+1, j][0] + grid_points_2d[i+1, j+1][0]) / 4
            center_y = (grid_points_2d[i, j][1] + grid_points_2d[i, j+1][1] + 
                     grid_points_2d[i+1, j][1] + grid_points_2d[i+1, j+1][1]) / 4
            
            center = (int(center_x), int(center_y))
            cell_centers_2d[(i, j)] = center
            
            # Draw the center of the cell
            cv2.circle(image, center, 5, (255, 0, 0), -1)
    
    return image, cell_centers_2d, grid_points_2d

# Function to calculate 3D cell centers in camera coordinates
def calculate_3d_cell_centers(cell_centers_2d, ids):
    """
    Estimate the 3D coordinates of cell centers in camera frame
    based on the known 3D positions of the ArUco markers.
    """
    if ids is None or len(ids) < 4:
        return {}
    
    # Ensure we have markers 0, 1, 2, and 3
    marker_ids = [id[0] for id in ids]
    if not all(i in marker_ids for i in range(4)):
        return {}
    
    # Get the 3D positions of the markers in camera coordinates
    marker_positions_3d = [camera_marker_positions[i] for i in range(4)]
    
    # Calculate 3D cell centers based on bilinear interpolation of marker positions
    cell_centers_3d = {}
    
    # For each cell
    for (row, col), (center_x, center_y) in cell_centers_2d.items():
        # Calculate normalized position within the grid (0-1 range)
        u = col / GRID_COLS
        v = row / GRID_ROWS
        
        # Bilinear interpolation in 3D
        # First interpolate top and bottom positions
        top_pos = marker_positions_3d[0] * (1 - u) + marker_positions_3d[1] * u
        bottom_pos = marker_positions_3d[3] * (1 - u) + marker_positions_3d[2] * u
        
        # Then interpolate between top and bottom
        cell_center_3d = top_pos * (1 - v) + bottom_pos * v
        
        # Store result
        cell_centers_3d[(row, col)] = cell_center_3d
    
    return cell_centers_3d

# Function to convert 3D points from camera to world coordinates
def camera_to_world_coordinates(camera_points):
    """
    Convert points from camera coordinates to world coordinates
    based on the known positions of markers in both coordinate systems.
    """
    # This is a simplified approach using transformation estimation
    world_points = {}
    
    # Only proceed if we have positions for all markers
    if len(camera_marker_positions) < 4 or len(world_marker_positions) < 4:
        return world_points
    
    # Get the camera and world positions of the markers
    camera_positions = np.array([camera_marker_positions[i] for i in range(4)])
    world_positions = np.array([world_marker_positions[i] for i in range(4)])
    
    # Calculate centroid of points in both coordinate systems
    camera_centroid = np.mean(camera_positions, axis=0)
    world_centroid = np.mean(world_positions, axis=0)
    
    # Center the points
    camera_centered = camera_positions - camera_centroid
    world_centered = world_positions - world_centroid
    
    # Calculate the cross-covariance matrix
    H = np.dot(camera_centered.T, world_centered)
    
    # SVD decomposition
    U, _, Vt = np.linalg.svd(H)
    
    # Calculate rotation matrix
    R = np.dot(Vt.T, U.T)
    
    # Check for reflection case
    if np.linalg.det(R) < 0:
        Vt[-1,:] *= -1
        R = np.dot(Vt.T, U.T)
    
    # Calculate translation
    t = world_centroid - np.dot(R, camera_centroid)
    
    # Convert each camera point to world coordinates
    for (row, col), point in camera_points.items():
        world_points[(row, col)] = np.dot(R, point) + t
    
    return world_points

# Function to print cell centers
def print_cell_centers():
    print("\n--- Cell Center Coordinates ---")
    for (row, col) in camera_cell_centers.keys():
        camera_pos = camera_cell_centers[(row, col)]
        world_pos = world_cell_centers.get((row, col), np.array([0, 0, 0]))
        print(f"Cell ({row}, {col}):")
        print(f"  Camera coordinates: {camera_pos}")
        print(f"  World coordinates: {world_pos}")

if __name__ == "__main__":
    # Frame processing control variables
    frame_counter = 0
    last_raw = None
    image_to_display = np.zeros((height, width, 4), dtype=np.uint8)

    # Main loop
    while supervisor.step(TIME_STEP) != -1:
        raw = camera.getImage()
        frame_counter += 1

        if frame_counter % 3 == 0 and raw != last_raw:
            # Detect markers
            image_markers, corners, ids = detect_markers(raw, width, height)

            if ids is not None and len(ids) >= 4:
                # Draw area
                image_area, sorted_corners = draw_area(image_markers.copy(), corners, ids)
                
                # Create grid and get cell centers in 2D
                image_grid, cell_centers_2d, grid_points = create_grid(image_area.copy(), sorted_corners)
                image_to_display = image_grid
                
                # Update marker positions
                update_world_marker_positions()
                update_camera_marker_positions(corners, ids)
                
                # Calculate 3D cell centers in camera coordinates
                camera_cell_centers = calculate_3d_cell_centers(cell_centers_2d, ids)
                
                # Convert to world coordinates
                world_cell_centers = camera_to_world_coordinates(camera_cell_centers)
                
                # Print marker positions and cell centers
                print_marker_positions(corners, ids)
                print_cell_centers()
            else:
                image_to_display = image_markers
            
            # Update display in Webots
            ir = display.imageNew(image_to_display.tobytes(), display.BGRA, width, height)
            display.imagePaste(ir, 0, 0, False)
            display.imageDelete(ir)

            last_raw = raw