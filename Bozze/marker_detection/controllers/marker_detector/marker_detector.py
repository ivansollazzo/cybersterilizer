'''
ArUco Marker Detection and Dynamic Grid Controller
Cybersterilizer Project
Università degli Studi di Palermo
'''

from controller import Supervisor
import cv2
import cv2.aruco as aruco
import numpy as np

# === Initialization ===
supervisor = Supervisor()
time_step = int(supervisor.getBasicTimeStep())

# === Camera Configuration ===
camera = supervisor.getDevice('camera')
camera.enable(time_step)
img_width, img_height = camera.getWidth(), camera.getHeight()
fov = camera.getFov()

# Compute camera intrinsic matrix
cx, cy = img_width / 2, img_height / 2
fx = cx / np.tan(fov / 2)
camera_matrix = np.array([[fx, 0, cx], [0, fx, cy], [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros((4, 1), dtype=np.float32)

# Display for overlay
display = supervisor.getDevice('display')

# === ArUco Settings ===
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
detector_params = aruco.DetectorParameters()
MARKER_SIZE = 0.05  # meters
CELL_AREA_TARGET = 500.0  # cm^2

# Map marker IDs to DEF names in Webots
marker_defs = {0: 'ARUCO_1', 1: 'ARUCO_2', 2: 'ARUCO_3', 3: 'ARUCO_4'}

# Storage for marker positions
cam_positions = {i: np.zeros(3, np.float32) for i in marker_defs}
world_positions = {i: np.zeros(3, np.float32) for i in marker_defs}

# --- Helpers ---

def update_cam_positions(corners, ids):
    """Estimate each marker's 3D position in camera frame."""
    if ids is None:
        return
    _, translations, _ = aruco.estimatePoseSingleMarkers(
        corners, MARKER_SIZE, camera_matrix, dist_coeffs)
    for idx, marker_id in enumerate(ids.flatten()):
        cam_positions[marker_id] = translations[idx][0]


def update_world_positions():
    """Fetch each marker's 3D position from Webots world frame."""
    for m_id, def_name in marker_defs.items():
        node = supervisor.getFromDef(def_name)
        if node:
            world_positions[m_id] = np.array(node.getPosition(), np.float32)


def compute_world_center():
    """Compute centroid of the four markers in world space."""
    pts = np.stack([world_positions[i] for i in sorted(marker_defs)])
    return pts.mean(axis=0)


def print_marker_table():
    """Print camera vs world positions of markers as a table."""
    header = "Marker      |   Cam X   Cam Y   Cam Z ||   Wld X   Wld Y   Wld Z"
    print(header)
    print('-' * len(header))
    for m_id, def_name in marker_defs.items():
        cam = cam_positions[m_id]
        wld = world_positions[m_id]
        print(f"{def_name:<10} | {cam[0]:8.3f}{cam[1]:8.3f}{cam[2]:8.3f} || {wld[0]:8.3f}{wld[1]:8.3f}{wld[2]:8.3f}")


def compute_grid_dims():
    """Determine grid rows and cols so cells are ~CELL_AREA_TARGET cm^2."""
    corners = [world_positions[i] for i in range(4)]
    # Split quad into two triangles for area
    a1 = 0.5 * np.linalg.norm(np.cross(corners[1]-corners[0], corners[2]-corners[0]))
    a2 = 0.5 * np.linalg.norm(np.cross(corners[2]-corners[0], corners[3]-corners[0]))
    total_area_cm2 = (a1 + a2) * 1e4
    num_cells = max(1, int(round(total_area_cm2 / CELL_AREA_TARGET)))
    rows = int(np.floor(np.sqrt(num_cells)))
    cols = int(np.ceil(num_cells / rows))
    return rows, cols


def detect_and_draw_markers(frame_raw):
    """Detect markers, draw axes, return BGR frame and marker data."""
    img_bgra = np.frombuffer(frame_raw, np.uint8).reshape(img_height, img_width, 4)
    img_bgr = cv2.cvtColor(img_bgra, cv2.COLOR_BGRA2BGR)
    corners, ids, _ = aruco.detectMarkers(img_bgr, aruco_dict, parameters=detector_params)
    if ids is not None:
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
            corners, MARKER_SIZE, camera_matrix, dist_coeffs)
        for rv, tv in zip(rvecs, tvecs):
            cv2.drawFrameAxes(img_bgr, camera_matrix, dist_coeffs, rv, tv, MARKER_SIZE/2)
    img_marked = aruco.drawDetectedMarkers(img_bgr.copy(), corners, ids)
    return img_marked, corners, ids


def sort_quad(corners, ids):
    """Order quad corners clockwise and align IDs."""
    pts = np.array([c.reshape(-1,2).mean(axis=0) for c in corners[:4]], np.float32)
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:,1]-center[1], pts[:,0]-center[0])
    order = np.argsort(angles)
    return pts[order], ids.flatten()[order]


def draw_grid_overlay(img, quad_pts, rows, cols):
    """Draw grid lines and compute 2D centers inside the quad."""
    tl, tr, br, bl = quad_pts
    grid = np.zeros((rows+1, cols+1, 2), np.float32)
    for i in range(rows+1):
        alpha = i/rows
        left  = tl*(1-alpha) + bl*alpha
        right = tr*(1-alpha) + br*alpha
        for j in range(cols+1):
            grid[i,j] = left*(1-j/cols) + right*(j/cols)
    # Draw grid lines
    for i in range(rows+1): cv2.polylines(img, [grid[i].astype(int)], False, (0,0,255), 1)
    for j in range(cols+1): cv2.polylines(img, [grid[:,j].astype(int)], False, (0,0,255), 1)
    # Compute and mark cell centers (2D)
    centers2d = {}
    for i in range(rows):
        for j in range(cols):
            pts = grid[[i,i+1,i+1,i],[j,j,j+1,j+1]]
            center = tuple(pts.mean(axis=0).astype(int))
            centers2d[(i,j)] = center
            cv2.circle(img, center, 3, (255,0,0), -1)
    return img, centers2d


def interpolate_cam_centers(centers2d, rows, cols):
    """Interpolate 3D centers in camera frame from marker positions."""
    centers3d = {}
    for (i,j), _ in centers2d.items():
        u, v = (j+0.5)/cols, (i+0.5)/rows
        top = cam_positions[0]*(1-u) + cam_positions[1]*u
        bot = cam_positions[3]*(1-u) + cam_positions[2]*u
        centers3d[(i,j)] = top*(1-v) + bot*v
    return centers3d


def interpolate_world_centers(sorted_ids, rows, cols):
    """Interpolate 3D centers in world frame using bilinear interpolation."""
    p0, p1, p2, p3 = [world_positions[id_] for id_ in sorted_ids]
    centers3d = {}
    for i in range(rows):
        for j in range(cols):
            u, v = (j+0.5)/cols, (i+0.5)/rows
            centers3d[(i,j)] = (
                (1-u)*(1-v)*p0 + u*(1-v)*p1 + u*v*p2 + (1-u)*v*p3
            )
    return centers3d


def print_cell_table(cam_centers, world_centers):
    """Print camera vs world centers of each grid cell."""
    header = "Cell (i,j) |   Cam X   Cam Y   Cam Z ||   Wld X   Wld Y   Wld Z"
    print(header)
    print('-'*len(header))
    for key in sorted(cam_centers):
        c = cam_centers[key]
        w = world_centers[key]
        print(f"({key[0]},{key[1]})        |{c[0]:8.3f}{c[1]:8.3f}{c[2]:8.3f} ||{w[0]:8.3f}{w[1]:8.3f}{w[2]:8.3f}")

# === Main Processing Loop ===
frame_count = 0
prev_raw = None
while supervisor.step(time_step) != -1:
    raw = camera.getImage()
    frame_count += 1
    # Process every 3 frames and if image changed
    if frame_count % 3 == 0 and raw != prev_raw:
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
            # Print tables
            print_marker_table()
            print(f"Grid size: {rows}x{cols}  (target cell area: {CELL_AREA_TARGET} cm^2)")
            print_cell_table(cam_centers, world_centers)
            # Print overall grid center
            wc = compute_world_center()
            print(f"World Grid Center: {wc[0]:.3f}, {wc[1]:.3f}, {wc[2]:.3f}")
        prev_raw = raw
        # Update display
        img_bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        img_buffer = img_bgra.tobytes()
        ir = display.imageNew(img_buffer, display.BGRA, img_width, img_height)
        display.imagePaste(ir, 0, 0, False)
        display.imageDelete(ir)
