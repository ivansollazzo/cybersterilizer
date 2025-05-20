# Import needed libraries
from controller import Robot, Motor, DistanceSensor, PositionSensor
import numpy as np
from ikpy.chain import Chain
from ikpy.link import OriginLink, URDFLink
from scipy.spatial.transform import Rotation as R

# Instantiate a Robot object
robot = Robot()

# Initialize the timestep
timestep = 32

# Define dictionaries for robot and end effector motors
ur_motors = {}
hand_motors = {}

# Get hand motors
hand_motors['finger_1_joint_1'] = robot.getDevice('finger_1_joint_1')
hand_motors['finger_2_joint_1'] = robot.getDevice('finger_2_joint_1')
hand_motors['finger_middle_joint_1'] = robot.getDevice('finger_middle_joint_1')

# Define motor names
motor_names = [
    'shoulder_pan_joint',    # Joint 0
    'shoulder_lift_joint',   # Joint 1
    'elbow_joint',           # Joint 2
    'wrist_1_joint',         # Joint 3
    'wrist_2_joint',         # Joint 4
    'wrist_3_joint'          # Joint 5
]

# Initialize motors
for name in motor_names:
    ur_motors[name] = robot.getDevice(name)
    ur_motors[name].setVelocity(1.0)

# Get the position sensors
position_sensors = {}

for name in motor_names:
    sensor = robot.getDevice(name + "_sensor")
    if sensor:
        position_sensors[name] = sensor
        position_sensors[name].enable(timestep)

# Get the distance sensor
distance_sensor = robot.getDevice('distance_sensor')
distance_sensor.enable(timestep)

'''

Tabella di Denavit-Hartenberg per il robot UR10e:

|---------------------------------------------------|
| Giunto | a (m) | alpha (rad) | d (m) | θ (variabile) |
|--------|-------|---------|-------|---------------|
| 1      | 0     | π/2     | 0.181 | θ₁            |
| 2      | -0.613| 0       | 0     | θ₂+π/2        |
| 3      | -0.571| 0       | 0     | θ₃            |
| 4      | 0     | π/2     | 0.135 | θ₄+π          |
| 5      | 0     | -π/2    | 0.12  | θ₅            |
| 6      | 0     | 0       | 0.16  | θ₆            |
|---------------------------------------------------|

'''

# Load the UR10e robot model from URDF
chain = Chain.from_urdf_file("../../specs/ur10e.urdf", active_links_mask=[False, True, True, True, True, True, True, False])


# Function to convert the robot's kinematics to Webots global frame
def convert_robot_to_webots_global_frame(T_robot_ee):

    # Parametri globali del robot
    translation = [-0.56, -0.1, 0.61]
    axis = np.array([4.66309e-07, -7.19235e-09, 1])
    angle = -1.5707953071795862  # in radianti

    # Normalizza asse
    axis = axis / np.linalg.norm(axis)

    # Costruisci rotazione con scipy
    rot = R.from_rotvec(angle * axis)
    R_mat = rot.as_matrix()  # matrice di rotazione 3x3

    # Costruisci matrice di trasformazione omogenea 4x4
    T_world_robot = np.eye(4)
    T_world_robot[:3, :3] = R_mat
    T_world_robot[:3, 3] = translation

    # Applica trasformazione globale al risultato FK locale
    T_world_ee = T_world_robot @ T_robot_ee

    return T_world_ee

# Function to convert the webot's global frame to the robot's local frame
def convert_webots_global_to_robot_frame(T_world_ee):

    # Parametri globali del robot
    translation = [-0.56, -0.1, 0.61]
    axis = np.array([4.66309e-07, -7.19235e-09, 1])
    angle = -1.5707953071795862  # radianti

    # Normalizza asse
    axis = axis / np.linalg.norm(axis)

    # Costruisci la rotazione usando scipy
    rot = R.from_rotvec(angle * axis)
    R_mat = rot.as_matrix()  # matrice di rotazione 3x3

    # Costruisci la matrice di trasformazione omogenea 4x4
    T_world_robot = np.eye(4)
    T_world_robot[:3, :3] = R_mat
    T_world_robot[:3, 3] = translation

    # Inverti la trasformazione globale robot → mondo
    T_robot_world = np.linalg.inv(T_world_robot)

    # Applica trasformazione inversa: mondo → robot
    T_robot_ee = T_robot_world @ T_world_ee

    return T_robot_ee

# Function to build the transformation matri, given a position and orientation
def build_transformation_matrix(position, axis, angle):
    axis = np.array(axis)
    norm = np.linalg.norm(axis)
    if norm < 1e-10:
        print("ATTENZIONE: Asse di rotazione quasi nullo! Uso asse Z come default")
        axis = np.array([0, 0, 1])
    else:
        axis = axis / norm

    # Costruisci rotazione da vettore rotazione (angolo * asse)
    rot = R.from_rotvec(angle * axis)
    R_mat = rot.as_matrix()  # matrice 3x3

    # Costruisci matrice di trasformazione 4x4
    T = np.eye(4)
    T[:3, :3] = R_mat
    T[:3, 3] = position

    return T

# Function to calculate current joint angles
def get_current_joint_positions():
    
    # First joint position is the base
    joint_positions = [0.0]
    
    # Get the current joint positions from the position sensors
    for name in ur_motors.keys():
        if name in position_sensors and position_sensors[name]:
            position = position_sensors[name].getValue()
            joint_positions.append(position)
    
    # Append the end effector position
    joint_positions.append(0.0)
    
    return joint_positions

# Current robot state
state = "WAITING"

while robot.step(timestep) != -1:
    
    if state == "WAITING":

        '''
        # Get the current joint angles
        joint_angles = get_current_joint_positions()

        print(f"Current joint angles: {joint_angles}")

        # Now convert to the robot's local frame
        T_robot_ee = chain.forward_kinematics(joint_angles)

        print(f"Current end effector position in robot local frame: {T_robot_ee[:3, 3]}")

        # Convert to the Webots global frame
        T_world_ee = convert_robot_to_webots_global_frame(T_robot_ee)

        # Print the coordinates
        print(f"Current end effector position in Webots global frame: {T_world_ee[:3, 3]}")
        '''
        
        if distance_sensor.getValue() < 500:
            for motor in hand_motors.values():
                motor.setPosition(0.85)
            print("Grasping can")
            state = "GRASPING"
    
    elif state == "GRASPING":

        # Get the current joint angles
        joint_angles = get_current_joint_positions()

        # Print the joint matrix
        T_robot_ee = chain.forward_kinematics(joint_angles)

        # Print the coordinates
        print(f"Current end effector position in robot local frame: {T_robot_ee}")

        # Convert to the Webots global frame
        T_world_ee = convert_robot_to_webots_global_frame(T_robot_ee)

        # Print the coordinates
        print(f"Current end effector position in Webots global frame: {T_world_ee[:3,3]}")

        # Example of target point
        position = [0.4, -1.73, 0.71]
        axis = [0, 0, 1]
        angle = -0.2618

        # Build the transformation matrix
        T_target = build_transformation_matrix(position, axis, angle)

        # Convert to the robot's local frame
        T_robot_ee = convert_webots_global_to_robot_frame(T_target)

        # Calculate the joint angles using inverse kinematics
        joint_angles = chain.inverse_kinematics_frame(T_robot_ee)

        # Print the joint angles
        print(f"Joint angles: {joint_angles}")

        # Convert to the Webots global frame from joint angles
        T_robot_ee = chain.forward_kinematics(joint_angles)

        # Convert to the Webots global frame
        T_world_ee = convert_robot_to_webots_global_frame(T_robot_ee)

        # Print the coordinates
        print(f"End effector position in Webots global frame: {T_world_ee[:3, 3]}")

        
        # Set the target joint angles
        for idx, motor in enumerate(ur_motors.values()):
            motor.setPosition(joint_angles[idx+1])
        
        
        print("Rotating arm")
        state = "ROTATING"
    
    elif state == "ROTATING":
        print(f"Current joint angles: {get_current_joint_positions()}")
    
    elif state == "RELEASING":
        for motor in ur_motors.values():
            motor.setPosition(0.0)
        print("Rotating arm back")
        state = "ROTATING_BACK"
    
    elif state == "ROTATING_BACK":
        # Example check, da adattare al tuo sensore
        # if position_sensor.getValue() > -0.1:
        #    print("Waiting for next can")
        #    state = "WAITING"
        pass