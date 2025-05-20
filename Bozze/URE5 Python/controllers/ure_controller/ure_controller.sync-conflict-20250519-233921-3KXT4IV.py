# Import needed libraries
from controller import Robot, Motor, DistanceSensor, PositionSensor
from scipy.spatial.transform import Rotation as R
from ikpy.link import URDFLink
import ikpy.chain
import numpy as np

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

# Define the links for IKPY chain
links = [
    URDFLink(
        name="base_link",
        bounds=(0, 0),
        origin_translation=[0, 0, 0],
        origin_orientation=[0, 0, 0],
        rotation=[0, 0, 0],
        use_symbolic_matrix=True
    ),
    URDFLink(
        name="shoulder_pan_joint",
        bounds=(-6.28319, 6.28319),
        origin_translation=[0, 0, 0.181],
        origin_orientation=[0, 0, 0],
        rotation=[0, 0, 1],
        use_symbolic_matrix=True
    ),
    URDFLink(
        name="shoulder_lift_joint",
        bounds=(-6.28319, 6.28319),
        origin_translation=[0, 0.176, 0],
        origin_orientation=[0, 1.5708, 0],
        rotation=[0, 1, 0],
        use_symbolic_matrix=True
    ),
    URDFLink(
        name="elbow_joint",
        bounds=(-3.14159, 3.14159),
        origin_translation=[0, -0.137, 0.613],
        origin_orientation=[0, 0, 0],
        rotation=[0, 1, 0],
        use_symbolic_matrix=True
    ),
    URDFLink(
        name="wrist_1_joint",
        bounds=(-6.28319, 6.28319),
        origin_translation=[0, 0, 0.571],
        origin_orientation=[0, 1.5708, 0],
        rotation=[0, 1, 0],
        use_symbolic_matrix=True
    ),
    URDFLink(
        name="wrist_2_joint",
        bounds=(-6.28319, 6.28319),
        origin_translation=[0, 0.135, 0],
        origin_orientation=[0, 0, 0],
        rotation=[0, 0, 1],
        use_symbolic_matrix=True
    ),
    URDFLink(
        name="wrist_3_joint",
        bounds=(-6.28319, 6.28319),
        origin_translation=[0, 0, 0.12],
        origin_orientation=[0, 0, 0],
        rotation=[0, 1, 0],
        use_symbolic_matrix=True
    ),
    URDFLink(
        name="end_effector",
        bounds=(0, 0),
        origin_translation=[0, 0.16, 0],
        origin_orientation=[0, 0, 0],
        rotation=[0, 0, 0],
        use_symbolic_matrix=True
    )
]

# Create the chain
chain = ikpy.chain.Chain(name='ur10e', links=links, active_links_mask=[False, True, True, True, True, True, True, False])

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

def axis_angle_to_rotation_matrix(axis_angle):
    axis = np.array(axis_angle[:3])
    angle = axis_angle[3]
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c = np.cos(angle)
    s = np.sin(angle)
    C = 1 - c

    R = np.array([
        [c + x*x*C,     x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s, c + y*y*C,     y*z*C - x*s],
        [z*x*C - y*s, z*y*C + x*s, c + z*z*C    ]
    ])
    return R

def transform_target_to_base(target_global, base_position, base_rotation_axis_angle):
    # Calcola la matrice di rotazione 3x3
    R = axis_angle_to_rotation_matrix(base_rotation_axis_angle)

    # Costruisce la trasformazione omogenea base→mondo
    T_base_world = np.eye(4)
    T_base_world[:3, :3] = R
    T_base_world[:3, 3] = base_position

    # Inversa → mondo→base
    T_world_base = np.linalg.inv(T_base_world)

    # Applica trasformazione al target
    target_global_hom = np.array(list(target_global) + [1.0])
    target_base_hom = T_world_base @ target_global_hom

    return target_base_hom[:3]

# Robot base position and orientation in Webots global frame
base_position = [-0.56, -0.1, 0.61]
base_rotation = [4.66309e-07, -7.19235e-09, 1, -1.5707953071795862]

# Current robot state
state = "WAITING"

while robot.step(timestep) != -1:
    
    if state == "WAITING":
        if distance_sensor.getValue() < 500:
            for motor in hand_motors.values():
                motor.setPosition(0.85)
            print("Grasping can")
            state = "GRASPING"
    
    elif state == "GRASPING":
        target_global = [0, 0.69, 0.55]


        target_base = transform_target_to_base(target_global, base_position, base_rotation)
        print("target_base =", target_base)

        target_frame = np.eye(4)
        target_frame[:3, 3] = target_base

        joint_angles = chain.inverse_kinematics_frame(target_frame, initial_position=get_current_joint_positions())

        print("Target joint angles:", joint_angles)

        for idx, name in enumerate(motor_names):
            motor = ur_motors[name]
            motor.setPosition(joint_angles[idx + 1])  # +1 per skip base_link
        
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