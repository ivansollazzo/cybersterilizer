# Import needed libraries
import numpy as np
import ikpy
from ikpy.chain import Chain
from controller import Supervisor
import tempfile
import math

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

# Main loop
while supervisor.step(timeStep) != -1:
    
    # Get the absolute postion of the target and the arm base.
    targetPosition = [0.13, -0.01, 0.79]
    armPosition = arm.getPosition()

    # Compute the position of the target relatively to the arm.
    # x and y axis are inverted because the arm is not aligned with the Webots global axes.
    x = -(targetPosition[1] - armPosition[1])
    y = targetPosition[0] - armPosition[0]
    z = targetPosition[2] - armPosition[2]

    # Calculate the inverse kinematics of the arm.
    initial_position = [0] + [m.getPositionSensor().getValue() for m in motors] + [0,0]
    ikResults = chain.inverse_kinematics([x, y, z], max_iter=IKPY_MAX_ITERATIONS, initial_position=initial_position)

    # Actuate the arm motors with the IK results.
    for i in range(len(motors)):
        motors[i].setPosition(ikResults[i + 1])
