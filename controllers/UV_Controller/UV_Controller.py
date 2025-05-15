"""UV_Controller controller."""

from controller import Supervisor
import math

# Inizializzazione del robot Supervisor
robot = Supervisor()
timestep = int(robot.getBasicTimeStep())

# Funzione per determinare se il target è dentro lo spotlight
def is_in_spotlight(spot_pos, spot_dir, target_pos, spot_angle, spot_range):
    to_target = [target_pos[i] - spot_pos[i] for i in range(3)]
    distance = math.sqrt(sum(x**2 for x in to_target))

    if distance > spot_range:
        return False

    # Normalizza il vettore verso il target
    norm_to_target = [x / distance for x in to_target]
    dot_product = sum(norm_to_target[i] * spot_dir[i] for i in range(3))
    dot_product = max(min(dot_product, 1.0), -1.0)  # Protezione da errori numerici

    angle = math.acos(dot_product)
    
    print(f"🎯 Distance = {distance:.2f}, Angle = {math.degrees(angle):.2f}° (limit: {math.degrees(spot_angle/2):.2f}°)")
    return angle < (spot_angle / 2)

# Recupera nodi da DEF
light_detector = robot.getFromDef("detector")
light_killer = robot.getFromDef("killer")
bacteria = robot.getFromDef("bacteria")
end_effector = robot.getFromDef("EndEffector")

# Controllo presenza nodi
if not (light_detector and light_killer and bacteria and end_effector):
    print("❌ Uno o più nodi DEF non trovati!")
    exit(1)

# Imposta beamWidth a 60° (in radianti)
beam_width_degrees = 60
light_detector.getField("beamWidth").setSFFloat(math.radians(beam_width_degrees))

# Leggi parametri spotlight aggiornati
spot_angle = math.radians(beam_width_degrees)
spot_range = light_detector.getField("radius").getSFFloat()

# Impostazioni iniziali luci
light_detector.getField("intensity").setSFFloat(1.0)
light_killer.getField("intensity").setSFFloat(1.0)
light_detector.getField("color").setSFColor([0.667, 0.333, 1])  # blu/viola
light_killer.getField("color").setSFColor([0, 0.667, 1])        # azzurro

# Loop principale
while robot.step(timestep) != -1:
    spot_pos = end_effector.getPosition()
    target_pos = bacteria.getPosition()

    # Calcola direzione dinamica
    to_target = [target_pos[i] - spot_pos[i] for i in range(3)]
    distance = math.sqrt(sum(x**2 for x in to_target))
    spot_dir = [x / distance for x in to_target] if distance > 0 else [0, 0, -1]

    visible = is_in_spotlight(spot_pos, spot_dir, target_pos, spot_angle, spot_range)
  

    # Gestione trasparenza
    appearance_field = bacteria.getField("children").getMFNode(0).getField("appearance")
    if appearance_field:
        appearance_node = appearance_field.getSFNode()
        if appearance_node:
            transparency_field = appearance_node.getField("transparency")
            if transparency_field:
                transparency_field.setSFFloat(0.0 if visible else 1.0)
