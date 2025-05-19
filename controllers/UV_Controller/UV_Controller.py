from controller import Supervisor
import random

robot = Supervisor()
timestep = int(robot.getBasicTimeStep())

# Parametri arena
arena = robot.getFromDef("RectangleArena")
arena_size = 1.0  # larghezza e lunghezza (assunta quadrata)
spawn_z = 0.07
num_bacteria = 5
regen_delay = 5.0  # secondi

# Ottieni i nodi esistenti
uv_detector = robot.getFromDef("UVDetector")
light_killer = robot.getFromDef("killer")
bacteria_group = robot.getFromDef("BacteriaGroup")  # nel .wbt: DEF BacteriaGroup Group { children [ ] }

if uv_detector is None or light_killer is None or bacteria_group is None:
    print("Errore: Nodo mancante.")
    exit(1)

# Campo per aggiungere nodi dinamicamente
group_field = bacteria_group.getField("children")

# Prototipo batterio (semplice sfera)
bacteria_proto = """
Solid {
  children [
    Shape {
      appearance Appearance {
        material Material {
          diffuseColor 0.8 0.1 0.1
          transparency 0.0
        }
      }
      geometry Sphere {
        radius 0.02
      }
    }
  ]
  name "bacteria"
  boundingObject Sphere {
    radius 0.02
  }
  
}
"""

# Funzione per generare coordinate randomiche nell’arena
def random_position():
    return [
        random.uniform(-arena_size / 2, arena_size / 2),
        random.uniform(-arena_size / 2, arena_size / 2),
        spawn_z
    ]

# Crea n batteri iniziali
bacteria_list = []
for _ in range(num_bacteria):
    group_field.importMFNodeFromString(-1, bacteria_proto)
    node = group_field.getMFNode(group_field.getCount() - 1)
    node.getField("translation").setSFVec3f(random_position())
    shape = node.getField("children").getMFNode(0)
    material = shape.getField("appearance").getSFNode().getField("material").getSFNode()
    bacteria_list.append({
        "node": node,
        "transparency": material.getField("transparency"),
        "hidden": False,
        "waiting": False,
        "wait_time": 0.0,
        "regen_time": 0.0
    })

# Parametri temporali
wait_duration = 3.0

# Simulazione
while robot.step(timestep) != -1:
    all_hidden = True
    for bac in bacteria_list:
        node = bac["node"]

        if bac["hidden"]:
            bac["regen_time"] += timestep / 1000.0
            if bac["regen_time"] >= regen_delay:
                new_pos = random_position()
                node.getField("translation").setSFVec3f(new_pos)
                bac["transparency"].setSFFloat(0.0)
                bac["hidden"] = False
                bac["regen_time"] = 0.0
                print("Batterio rigenerato in posizione:", new_pos)
            continue  # skip controllo UV

        all_hidden = False

        detector_pos = uv_detector.getPosition()
        bacteria_pos = node.getPosition()

        dx = detector_pos[0] - bacteria_pos[0]
        dy = detector_pos[1] - bacteria_pos[1]
        dz = detector_pos[2] - bacteria_pos[2]
        distance = (dx**2 + dy**2 + dz**2) ** 0.5

        orientation = uv_detector.getOrientation()
        dir_vector = [orientation[2], orientation[5], orientation[8]]
        to_bacteria = [
            bacteria_pos[0] - detector_pos[0],
            bacteria_pos[1] - detector_pos[1],
            bacteria_pos[2] - detector_pos[2]
        ]

        dot = sum(d * t for d, t in zip(dir_vector, to_bacteria))
        dir_mag = sum(d**2 for d in dir_vector) ** 0.5
        to_bac_mag = sum(t**2 for t in to_bacteria) ** 0.5
        cos_angle = dot / (dir_mag * to_bac_mag + 1e-6)

        if distance < 1.0 and cos_angle > 0.99:
            bac["transparency"].setSFFloat(0.0)
            light_killer.getField("intensity").setSFFloat(5.0)

            if not bac["waiting"]:
                bac["waiting"] = True
                bac["wait_time"] = 0.0
                print("Batterio esposto alla luce.")
            else:
                bac["wait_time"] += timestep / 1000.0
                if bac["wait_time"] >= wait_duration:
                    node.getField("translation").setSFVec3f([1000, 1000, 1000])
                    bac["hidden"] = True
                    bac["waiting"] = False
                    bac["wait_time"] = 0.0
                    print("Batterio eliminato.")
        else:
            bac["waiting"] = False
            bac["wait_time"] = 0.0
            bac["transparency"].setSFFloat(1.0)
            light_killer.getField("intensity").setSFFloat(0.0)

    if all_hidden:
        print("Tutti i batteri eliminati. In attesa di rigenerazione...")
