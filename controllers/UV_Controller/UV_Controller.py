from controller import Supervisor

robot = Supervisor()
timestep = int(robot.getBasicTimeStep())

# Ottieni i nodi
uv_detector = robot.getFromDef("UVDetector")
bacteria_solid = robot.getFromDef("bacteria")
light_killer = robot.getFromDef("killer")
light_killer.getField("intensity").setSFFloat(0.8)

if uv_detector is None or bacteria_solid is None:
    print("Errore: Nodo non trovato.")
    exit(1)

# Imposta la posizione iniziale
bacteria_solid.getField('translation').setSFVec3f([0, 0, 0.07])

# Ottieni campi grafici
bacteria_shape = bacteria_solid.getField("children").getMFNode(0)
appearance = bacteria_shape.getField("appearance").getSFNode()
transparency_field = appearance.getField("transparency")

# Stati
bacteria_hidden = False
waiting_to_hide = False
wait_time = 0.0
wait_duration = 3.0  # secondi di attesa prima di nascondere

while True:
    step_result = robot.step(timestep)

    if step_result == -1:
        # Fine simulazione: ripristina posizione e visibilità
        bacteria_solid.getField('translation').setSFVec3f([0, 0, 0.07])
        transparency_field.setSFFloat(0.0)
        print("Fine simulazione: batterio riposizionato.")
        break

    if not bacteria_hidden:
        detector_pos = uv_detector.getPosition()
        bacteria_pos = bacteria_solid.getPosition()

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

        dot_product = sum(d * t for d, t in zip(dir_vector, to_bacteria))
        dir_mag = sum(d**2 for d in dir_vector) ** 0.5
        to_bac_mag = sum(t**2 for t in to_bacteria) ** 0.5
        cos_angle = dot_product / (dir_mag * to_bac_mag + 1e-6)

        if distance < 0.5 and cos_angle > 0.99:
            transparency_field.setSFFloat(0.0)
            light_killer.getField("intensity").setSFFloat(5.0)

            if not waiting_to_hide:
                waiting_to_hide = True
                wait_time = 0.0
                print("Batterio individuato. Attesa per eliminazione.")

            else:
                wait_time += timestep / 1000.0

                if wait_time >= wait_duration:
                    bacteria_solid.getField('translation').setSFVec3f([1000, 1000, 1000])
                    bacteria_hidden = True
                    waiting_to_hide = False
                    print("Batterio eliminato")

        else:
            waiting_to_hide = False
            wait_time = 0.0
            light_killer.getField("intensity").setSFFloat(0.0)
            transparency_field.setSFFloat(1.0)

    else:
        light_killer.getField("intensity").setSFFloat(0.0)
