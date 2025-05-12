"""UV_Controller controller."""

from controller import Supervisor

# Crea istanza del Supervisor
robot = Supervisor()
timestep = int(robot.getBasicTimeStep())

# Recupera i nodi Spotlight tramite DEF
light_detector = robot.getFromDef("detector")
light_killer = robot.getFromDef("killer")

# Controllo di esistenza
if light_detector and light_killer:
    # Modifica intensità
    light_detector.getField("intensity").setSFFloat(0.6)
    light_killer.getField("intensity").setSFFloat(0.0)

    # Modifica colore
             # bianco
    light_detector.getField("color").setSFColor([0.667, 0.333, 1])    # blu
    light_killer.getField("color").setSFColor([0, 0.667, 1])      # rosso

    print("✔️ Spotlights aggiornate!")
else:
    print("❌ Uno o più nodi DEF non trovati!")

# Loop principale (serve per mantenere vivo il controller)
while robot.step(timestep) != -1:
   
    pass
    
class endEffector:
    def __init__(self,direction, threshold,intensity):
        self.direction = direction
        self.threshold = threshold
        self.intensity = intensity
        self.detector = robot.getFromDef("detector")
        self.killer = robot.getFromDef("killer")
        
    def initialize(self):
        if self.detector and self.killer:
           self.detector.getField('direction').setSFVec3f(self.direction)
           if self.threshold < 0.5:
               self.killer.getField('direction').setSFVec3f(self.direction)
               #logica che uccide i batteri
  
           