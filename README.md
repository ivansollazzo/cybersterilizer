# Cybersterilizer: la robotica che fa luce sul domani della sanità

Cybersterilizer è un progetto di robotica medica sviluppato come caso di studio per la **sterilizzazione intelligente di aree chirurgiche** tramite doppia luce UV, con particolare attenzione alla **rilevazione automatizzata della carica batterica** e alla **precisione del trattamento localizzato**.

Il sistema si basa su un **braccio robotico antropomorfo a 6 gradi di libertà**, equipaggiato con un end effector progettato ad hoc, ed è simulato tramite l’ambiente **Webots**. Il cuore del progetto è rappresentato dall’integrazione tra computer vision (OpenCV + marker ArUco), cinematica robotica (ikpy), e logica di controllo a stati finiti.

---

## 🔬 Obiettivi del progetto

- Automatizzare il processo di **rilevamento e sterilizzazione** di superfici chirurgiche.
- Migliorare la **ripetibilità** e ridurre l’**errore umano** nelle operazioni di disinfezione.
- Validare l’approccio tramite **simulazione realistica 3D** e prototipazione virtuale.

---

## ⚙️ Architettura del sistema

- **Robot base**: ABB IRB 4600/40 a 6 gradi di libertà.
- **End effector**: modulo cilindrico contenente:
  - Videocamera per visione artificiale.
  - Lampada UV-A a 405 nm per rilevazione batterica.
  - Lampada UV-C a 222 nm per sterilizzazione selettiva.

- **Simulazione e controllo**:
  - Simulazione Webots.
  - Logica a macchina a stati finiti.
  - Cinematica diretta/inversa risolta con **ikpy**.
  - Visione artificiale con **OpenCV** e **marker ArUco**.

---

## 🧠 Funzionalità principali

### 🔍 Rilevamento automatico dell’area
- Riconoscimento marker ArUco (DICT_4X4_250) per delimitare l’area da trattare.
- Generazione dinamica di una **griglia virtuale** di celle su cui operare.

### 🧬 Rilevazione batterica e sterilizzazione mirata
- Attivazione della **luce UV-A** per individuare batteri per fluorescenza.
- In caso di presenza rilevata, attivazione della **luce UV-C** germicida.

### 🧩 Cinematica e navigazione
- Uso della **convenzione Denavit-Hartenberg** per descrivere il robot.
- **Kinematics solver** basato su Jacobiano e pseudoinversa.

### 💡 Visione artificiale e controllo
- Estrazione marker con `cv2.aruco.detectMarkers()`.
- Rilevamento dei batteri con `cv2.findContours()` su maschere HSV.

### 💻 Interazione utente
- Input da tastiera (simulazione): start, conferma, emergenza.
- Visualizzazione live delle griglie e contorni dei batteri.

---

## 🖥️ Tecnologie e strumenti

| Tecnologia | Ruolo |
|------------|-------|
| Webots     | Simulazione robotica 3D |
| Python     | Linguaggio principale |
| OpenCV     | Computer vision |
| ikpy       | Cinematica diretta e inversa |
| Shapr3D    | Modellazione CAD dell’end-effector |
| ArUco      | Riconoscimento marker e stima della posa |

---

## 📎 Autori

- Ambrogi Federico Ennio
- Burgio Gabriele
- Masi Luca
- Sollazzo Ivan

**Docente:** Prof. Antonio Chella
**Istituzione:** Università degli Studi di Palermo
**Corso di Laurea:** Corso di Laurea Magistrale in Ingegneria Informatica

---

## 📚 Bibliografia

Consulta la documentazione allegata (`Documentazione Cybersterilizer.pdf`) per dettagli su fonti scientifiche, modelli matematici e codice di controllo.

---
