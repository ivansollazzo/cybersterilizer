# Cybersterilizer: Shining a Light on the Future of Healthcare

Cybersterilizer is a medical robotics project developed as a case study for **intelligent sterilization of surgical areas** using dual UV light, with a focus on **automated bacterial load detection** and **precise localized treatment**.

The system is built around a **6-degree-of-freedom anthropomorphic robotic arm** equipped with a custom-designed end effector, and is simulated using the **Webots** environment. At its core lies the integration of computer vision (OpenCV + ArUco markers), robotic kinematics (ikpy), and finite state machine control logic.

---

## 🔬 Project Goals

* Automate the **detection and sterilization** of surgical surfaces.
* Improve **repeatability** and reduce **human error** in disinfection procedures.

---

## ⚙️ System Architecture

* **Base robot**: ABB IRB 4600/40 with 6 degrees of freedom.

* **End effector**: cylindrical module containing:

  * Camera for computer vision.
  * 405 nm UV-A lamp for bacterial detection.
  * 222 nm UV-C lamp for selective sterilization.

* **Simulation and control**:

  * Webots simulation.
  * Finite state machine logic.
  * Forward/inverse kinematics solved with **ikpy**.
  * Computer vision with **OpenCV** and **ArUco markers**.

---

## 🧠 Main Features

### 🔍 Automatic Area Detection

* ArUco marker recognition (DICT\_4X4\_250) to define the treatment area.
* Dynamic generation of a **virtual grid** of cells to process.

### 🧬 Bacterial Detection and Targeted Sterilization

* Activation of **UV-A light** to detect bacteria via fluorescence.
* Upon detection, activation of germicidal **UV-C light**.

### 🧩 Kinematics and Navigation

* Use of the **Denavit-Hartenberg convention** to describe the robot.
* **Kinematics solver** based on the Jacobian and its pseudoinverse.

### 💡 Computer Vision and Control

* Marker extraction with `cv2.aruco.detectMarkers()`.
* Bacteria detection with `cv2.findContours()` on HSV masks.

### 💻 User Interaction

* Keyboard input (simulation): start, confirm, emergency stop.
* Live display of grids and detected bacterial contours.

---

## 🖥️ Technologies and Tools

| Technology | Role                                   |
| ---------- | -------------------------------------- |
| Webots     | 3D robotics simulation                 |
| Python     | Main programming language              |
| OpenCV     | Computer vision                        |
| ikpy       | Forward and inverse kinematics         |
| Shapr3D    | CAD modeling of the end effector       |
| ArUco      | Marker recognition and pose estimation |

---

## 📎 Authors

* Ambrogi Federico Ennio
* Burgio Gabriele
* Masi Luca
* Sollazzo Ivan

**Instructor:** Prof. Antonio Chella
**Institution:** University of Palermo
**Degree Program:** Master’s Degree in Computer Engineering

---

## 📚 References

See the attached documentation (`Documentazione Cybersterilizer.pdf`) for details on scientific sources, mathematical models, and control code.
