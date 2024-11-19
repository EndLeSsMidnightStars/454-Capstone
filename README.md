# 454-Capstone
# Turret Targeting System and Drone Detection

This repository contains the implementation of a turret targeting system and a drone detection mechanism using microphones, real-time signal processing, and serial communication with motors. The project includes live video feed, motion detection, and trilateration-based position estimation of drones.

---

## Features

- **Turret Targeting System**:
  - Uses OpenCV for video processing.
  - Detects motion and controls turret movement through serial communication.
- **Drone Detection**:
  - Captures audio signals using multiple microphones.
  - Estimates drone position via trilateration using sound pressure levels (SPL).
  - Visualizes the drone position in a 3D plot in real-time.

---

## Requirements

Before running the project, ensure you have Python installed and the required libraries. Use the `install_requirements.py` script to set up the environment.

### Libraries

- `opencv-python` - For video processing.
- `mss` - For screen capturing.
- `pyserial` - For serial communication with motors.
- `numpy` - For numerical computations.
- `matplotlib` - For visualization.
- `scipy` - For signal processing.
- `sounddevice` - For real-time audio capture.

---

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/EndLeSsMidnightStars/454-Capstone.git
