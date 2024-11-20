# drone_detection_gui.py

import sys
import numpy as np
import cv2
from scipy.signal.windows import hamming
from scipy.fft import fft, fftfreq
import sounddevice as sd
from scipy.optimize import curve_fit
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget,
    QGridLayout, QLineEdit, QTextEdit, QMessageBox
)
from PySide6.QtCore import QThread, Signal, Qt

# Real-time audio triangulation and calibration system
class TriangulationSystem:
    def __init__(self):
        self.a_calib = None
        self.b_calib = None
        self.calibrated = False

        # Microphone positions (modify as needed)
        self.mic_positions = np.array([
            [0, 0, 0],
            [5, 0, 0],
            [0, 5, 0],
            [5, 5, 0]
        ])

        # Maximum Z distance
        self.MAX_Z_DISTANCE = 2  # Maximum z distance the system can detect

    def calibrate(self, distances, spl_values):
        """
        Calibrate SPL-to-distance mapping using provided data.
        Returns calibration constants (a, b).
        """
        def model_func(d, a, b):
            return a - 20 * np.log10(d) + b

        params, _ = curve_fit(model_func, distances, spl_values)
        self.a_calib, self.b_calib = params
        self.calibrated = True
        return params

    def get_real_time_mic_input(self, channels, device=None):
        """
        Capture real-time microphone input and calculate SPL values.
        """
        duration = 0.2  # Duration in seconds
        Fs = 48000
        try:
            recording = sd.rec(int(duration * Fs), samplerate=Fs, channels=len(channels), dtype='float64', device=device)
            sd.wait()
        except Exception as e:
            print(f"Error in audio capture: {e}")
            return [0] * len(channels)

        mic_spl_values = []
        for idx, ch in enumerate(channels):
            # Extract and process channel data
            channel_data = recording[:, idx]
            windowed_data = channel_data * hamming(len(channel_data))
            fft_result = np.abs(fft(windowed_data))[:len(windowed_data) // 2]
            freqs = fftfreq(len(windowed_data), d=1/Fs)[:len(windowed_data) // 2]
            freq_mask = (freqs >= 200) & (freqs <= 5000)
            filtered_fft_result = fft_result[freq_mask]
            if len(filtered_fft_result) == 0:
                mic_spl = -80
            else:
                rms_value = np.sqrt(np.mean(filtered_fft_result**2))
                rms_value = max(rms_value, 1e-12)
                mic_spl = 20 * np.log10(rms_value / 20e-6)
            mic_spl_values.append(mic_spl)

        return mic_spl_values

    def estimate_position(self, mic_spl_values):
        """
        Estimate the drone's position based on microphone SPL values.
        """
        if not self.calibrated:
            print("System not calibrated.")
            return (0, 0, 0)

        distances = 10 ** ((self.a_calib + self.b_calib - np.array(mic_spl_values)) / 20)
        distances = np.maximum(distances, 1e-6)  # Avoid zero or negative distances

        # Use trilateration to estimate position
        # Extract positions
        x1, y1, z1 = self.mic_positions[0]
        x2, y2, z2 = self.mic_positions[1]
        x3, y3, z3 = self.mic_positions[2]
        x4, y4, z4 = self.mic_positions[3]

        d1, d2, d3, d4 = distances[:4]

        # Set up equations for trilateration using mics 1, 2, and 3
        # Equation A
        A = np.array([
            [x2 - x1, y2 - y1],
            [x3 - x1, y3 - y1]
        ])

        # Equation B
        B = 0.5 * np.array([
            d1**2 - d2**2 - x1**2 + x2**2 - y1**2 + y2**2,
            d1**2 - d3**2 - x1**2 + x3**2 - y1**2 + y3**2
        ])

        try:
            # Solve for x and y
            position_2d = np.linalg.solve(A, B)
            x, y = position_2d
        except np.linalg.LinAlgError:
            # Singular matrix, cannot solve
            x, y = np.mean([x1, x2, x3]), np.mean([y1, y2, y3])

        # Constrain x and y within bounds
        x = np.clip(x, 0, self.mic_positions[:,0].max())
        y = np.clip(y, 0, self.mic_positions[:,1].max())

        # Calculate z using mic4
        r_xy = np.sqrt((x - x4)**2 + (y - y4)**2)
        d4 = distances[3]
        z_squared = d4**2 - r_xy**2

        if z_squared >= 0:
            z = np.sqrt(z_squared)
        else:
            z = 0  # If negative due to noise, set z to 0

        # Constrain z to be within [0, MAX_Z_DISTANCE]
        z = np.clip(z, 0, self.MAX_Z_DISTANCE)

        # Return the estimated position
        return (x, y, z)

# Thermal Camera Thread
class ThermalCameraThread(QThread):
    frame_ready = Signal(np.ndarray)
    stop_signal = False

    def run(self):
        while not self.stop_signal:
            frame = self.capture_frame()
            self.frame_ready.emit(frame)

    def stop(self):
        self.stop_signal = True

    @staticmethod
    def capture_frame():
        # Replace with actual Lepton camera capture
        frame = np.random.randint(0, 255, (240, 320), dtype=np.uint8)  # Simulated frame
        return cv2.resize(frame, (640, 480))

# Triangulation Thread
class TriangulationThread(QThread):
    position_updated = Signal(tuple)
    stop_signal = False

    def __init__(self, triangulation_system):
        super().__init__()
        self.triangulation_system = triangulation_system

    def run(self):
        self.stop_signal = False
        mic_channels = [1, 2, 3, 4]
        while not self.stop_signal:
            mic_spl_values = self.triangulation_system.get_real_time_mic_input(mic_channels)
            position = self.triangulation_system.estimate_position(mic_spl_values)
            self.position_updated.emit(position)
            self.msleep(200)  # Update every 200ms

    def stop(self):
        self.stop_signal = True

# GUI Class
class DroneDetectionGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Drone Detection System")
        self.setGeometry(100, 100, 800, 600)

        self.triangulation_system = TriangulationSystem()

        # Main Layout
        main_layout = QVBoxLayout()

        # Calibration Section for one mic at distances 1m to 5m
        calibration_label = QLabel("Calibration Inputs for Mic at Distances 1m to 5m")
        calibration_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(calibration_label)

        self.calibration_grid = QGridLayout()
        self.spl_inputs = []
        self.distances = [1, 2, 3, 4, 5]

        for idx, distance in enumerate(self.distances):
            distance_label = QLabel(f"Distance {distance}m SPL:")
            spl_input = QLineEdit()
            self.spl_inputs.append(spl_input)
            self.calibration_grid.addWidget(distance_label, idx, 0)
            self.calibration_grid.addWidget(spl_input, idx, 1)

        main_layout.addLayout(self.calibration_grid)

        calibration_button = QPushButton("Calibrate System")
        calibration_button.clicked.connect(self.calibrate_system)
        main_layout.addWidget(calibration_button)

        # Triangulation Section
        triangulation_button = QPushButton("Start Triangulation")
        triangulation_button.clicked.connect(self.start_triangulation)
        main_layout.addWidget(triangulation_button)

        stop_triangulation_button = QPushButton("Stop Triangulation")
        stop_triangulation_button.clicked.connect(self.stop_triangulation)
        main_layout.addWidget(stop_triangulation_button)

        # Thermal Camera Section
        thermal_button = QPushButton("Start Thermal Camera")
        thermal_button.clicked.connect(self.start_thermal_camera)
        main_layout.addWidget(thermal_button)

        stop_thermal_button = QPushButton("Stop Thermal Camera")
        stop_thermal_button.clicked.connect(self.stop_thermal_camera)
        main_layout.addWidget(stop_thermal_button)

        # Output Display
        self.output_display = QTextEdit()
        self.output_display.setReadOnly(True)
        main_layout.addWidget(self.output_display)

        # Add margins and spacing
        main_layout.setContentsMargins(30, 30, 30, 30)
        main_layout.setSpacing(10)

        # Set layout
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # Thermal Camera Thread
        self.thermal_thread = ThermalCameraThread()
        self.thermal_thread.frame_ready.connect(self.update_thermal_view)

        # Triangulation Thread
        self.triangulation_thread = TriangulationThread(self.triangulation_system)
        self.triangulation_thread.position_updated.connect(self.update_position_display)

    def calibrate_system(self):
        try:
            spl_values = [float(input_box.text()) for input_box in self.spl_inputs]
            distances = self.distances
            params = self.triangulation_system.calibrate(distances, spl_values)
            self.output_display.append(f"Calibration successful: a={params[0]:.2f}, b={params[1]:.2f}")
        except ValueError as e:
            self.output_display.append(f"Calibration error: {e}")
            QMessageBox.warning(self, "Calibration Error", "Please enter valid SPL values for all distances.")

    def start_triangulation(self):
        if not self.triangulation_system.calibrated:
            QMessageBox.warning(self, "Calibration Required", "Please calibrate the system before starting triangulation.")
            return
        if not self.triangulation_thread.isRunning():
            self.triangulation_thread.start()
            self.output_display.append("Triangulation started.")

    def stop_triangulation(self):
        if self.triangulation_thread.isRunning():
            self.triangulation_thread.stop()
            self.triangulation_thread.quit()
            self.triangulation_thread.wait()
            self.output_display.append("Triangulation stopped.")

    def start_thermal_camera(self):
        if not self.thermal_thread.isRunning():
            self.thermal_thread = ThermalCameraThread()
            self.thermal_thread.frame_ready.connect(self.update_thermal_view)
            self.thermal_thread.start()
            self.output_display.append("Thermal camera started.")

    def stop_thermal_camera(self):
        if self.thermal_thread.isRunning():
            self.thermal_thread.stop()
            self.thermal_thread.quit()
            self.thermal_thread.wait()
            cv2.destroyAllWindows()
            self.output_display.append("Thermal camera stopped.")

    def update_thermal_view(self, frame):
        cv2.imshow("Thermal Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.stop_thermal_camera()

    def update_position_display(self, position):
        x, y, z = position
        self.output_display.append(f"Drone Position: X={x:.2f}, Y={y:.2f}, Z={z:.2f}")

# Main Execution
if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Global Stylesheet
    app.setStyleSheet("""
        QWidget {
            background-color: #f0f0f0; /* Light gray background */
            font-family: Arial;
        }
        QPushButton {
            background-color: #007BFF;
            color: white;
            border-radius: 5px;
            padding: 10px;
        }
        QPushButton:hover {
            background-color: #0056b3;
        }
        QLabel {
            font-size: 14px;
            color: #333333;
        }
        QLineEdit {
            color: black;  /* Changes text color to black */
            background-color: white;  /* Optional: ensures the background is white */
            border: 1px solid #cccccc;
            padding: 5px;
            border-radius: 4px;
        }
        QLineEdit:focus {
            border: 1px solid #007BFF;
            background-color: #e6f7ff;
        }
        QTextEdit {
            background-color: white;
            border: 1px solid #cccccc;
            padding: 5px;
            border-radius: 4px;
        }
    """)

    gui = DroneDetectionGUI()
    gui.show()
    sys.exit(app.exec())
