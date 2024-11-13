import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hamming
import sounddevice as sd
from scipy.fft import fft, fftfreq
from scipy.optimize import curve_fit
from mpl_toolkits.mplot3d import Axes3D

# Simulation parameters
demo_mode = False  # Set to False to capture from multiple microphones
Fs = 48000  # Sampling frequency (Hz)
freq_range = (200, 5000)  # Frequency range for analysis (in Hz)
NO_SIGNAL_THRESHOLD = 60  # Threshold in dB to determine if there is no significant signal
SMOOTHING_FACTOR = 0.8  # Smoothing factor for exponential moving average

# Distance between microphones (variable)
mic_distance = 5  # Distance between microphones in meters (can be changed)

# Maximum Z distance
MAX_Z_DISTANCE = 2  # Maximum z distance the system can detect

# Microphone positions (variables)
mic_positions = np.array([
    [0, 0, 0],  # Mic1 at (0, 0, 0)
    [mic_distance, 0, 0],  # Mic2 at (mic_distance, 0, 0)
    [0, mic_distance, 0],  # Mic3 at (0, mic_distance, 0)
    [mic_distance, mic_distance, 0],  # Mic4 at (mic_distance, mic_distance, 0)
])

# Calibration data (to be updated with actual measurements)
# SPL readings at known distances (in meters)
calibration_distances = np.array([1, 2, 3, 4, 5])  # Example distances
calibration_spl_values = np.array([68, 65, 60, 59, 58])  # Corresponding SPL readings

# Calibration constants (to be determined through calibration)
# These will be updated by the calibrate_spl_to_distance function
calibration_params = None  # Placeholder for calibration parameters (a, b)
tuning_factor = 1.0  # Tuning factor 'k' for distance estimation

# Function for calibration
def calibrate_spl_to_distance(spl_values, distances):
    """
    Calibrate the relationship between SPL and distance.
    Returns calibration constants a and b.
    """
    def model_func(d, a, b):
        return a - 20 * np.log10(d) + b

    # Use curve fitting to find the best-fit parameters
    params, _ = curve_fit(model_func, distances, spl_values)
    return params  # Returns (a, b)

# Perform calibration using the provided calibration data
calibration_params = calibrate_spl_to_distance(calibration_spl_values, calibration_distances)
a_calib, b_calib = calibration_params

# Function to estimate distance from SPL using calibration and tuning factor
def estimate_distance_from_spl(spl, a, b, k=1.0):
    """
    Estimate the distance from SPL using calibrated parameters and tuning factor.
    """
    distance = k * 10 ** ((a + b - spl) / 20)
    return distance

# Function to estimate drone position based on trilateration
def estimate_position(mic_spl_values):
    mic_spl_values = np.array(mic_spl_values)

    # Check if all signals are below the threshold
    if np.all(mic_spl_values <= NO_SIGNAL_THRESHOLD):
        return np.array([mic_distance / 2, mic_distance / 2, 0])  # Default to center if no signal

    # Estimate distances using the calibrated function and tuning factor
    distances = estimate_distance_from_spl(mic_spl_values, a_calib, b_calib, tuning_factor)

    # Avoid negative distances
    distances = np.maximum(distances, 1e-6)

    # Extract positions
    x1, y1, z1 = mic_positions[0]
    x2, y2, z2 = mic_positions[1]
    x3, y3, z3 = mic_positions[2]
    x4, y4, z4 = mic_positions[3]

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
        x, y = mic_distance / 2, mic_distance / 2  # Default to center

    # Constrain x and y within bounds
    x = np.clip(x, 0, mic_distance)
    y = np.clip(y, 0, mic_distance)

    # Calculate z using mic4
    r_xy = np.sqrt((x - x4)**2 + (y - y4)**2)
    d4 = distances[3]
    z_squared = d4**2 - r_xy**2

    if z_squared >= 0:
        z = np.sqrt(z_squared)
    else:
        z = 0  # If negative due to noise, set z to 0

    # Constrain z to be within [0, MAX_Z_DISTANCE]
    z = np.clip(z, 0, MAX_Z_DISTANCE)

    # Return the estimated position
    return np.array([x, y, z])

# Function to capture real-time microphone input and calculate SPL
def get_real_time_mic_input(channels, device=None):
    duration = 0.2  # Duration in seconds
    try:
        # Record from specified channels using mapping
        recording = sd.rec(
            int(duration * Fs),
            samplerate=Fs,
            channels=len(channels),
            dtype='float64',
            device=device,
            mapping=channels
        )
        sd.wait()  # Wait until the recording is finished
        audio_time_data = recording  # Shape: (samples, len(channels))
    except Exception as e:
        print(f"Error in audio capture: {e}")
        return [NO_SIGNAL_THRESHOLD] * len(channels)

    mic_spl_values = []
    for idx in range(len(channels)):
        # Extract channel data
        channel_data = audio_time_data[:, idx]
        # Apply Hamming window
        windowed_data = channel_data * hamming(len(channel_data))
        # Apply FFT
        fft_result = fft(windowed_data)
        fft_magnitude = np.abs(fft_result)[:len(windowed_data) // 2]
        freqs = fftfreq(len(windowed_data), d=1/Fs)[:len(windowed_data) // 2]
        # Filter to desired frequency range
        freq_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
        filtered_fft_magnitude = fft_magnitude[freq_mask]
        # Compute RMS of the filtered signal
        if len(filtered_fft_magnitude) == 0:
            mic_spl = NO_SIGNAL_THRESHOLD  # No significant signal detected
        else:
            rms_value = np.sqrt(np.mean(filtered_fft_magnitude**2))
            # Avoid log of zero
            rms_value = max(rms_value, 1e-12)
            # Calculate SPL value relative to a reference RMS value
            mic_spl = 20 * np.log10(rms_value / 20e-6)  # Reference value set to 20 µPa (threshold of hearing)
        mic_spl_values.append(mic_spl)
    return mic_spl_values

# Placeholder driver function for targeting system
def targeting_system_driver(position):
    # Currently does nothing
    pass

# Function to start the live plot
def start_live_plot():
    input("Press Enter to start calculating position...")

    # List available host APIs
    print("Available Host APIs:")
    hostapis = sd.query_hostapis()
    for idx, api in enumerate(hostapis):
        print(f"Host API {idx}: {api['name']}")

    # Select ASIO host API
    hostapi_index = None
    for idx, api in enumerate(hostapis):
        if 'ASIO' in api['name']:
            hostapi_index = idx
            break
    if hostapi_index is None:
        print("ASIO host API not found. Please ensure ASIO drivers are installed.")
        return
    print(f"Using Host API {hostapi_index}: {hostapis[hostapi_index]['name']}")

    # List devices under ASIO host API
    print("\nAvailable Devices under ASIO:")
    devices = sd.query_devices()
    asio_devices = []
    for idx, dev in enumerate(devices):
        if dev['hostapi'] == hostapi_index:
            print(f"Device {idx}: {dev['name']}, Max Input Channels: {dev['max_input_channels']}")
            asio_devices.append((idx, dev))

    if not asio_devices:
        print("No devices found under ASIO host API.")
        return

    device_index = int(input("Enter the device index for your microphone input device: "))
    device_info = sd.query_devices(device_index, 'input')
    num_input_channels = device_info['max_input_channels']

    print("Detailed device info:")
    print(device_info)

    if not demo_mode:
        if num_input_channels < 4:
            print(f"Selected device supports only {num_input_channels} input channels. Need at least 4 channels.")
            return
        # Use one-based indexing for ASIO devices
        channels = [1, 2, 3, 4]
    else:
        if num_input_channels < 1:
            print(f"Selected device supports only {num_input_channels} input channels. Need at least 1 channel.")
            return
        channels = [1]  # Only one channel in demo mode

    # Live plot setup
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Initial smoothed position
    smoothed_position = np.array([mic_distance / 2, mic_distance / 2, 0])

    # Create initial scatter plot with one point
    sc = ax.scatter(smoothed_position[0], smoothed_position[1], smoothed_position[2], c='r', marker='o', label='Drone Position')

    # Add vertical dashed line to indicate height
    line, = ax.plot([smoothed_position[0], smoothed_position[0]],
                    [smoothed_position[1], smoothed_position[1]],
                    [0, smoothed_position[2]], linestyle='--', color='b', label='Height Indicator')

    ax.set_xlim(0, mic_distance)
    ax.set_ylim(0, mic_distance)
    ax.set_zlim(0, MAX_Z_DISTANCE)
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_zlabel('Z Position (m)')
    ax.set_title('Live Drone Position')

    # Add legend
    ax.legend()

    # Function to handle plot close event
    def on_close(event):
        nonlocal running
        running = False

    fig.canvas.mpl_connect('close_event', on_close)

    # Main loop for live plotting
    t = 0
    running = True
    try:
        while running:
            t += 0.2  # Update interval
            if demo_mode:
                # Demo mode: Use real-time input for mic1 and fixed values for other mics
                mic_spl_values = get_real_time_mic_input(channels, device=device_index)
                mic_spl_values.extend([NO_SIGNAL_THRESHOLD] * (4 - len(mic_spl_values)))  # Fill remaining mics
            else:
                # Capture real-time input from four microphones
                mic_spl_values = get_real_time_mic_input(channels, device=device_index)
                if len(mic_spl_values) < 4:
                    mic_spl_values.extend([NO_SIGNAL_THRESHOLD] * (4 - len(mic_spl_values)))

            estimated_position = estimate_position(mic_spl_values)

            # Apply smoothing
            smoothed_position = SMOOTHING_FACTOR * smoothed_position + (1 - SMOOTHING_FACTOR) * estimated_position

            # Print the updated SPL values for all mics
            print(f"Updated SPL values: Mic1: {mic_spl_values[0]:.2f} dB, Mic2: {mic_spl_values[1]:.2f} dB, Mic3: {mic_spl_values[2]:.2f} dB, Mic4: {mic_spl_values[3]:.2f} dB")

            # Print the estimated position
            print(f"Estimated Position - X: {smoothed_position[0]:.2f} m, Y: {smoothed_position[1]:.2f} m, Z: {smoothed_position[2]:.2f} m")

            # Call the targeting system driver with the estimated position
            targeting_system_driver(smoothed_position)

            # Update live plot
            sc._offsets3d = (np.array([smoothed_position[0]]), np.array([smoothed_position[1]]), np.array([smoothed_position[2]]))

            # Update vertical line
            line.set_data([smoothed_position[0], smoothed_position[0]],
                          [smoothed_position[1], smoothed_position[1]])
            line.set_3d_properties([0, smoothed_position[2]])

            plt.draw()
            plt.pause(0.2)  # Adjust to control the speed of the live update
    except KeyboardInterrupt:
        print("Live plotting interrupted by user.")
    finally:
        plt.ioff()
        plt.show()

# Start live plotting
start_live_plot()
