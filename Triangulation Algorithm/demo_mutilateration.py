import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

# Define microphone positions
mic_distance = 5  # Distance between microphones in meters
mic_positions = np.array([
    [0, 0, 0],                  # Mic1 at (0, 0, 0)
    [mic_distance, 0, 0],       # Mic2 at (mic_distance, 0, 0)
    [0, mic_distance, 0],       # Mic3 at (0, mic_distance, 0)
    [mic_distance, mic_distance, 0],  # Mic4 at (mic_distance, mic_distance, 0)
])

# Assume an estimated drone position (for example purposes)
true_position = np.array([2.5, 2.5, 1.0]) 

# Calculate distances from the true position to each microphone
distances = np.linalg.norm(mic_positions - true_position, axis=1)

# Function to compute residuals for least squares optimization
def residuals(variables, positions, distances):
    x, y, z = variables
    residuals = []
    for (xi, yi, zi), di in zip(positions, distances):
        calculated_distance = np.sqrt((x - xi)**2 + (y - yi)**2 + (z - zi)**2)
        residual = calculated_distance - di
        residuals.append(residual)
    return residuals

# Initial guess for the drone's position
initial_guess = np.array([mic_distance / 2, mic_distance / 2, 0])

# Perform least squares optimization to estimate the position
result = least_squares(
    residuals,
    initial_guess,
    args=(mic_positions, distances)
)
estimated_position = result.x

# Print estimated position
print(f"Estimated Position: X={estimated_position[0]:.2f}, Y={estimated_position[1]:.2f}, Z={estimated_position[2]:.2f}")

# Create a 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the spheres representing distances from each microphone
u = np.linspace(0, 2 * np.pi, 50)
v = np.linspace(0, np.pi, 50)

# Colors for the spheres
sphere_colors = ['r', 'g', 'b', 'y']

for idx, ((xi, yi, zi), color) in enumerate(zip(mic_positions, sphere_colors)):
    # Create the sphere
    x_sphere = distances[idx] * np.outer(np.cos(u), np.sin(v)) + xi
    y_sphere = distances[idx] * np.outer(np.sin(u), np.sin(v)) + yi
    z_sphere = distances[idx] * np.outer(np.ones(np.size(u)), np.cos(v)) + zi
    # Plot the sphere
    ax.plot_wireframe(
        x_sphere, y_sphere, z_sphere,
        color=color, linewidth=0.5, alpha=0.5, label=f'Sphere {idx+1}'
    )

# Plot microphone positions
ax.scatter(
    mic_positions[:, 0], mic_positions[:, 1], mic_positions[:, 2],
    color='k', s=100, label='Microphones'
)

# Plot true drone position
ax.scatter(
    true_position[0], true_position[1], true_position[2],
    color='c', s=100, label='True Position'
)

# Add a vertical line to indicate the height of the drone
ax.plot(
    [true_position[0], true_position[0]],
    [true_position[1], true_position[1]],
    [0, true_position[2]],
    linestyle='--', color='m', label='Height Indicator'
)

# Set plot labels and title
ax.set_xlabel('X Position (m)')
ax.set_ylabel('Y Position (m)')
ax.set_zlabel('Z Position (m)')
ax.set_title('3D Multilateration Visualization')

# Set plot limits
ax.set_xlim(0, mic_distance)
ax.set_ylim(0, mic_distance)
ax.set_zlim(0, mic_distance)

# Add legend
ax.legend()

plt.show()
