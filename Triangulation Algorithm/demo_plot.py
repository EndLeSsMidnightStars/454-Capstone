import numpy as np
import matplotlib.pyplot as plt

# Set the axes limits from 0 to 1 for x, y, z
mic_distance = 1  # Maximum distance for x and y axes (in meters)
MAX_Z_DISTANCE = 1  # Maximum distance for z-axis (in meters)

# Get x, y, z values from user
x = float(input(f"Enter X position (0 to {mic_distance}): "))
y = float(input(f"Enter Y position (0 to {mic_distance}): "))
z = float(input(f"Enter Z position (0 to {MAX_Z_DISTANCE}): "))

# Constrain x, y, z within bounds
x = np.clip(x, 0, mic_distance)
y = np.clip(y, 0, mic_distance)
z = np.clip(z, 0, MAX_Z_DISTANCE)

# Plot setup
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the point
ax.scatter(x, y, z, c='r', marker='o', s=100, label='Drone Position')

# Add vertical dashed line to indicate height
ax.plot([x, x], [y, y], [0, z], linestyle='--', color='b', label='Height Indicator')

# Set plot limits and labels
ax.set_xlim(0, mic_distance)
ax.set_ylim(0, mic_distance)
ax.set_zlim(0, MAX_Z_DISTANCE)
ax.set_xlabel('X Position (m)')
ax.set_ylabel('Y Position (m)')
ax.set_zlabel('Z Position (m)')
ax.set_title('Live Drone Position')

# Add legend
ax.legend()

# Show the plot
plt.show()