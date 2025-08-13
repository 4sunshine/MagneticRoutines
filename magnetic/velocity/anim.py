import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyArrowPatch

# Parameters
R = 1.0       # Disk radius
L = 12        # Separation (L > 2R)
omega = 1.0   # Angular velocity

resolution = 100

Cx1 = - L / 2
Cy1 = 0

Cx2 = L / 2
Cy2 = 0

# Grid setup
x = np.linspace(-L, L, resolution)
y = np.linspace(-L, L, resolution)
X, Y = np.meshgrid(x, y)

# Distance functions
d1 = np.sqrt((X - Cx1)**2 + (Y - Cy1)**2)  # Distance to left disk
d2 = np.sqrt((X - Cx2)**2 + (Y - Cy2)**2)  # Distance to right disk

def decay_function(r, R, k):
    """Smooth decay function with g(R)=1 and g'(R)=0."""
    return np.where(r <= R, 1.0, np.exp(-k * (r - R)**2))

def angular_velocity(x, y, cx, cy, omega):
    v_x = - omega * (y - cy)
    v_y = omega * (x - cx)
    return v_x, v_y

# Precompute constant velocity fields
Vx1, Vy1 = angular_velocity(X, Y, Cx1, Cy1, omega)
Vx2, Vy2 = angular_velocity(X, Y, Cx2, Cy2, omega)

# Set up figure and axis
fig, ax = plt.subplots(figsize=(10, 8))

# Draw static elements (these won't change)
circle1 = plt.Circle((Cx1, Cy1), R, color='red', fill=False, linewidth=2, linestyle='--')
circle2 = plt.Circle((Cx2, Cy2), R, color='red', fill=False, linewidth=2, linestyle='--')
ax.add_patch(circle1)
ax.add_patch(circle2)
#ax.scatter([Cx1, Cx2], [Cy1, Cy2], color='red', s=100, label='Disk centers')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend()
ax.grid()
ax.axis('equal')

# Initialize plot elements that will change
mesh = ax.pcolormesh(X, Y, np.zeros_like(X), shading='auto', cmap='viridis',vmin=0., vmax=0.5)
cbar = fig.colorbar(mesh, ax=ax, label='Velocity magnitude')

# Define k values
k_values = np.arange(0., 15.5, 0.5)[::-1]
k_values = np.logspace(-2, 2, 60)[::-1]
num_frames = len(k_values)

def update(frame):
    k = k_values[frame]
    
    # Update decay functions
    g1 = decay_function(d1, R, k)
    g2 = decay_function(d2, R, k)

    # Compute velocity field
    Vx = g1 * Vx1 + g2 * Vx2
    Vy = g1 * Vy1 + g2 * Vy2

    # Compute magnitude
    magnitude = np.sqrt(Vx**2 + Vy**2)

    # Update magnitude colormap
    mesh.set_array(magnitude.ravel())
    
    # Clear only the dynamic elements (keep static elements)
    # Remove all collections except the static ones (mesh and circles)
    for artist in ax.collections:
        if artist != mesh and artist != circle1:
            artist.remove()
    for artist in ax.get_children():
        if isinstance(artist, FancyArrowPatch):
            artist.remove()
    # Remove all patches except circles
    # Actually, we don't add patches in update, so this isn't needed
    
    # Remove all streamplot related artists
    for artist in ax.artists + ax.lines:
        if hasattr(artist, '_streamplot') or 'stream' in str(type(artist)).lower():
            try:
                artist.remove()
            except:
                pass
    
    # Draw new streamlines
    ax.streamplot(X, Y, Vx, Vy, density=2, color='white', linewidth=1, arrowsize=1)

    # Update title
    ax.set_title(f'Velocity Field Between Two Rotating Disks (k = {k:.1f})')

# Create animation
ani = FuncAnimation(fig, update, frames=num_frames, blit=False, repeat=False, interval=200)  # 0.4s per frame

# To save as MP4 (requires ffmpeg):
ani.save("rotating_disks_animation_log.mp4", writer='ffmpeg', fps=5)  # 1 frame every 0.4s => 2.5 fps

# To save as GIF:
# ani.save("rotating_disks_animation.gif", writer='pillow', fps=2.5)

plt.tight_layout()
plt.show()