import numpy as np
import matplotlib.pyplot as plt

# Parameters
R = 1.0       # Disk radius
L = 12       # Separation (L > 2R)
omega = 1.0   # Angular velocity
k = (2. / R) ** 2  # Decay rate

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
    """Smooth decay function with g(R)=1 and g'(R)=0.
    
    Args:
        r: Distance from disk center
        R: Disk radius
        k: Decay strength parameter
        
    Returns:
        Decay factor (1 inside disk, exponential decay outside)
    """
    return np.where(r <= R, 1.0, np.exp(-k * (r - R)**2))

    # return np.exp(-k * (r - R)**2) if r > R else 1.0

g1 = decay_function(d1, R, k)
g2 = decay_function(d2, R, k)

def angular_velocity(x, y, cx, cy, omega):
    v_x = - omega * (y - cy)
    v_y = omega * (x - cx)
    return v_x, v_y

Vx1, Vy1 = angular_velocity(X, Y, Cx1, Cy1, omega)
Vx2, Vy2 = angular_velocity(X, Y, Cx2, Cy2, omega)

Vx = g1 * Vx1 + g2 * Vx2
Vy = g1 * Vy1 + g2 * Vy2
 
# Calculate field magnitude
magnitude = np.sqrt(Vx**2 + Vy**2)

plt.figure(figsize=(10, 8))

# Plot magnitude as colormap
plt.pcolormesh(X, Y, magnitude, shading='auto', cmap='viridis')
plt.colorbar(label='Velocity magnitude')

# Plot streamlines
plt.streamplot(X, Y, Vx, Vy, density=2, color='white', linewidth=1, arrowsize=1)



# Draw circles around each disk
circle1 = plt.Circle((Cx1, Cy1), R, 
                     color='red', 
                     fill=False, 
                     linewidth=2, 
                     linestyle='--', 
                     label='Disk boundary')
circle2 = plt.Circle((Cx2, Cy2), R, 
                     color='red', 
                     fill=False, 
                     linewidth=2, 
                     linestyle='--')

plt.gca().add_patch(circle1)
plt.gca().add_patch(circle2)


plt.scatter([Cx1, Cx2], [Cy1, Cy2], color='red', s=100, label='Disk centers')
plt.title('Velocity Field Between Two Rotating Disks')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.axis('equal')
plt.savefig("streamlines_with_magnitude.png")
plt.show()
