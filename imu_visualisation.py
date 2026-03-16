import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from ahrs.filters import Madgwick
from scipy.spatial.transform import Rotation as R

def get_phone_box(width=0.8, height=1.6, depth=0.1):
    """Returns the 8 vertices of a 3D phone centered at the origin."""
    x = [-width/2, width/2]
    y = [-height/2, height/2]
    z = [-depth/2, depth/2]
    vertices = np.array([[xi, yi, zi] for xi in x for yi in y for zi in z])
    # Define the 6 faces of the box using vertex indices
    faces = [
        [vertices[0], vertices[1], vertices[3], vertices[2]], # Bottom
        [vertices[4], vertices[5], vertices[7], vertices[6]], # Top
        [vertices[0], vertices[1], vertices[5], vertices[4]], # Front
        [vertices[2], vertices[3], vertices[7], vertices[6]], # Back
        [vertices[0], vertices[2], vertices[6], vertices[4]], # Left
        [vertices[1], vertices[3], vertices[7], vertices[5]]  # Right
    ]
    return vertices, faces

def animate_3d_phone(imu_seq):
    """
    imu_seq: Raw array of shape (T, 36) from your catalog.
    """
    # 1. Extract and un-scale raw data 
    # (Check your dataset.py exact scalings. Acc is /10, Mag is /1000)
    acc = imu_seq[:, 0:3] * 10.0
    gyr = imu_seq[:, 12:15] # Assuming no scaling applied to raw gyro in your dataset
    mag = imu_seq[:, 24:27] * 1000.0

    # 2. Sensor Fusion: Convert raw IMU to Quaternions
    madgwick = Madgwick(acc=acc, gyr=gyr, mag=mag)
    Q = madgwick.Q
    
    # 3. Setup 3D Plot
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    # 4. Animation Update Function
    def update(frame):
        ax.clear()
        
        # Keep axes fixed so the camera doesn't jump around
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f"Phone Orientation (Frame {frame})")
        
        # Get current rotation quaternion and apply to phone vertices
        rot = R.from_quat([Q[frame, 1], Q[frame, 2], Q[frame, 3], Q[frame, 0]]) # scipy expects x,y,z,w
        vertices, faces = get_phone_box()
        rotated_faces = [[rot.apply(v) for v in face] for face in faces]
        
        # Draw the phone
        phone = Poly3DCollection(rotated_faces, alpha=0.7, facecolors='cyan', edgecolors='black')
        ax.add_collection3d(phone)
        
        return phone,

    ani = animation.FuncAnimation(fig, update, frames=len(imu_seq), interval=100, blit=False)
    plt.show()