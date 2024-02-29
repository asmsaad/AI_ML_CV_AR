# from ultralytics import YOLO
# import cv2

# # load yolov8 model
# model = YOLO('yolov8n.py')

# # load video
# video_path = 'cross_road.mp4'
# cap = cv2. VideoCapture(video_path)

# ret = True
# # read frames
# while ret:
#     ret, frame = cap.read()

#     # detect objects
#     # track objects
#     results = model. track(frame, persist=True)

#     # plot results
#     frame_ = results [0] . plot()

#     # visualize
#     cv2. imshow('frame', frame_)
    
#     if cv2.waitkey(25) & 0xFF == ord('q'):
#         break
    
    

import numpy as np
import matplotlib.pyplot as plt

# Function to generate points for a pentagonal space-filling curve
def generate_pentagonal_curve(iterations=3):
    # Initial coordinates for the pentagon
    vertices = np.array([
        [1.0, 0.0],
        [0.809017, 0.587785],
        [0.309017, 0.951057],
        [-0.309017, 0.951057],
        [-0.809017, 0.587785]
    ])
    
    # Initialize points with the initial pentagon vertices
    points = vertices
    
    # Iteratively generate the curve
    for _ in range(iterations):
        new_points = []
        for i in range(len(points) - 1):
            # Calculate midpoints between consecutive vertices
            mid_point = (points[i] + points[i+1]) / 2
            new_points.append(points[i])
            # Calculate the new vertex
            new_vertex = mid_point + 0.587785 * (points[i+1] - points[i])
            new_points.append(new_vertex)
        new_points.append(points[-1])
        points = np.array(new_points)
    
    return points

# Generate pentagonal curve points
curve_points = generate_pentagonal_curve(iterations=5)

# Plot the curve
plt.figure(figsize=(8, 8))
plt.plot(curve_points[:, 0], curve_points[:, 1], color='blue')
plt.axis('equal')
plt.axis('off')

# Save the plot as a PNG image
plt.savefig('pentagonal_curve.png', dpi=300, bbox_inches='tight', pad_inches=0)

# Show the plot (optional)
plt.show()

