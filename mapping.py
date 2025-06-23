import cv2
import numpy as np
import os

# Function to draw a pixel grid on the frame
def draw_pixel_grid(frame, grid_size=20, color=(0, 255, 0), thickness=1):
    h, w, _ = frame.shape

    # Draw vertical lines
    for x in range(0, w, grid_size):
        cv2.line(frame, (x, 0), (x, h), color, thickness)

    # Draw horizontal lines
    for y in range(0, h, grid_size):
        cv2.line(frame, (0, y), (w, y), color, thickness)

    return frame

# Video file path
video_path = "barbell-path-tracker-master\\barbell-path-tracker-master\\Video\\GX010689_male.mp4"

# Open the video
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Output directory for frames
output_dir = "frames_with_grid"
os.makedirs(output_dir, exist_ok=True)

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame if needed
    frame = cv2.resize(frame, (1920, 1080))

    # Draw the pixel grid
    frame_with_grid = draw_pixel_grid(frame, grid_size=20)

    # Save every frame
    output_path = os.path.join(output_dir, f"frame_{frame_count:05d}.png")
    cv2.imwrite(output_path, frame_with_grid)
    print(f"Saved: {output_path}")

    frame_count += 1

cap.release()
print("All frames processed and saved.")
