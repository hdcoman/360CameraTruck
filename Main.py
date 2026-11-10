# 360CameraTruck
# 
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

# File path setup
image_dir = r"C:\Users\Harri\OneDrive\Documents\Senior\Senior Design\360CameraTruck\captured_frames"
image_path = os.path.join(image_dir, "camera_0.png")

# Grid configuration
GRID_SIZE = (17, 10)
SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Load image
img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError(f"Could not load image at path: {image_path}")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

height, width, channels = img.shape
print(f"Image Size Received: {width} x {height} pixels ({channels} channels)")

# Find chessboard corners
found, corners = cv2.findChessboardCorners(gray, GRID_SIZE)

print(corners.shape)
print(corners[:10])

# AI code to increase accuracy
if found:
    # 5. Refine corners to sub-pixel accuracy
    corners_subpix = cv2.cornerSubPix(
        gray, corners, (11, 11), (-1, -1), SUBPIX_CRITERIA
    )

    # ---------------------------------------------------------
    # MATRIX EXTRACTION
    # ---------------------------------------------------------
    # Reshape from (170, 1, 2) to (170, 2) [N_corners x (X, Y)]
    corner_matrix = corners_subpix.reshape(-1, 2)

    # Option A: Grid Matrix layout (10 rows, 17 cols, each containing (x,y))
    grid_matrix = corners_subpix.reshape(GRID_SIZE[1], GRID_SIZE[0], 2)

    print(f"Detected {len(corner_matrix)} total corner points.\n")
    
    print("First 5 Corner Coordinates [X, Y]:")
    print(corner_matrix[:5])
    print("\nShape of flat corner list:", corner_matrix.shape)
    print("Shape of structured grid matrix:", grid_matrix.shape)

    # Save array to a CSV file for downstream calibration math
    csv_path = os.path.join(image_dir, "camera_0_corners.csv")
    np.savetxt(csv_path, corner_matrix, delimiter=",", header="x_pixel,y_pixel", comments="")
    print(f"\nSaved corner coordinates matrix to: {csv_path}")

    # Draw and show corners
    display_img = img.copy()
    cv2.drawChessboardCorners(display_img, GRID_SIZE, corners_subpix, found)
    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB))
    plt.title("Detected Corners")
    plt.axis("off")
    plt.show()

else:
    print("Could not detect chessboard corners.")
# Begin process of finding calibration matrix
