import pandas as pd
import numpy as np
import math

# Load CSV data
file_path = "trajectory_coordinates.csv"
data = pd.read_csv(file_path)

# Remove last 15 rows
data_trimmed = data[:-150]

# Convert to numpy array for easier processing
coordinates = data_trimmed.to_numpy()

# Standardize X and Y values
x_0, y_0 = coordinates[0, 0], coordinates[0, 1]
standardized_coordinates = np.copy(coordinates)
standardized_coordinates[:, 0] = coordinates[:, 0] - x_0
standardized_coordinates[:, 1] = y_0 - coordinates[:, 1]

# Create DataFrame for standardized coordinates
standardized_df = pd.DataFrame(standardized_coordinates, columns=data.columns)

# Save to Excel
output_file = "standardized_coordinates.xlsx"
standardized_df.to_excel(output_file, index=False)

# Corrected print statement
print(f"Standardized coordinates saved to {output_file}")

# Conversion factor from pixels to cm
PIXEL_TO_CM = 45 / 90

FPS = 59.94

DT = 1 / FPS

def classify_trajectory(standardized_coordinates):
    # Initialize flags and variables
    x_negative = False
    x_cycles = 0
    increasing_y = True
    phase = "increasing"  # Track if x is increasing or decreasing

    # Check if x becomes negative soon after starting
    for i in range(1, len(standardized_coordinates)):
        x = standardized_coordinates[i, 0]
        y = standardized_coordinates[i, 1]

        if x < 0:
            if not np.any(standardized_coordinates[:i, 0] > 0):  # Ensure no positive X before
                print("Trajectory Type: Type 3\n")
                return "Type 3"
            x_negative = True

        # Check if y is still increasing or constant
        if i > 1 and y < standardized_coordinates[i - 1, 1]:
            increasing_y = False  # y started decreasing

        # Only track x cycles while y is increasing or constant
        if increasing_y:
            prev_x = standardized_coordinates[i - 1, 0]

            if phase == "increasing":
                if x < prev_x:  # x starts decreasing
                    phase = "decreasing"
            elif phase == "decreasing":
                if x > prev_x:  # x starts increasing again (one cycle complete)
                    x_cycles += 1
                    phase = "increasing"

    # Final classification based on checks
    if not x_negative:
        print("Trajectory Type: Type 2\n")
        return "Type 2"
    elif x_cycles >= 2:
        print("Trajectory Type: Type 4\n")
        return "Type 4"
    elif x_cycles == 1:
        print("Trajectory Type: Type 1\n")
        return "Type 1"
    else:
        print("No matching trajectory type detected.\n")
        return "-1"

# Classify trajectory
trajectory_type = "Type 2"
#classify_trajectory(standardized_coordinates)

# TYPE - 3
if trajectory_type == "Type 3":
    #TYPE - 3
    # Calculate required values
    # 1. Ymax
    y_max = np.max(standardized_coordinates[:, 1])  # Get the true max Y value
    y_max_index = np.where(standardized_coordinates[:, 1] == y_max)[0][0]  # Index of first occurrence of max Y
    x_corresponding = standardized_coordinates[y_max_index, 0] * PIXEL_TO_CM  # X at Ymax (converted to cm)
    y_max = (y_max * PIXEL_TO_CM) + 45  # Final Ymax in cm with offset
        
    # 2. Ycatch

    # Define the range to search for Ycatch
    search_range = standardized_df.iloc[y_max_index + 1 : y_max_index + 121]  # next 60 points after Ymax

    # Find the index of the minimum Y value in that range
    y_catch_index = search_range['Y'].idxmin()

    # Convert to cm and add bar height (45 cm from ground level)
    y_catch = (standardized_df.loc[y_catch_index, 'Y'] * PIXEL_TO_CM) + 45
    x_catch = standardized_df.loc[y_catch_index, 'X']


    # 3. Ydrop
    y_drop = y_max - y_catch if y_catch is not None else None

    # 4. Xnet
    x_net = x_catch

    # 5. X1
    x1, y1 = 1.0, 1.0
    index = -1
    for i in range(1, len(standardized_coordinates) - 1):
        if standardized_coordinates[i, 0] > 0 and standardized_coordinates[i, 0] >=  standardized_coordinates[i - 1, 0] and standardized_coordinates[i, 0] > standardized_coordinates[i + 1, 0]:
            x1 = standardized_coordinates[i, 0] * PIXEL_TO_CM
            y1 = standardized_coordinates[i, 1] * PIXEL_TO_CM
            index = i
            break

    # 6. Theta1 (in degrees)
    x_start = standardized_coordinates[0, 0] * PIXEL_TO_CM
    y_start = standardized_coordinates[0, 1] * PIXEL_TO_CM

    ymax_index = np.argmax(standardized_coordinates[:, 1])
    x_apex = standardized_coordinates[ymax_index, 0] * PIXEL_TO_CM
    y_apex = standardized_coordinates[ymax_index, 1] * PIXEL_TO_CM

    x1 = x_apex - x_start
    y1 = y_apex - y_start

    theta1 = math.degrees(math.atan2(x1, y1))

    # 7. X2
    temp = None
    for i in range(index, len(standardized_coordinates) - 1):
        if standardized_coordinates[i, 0] < standardized_coordinates[i + 1, 0]:
            temp = standardized_coordinates[i, 0] * PIXEL_TO_CM
            break
    x2 = x1 - temp

    # 8. Xloop
    x_loop = None
    if temp is not None and x_catch is not None:
        x_loop = x_catch - temp

    # 9. Velocity (using the first difference)
    velocities = []
    for i in range(1, len(standardized_coordinates)):
        delta_x = (standardized_coordinates[i, 0] - standardized_coordinates[i - 1, 0]) * PIXEL_TO_CM
        delta_y = (standardized_coordinates[i, 1] - standardized_coordinates[i - 1, 1]) * PIXEL_TO_CM
        velocity = math.sqrt(delta_x**2 + delta_y**2)
        velocities.append(velocity)
    velocities_cm_per_s = [v * FPS for v in velocities]
    avg_velocity_cm_per_s = np.mean(velocities_cm_per_s) if velocities_cm_per_s else None

    # 10. Acceleration (using the second difference)
    accelerations = []
    for i in range(1, len(velocities)):
        delta_v = velocities[i] - velocities[i - 1]
        accelerations.append(delta_v)
    accelerations_cm_per_s2 = [a * (FPS ** 2) for a in accelerations]
    avg_acceleration_cm_per_s2 = np.mean(accelerations_cm_per_s2) if accelerations_cm_per_s2 else None


# TYPE - 2
elif trajectory_type == "Type 2":
    #TYPE - 2 
    # Calculate required values
    # 1. Ymax
    y_max = np.max(standardized_coordinates[:, 1])  # Get the true max Y value
    y_max_index = np.where(standardized_coordinates[:, 1] == y_max)[0][0]  # Index of first occurrence of max Y
    x_corresponding = standardized_coordinates[y_max_index, 0] * PIXEL_TO_CM  # X at Ymax (converted to cm)
    y_max = (y_max * PIXEL_TO_CM) + 45  # Final Ymax in cm with offset
        
    # 2. Ycatch
     # Define the range to search for Ycatch
    search_range = standardized_df.iloc[y_max_index + 1 : y_max_index + 481]  # next 60 points after Ymax

    # Find the index of the minimum Y value in that range
    y_catch_index = search_range['Y'].idxmin()

    # Convert to cm and add bar height (45 cm from ground level)
    y_catch = (standardized_df.loc[y_catch_index, 'Y'] * PIXEL_TO_CM) + 45
    x_catch = standardized_df.loc[y_catch_index, 'X']


    # 3. Ydrop
    y_drop = y_max - y_catch if y_catch is not None else None

    # 4. Xnet
    x_net = x_catch

    # 5. X1
    x1, y1 = None, None
    index = -1
    for i in range(1, len(standardized_coordinates) - 1):
        if standardized_coordinates[i, 0] >=  standardized_coordinates[i - 1, 0] and standardized_coordinates[i, 0] > standardized_coordinates[i + 1, 0]:
            x1 = standardized_coordinates[i, 0] * PIXEL_TO_CM
            y1 = standardized_coordinates[i, 1] * PIXEL_TO_CM
            index = i
            break

    # 6. Theta1 (in degrees)
    x_start = standardized_coordinates[0, 0] * PIXEL_TO_CM
    y_start = standardized_coordinates[0, 1] * PIXEL_TO_CM

    ymax_index = np.argmax(standardized_coordinates[:, 1])
    x_apex = standardized_coordinates[ymax_index, 0] * PIXEL_TO_CM
    y_apex = standardized_coordinates[ymax_index, 1] * PIXEL_TO_CM

    x1 = x_apex - x_start
    y1 = y_apex - y_start

    theta1 = math.degrees(math.atan2(x1, y1))

    # 7. X2
    temp = None
    for i in range(index, len(standardized_coordinates) - 1):
        if standardized_coordinates[i, 0] < standardized_coordinates[i + 1, 0]:
            temp = standardized_coordinates[i, 0] * PIXEL_TO_CM
            break
    x2 = x1 - temp

    # 8. Xloop
    x_loop = None
    if temp is not None and x_catch is not None:
        x_loop = x_catch - temp

    # 9. Velocity (using the first difference)
    velocities = []
    for i in range(1, len(standardized_coordinates)):
        delta_x = (standardized_coordinates[i, 0] - standardized_coordinates[i - 1, 0]) * PIXEL_TO_CM
        delta_y = (standardized_coordinates[i, 1] - standardized_coordinates[i - 1, 1]) * PIXEL_TO_CM
        velocity = math.sqrt(delta_x**2 + delta_y**2)
        velocities.append(velocity)
    velocities_cm_per_s = [v * FPS for v in velocities]
    avg_velocity_cm_per_s = np.mean(velocities_cm_per_s) if velocities_cm_per_s else None

    # 10. Acceleration (using the second difference)
    accelerations = []
    for i in range(1, len(velocities)):
        delta_v = velocities[i] - velocities[i - 1]
        accelerations.append(delta_v)
    accelerations_cm_per_s2 = [a * (FPS ** 2) for a in accelerations]
    avg_acceleration_cm_per_s2 = np.mean(accelerations_cm_per_s2) if accelerations_cm_per_s2 else None

elif trajectory_type == "Type 1":
    #TYPE - 1 
    # Calculate required values
    # Calculate required values
    # 1. Ymax
    y_max = np.max(standardized_coordinates[:, 1])  # Get the true max Y value
    y_max_index = np.where(standardized_coordinates[:, 1] == y_max)[0][0]  # Index of first occurrence of max Y
    x_corresponding = standardized_coordinates[y_max_index, 0] * PIXEL_TO_CM  # X at Ymax (converted to cm)
    y_max = (y_max * PIXEL_TO_CM) + 45  # Final Ymax in cm with offset
        
    # 2. Ycatch
     # Define the range to search for Ycatch
    search_range = standardized_df.iloc[y_max_index + 1 : y_max_index + 121]  # next 60 points after Ymax

    # Find the index of the minimum Y value in that range
    y_catch_index = search_range['Y'].idxmin()

    # Convert to cm and add bar height (45 cm from ground level)
    y_catch = (standardized_df.loc[y_catch_index, 'Y'] * PIXEL_TO_CM) + 45
    x_catch = standardized_df.loc[y_catch_index, 'X']


    # 3. Ydrop
    y_drop = y_max - y_catch if y_catch is not None else None

    # 4. Xnet
    x_net = x_catch

    # 5. X1
    x1, y1 = None, None
    index = -1
    for i in range(1, len(standardized_coordinates) - 1):
        if standardized_coordinates[i, 0] >=  standardized_coordinates[i - 1, 0] and standardized_coordinates[i, 0] > standardized_coordinates[i + 1, 0]:
            x1 = standardized_coordinates[i, 0] * PIXEL_TO_CM
            y1 = standardized_coordinates[i, 1] * PIXEL_TO_CM
            index = i
            break

    # 6. Theta1 (in degrees)
    x_start = standardized_coordinates[0, 0] * PIXEL_TO_CM
    y_start = standardized_coordinates[0, 1] * PIXEL_TO_CM

    ymax_index = np.argmax(standardized_coordinates[:, 1])
    x_apex = standardized_coordinates[ymax_index, 0] * PIXEL_TO_CM
    y_apex = standardized_coordinates[ymax_index, 1] * PIXEL_TO_CM

    x1 = x_apex - x_start
    y1 = y_apex - y_start

    theta1 = math.degrees(math.atan2(x1, y1))

    # 7. X2
    temp = None
    for i in range(index, len(standardized_coordinates) - 1):
        if standardized_coordinates[i, 0] < 0 and standardized_coordinates[i, 0] < standardized_coordinates[i + 1, 0]:
            temp = standardized_coordinates[i, 0] * PIXEL_TO_CM
            break
    x2 = x1 - temp

    # 8. Xloop
    x_loop = None
    if temp is not None and x_catch is not None:
        x_loop = x_catch - temp

    # 9. Velocity (using the first difference)
    velocities = []
    for i in range(1, len(standardized_coordinates)):
        delta_x = (standardized_coordinates[i, 0] - standardized_coordinates[i - 1, 0]) * PIXEL_TO_CM
        delta_y = (standardized_coordinates[i, 1] - standardized_coordinates[i - 1, 1]) * PIXEL_TO_CM
        velocity = math.sqrt(delta_x**2 + delta_y**2)
        velocities.append(velocity)
    velocities_cm_per_s = [v * FPS for v in velocities]
    avg_velocity_cm_per_s = np.mean(velocities_cm_per_s) if velocities_cm_per_s else None

    # 10. Acceleration (using the second difference)
    accelerations = []
    for i in range(1, len(velocities)):
        delta_v = velocities[i] - velocities[i - 1]
        accelerations.append(delta_v)
    accelerations_cm_per_s2 = [a * (FPS ** 2) for a in accelerations]
    avg_acceleration_cm_per_s2 = np.mean(accelerations_cm_per_s2) if accelerations_cm_per_s2 else None

elif trajectory_type == "Type 4":
    # TYPE - 4 
    # Calculate required values
    # Calculate required values
    # 1. Ymax
    y_max = np.max(standardized_coordinates[:, 1])  # Get the true max Y value
    y_max_index = np.where(standardized_coordinates[:, 1] == y_max)[0][0]  # Index of first occurrence of max Y
    x_corresponding = standardized_coordinates[y_max_index, 0] * PIXEL_TO_CM  # X at Ymax (converted to cm)
    y_max = (y_max * PIXEL_TO_CM) + 45  # Final Ymax in cm with offset
        
    # 2. Ycatch
     # Define the range to search for Ycatch
    search_range = standardized_df.iloc[y_max_index + 1 : y_max_index + 81]  # next 60 points after Ymax

    # Find the index of the minimum Y value in that range
    y_catch_index = search_range['Y'].idxmin()

    # Convert to cm and add bar height (45 cm from ground level)
    y_catch = (standardized_df.loc[y_catch_index, 'Y'] * PIXEL_TO_CM) + 45
    x_catch = standardized_df.loc[y_catch_index, 'X']

    # 3. Ydrop
    y_drop = y_max - y_catch if y_catch is not None else None

    # 4. Xnet
    x_net = x_catch

    # 5. X1
    x1, y1 = 1.0, 1.0
    index = -1
    index2 = -1
    max_x = float('-inf')

    # Start from the first coordinate after (0, 0)
    for i in range(1, len(standardized_coordinates)):
        current_x = standardized_coordinates[i, 0]
        current_y = standardized_coordinates[i, 1]

        # Stop when the second (0, 0) occurs
        if current_x == 0 and current_y == 0:
            index2 = current_x
            break

        # Update max_x and corresponding y1 if a larger x-coordinate is found
        if current_x > max_x:
            max_x = current_x
            x1 = current_x * PIXEL_TO_CM
            y1 = current_y * PIXEL_TO_CM
            index = i

    # 6. Theta1 (in degrees)
    x_start = standardized_coordinates[0, 0] * PIXEL_TO_CM
    y_start = standardized_coordinates[0, 1] * PIXEL_TO_CM

    ymax_index = np.argmax(standardized_coordinates[:, 1])
    x_apex = standardized_coordinates[ymax_index, 0] * PIXEL_TO_CM
    y_apex = standardized_coordinates[ymax_index, 1] * PIXEL_TO_CM

    x1 = x_apex - x_start
    y1 = y_apex - y_start

    theta1 = math.degrees(math.atan2(x1, y1))

    # 7. X2
    temp = None
    for i in range(index2+1, len(standardized_coordinates) - 1):
        if standardized_coordinates[i, 0] < 0 and standardized_coordinates[i, 0] < standardized_coordinates[i + 1, 0]:
            temp = standardized_coordinates[i, 0] * PIXEL_TO_CM
            break
    x2 = x1 - temp

    # 8. Xloop
    x_loop = None
    if temp is not None and x_catch is not None:
        x_loop = x_catch - temp

    # 9. Velocity (using the first difference)
    velocities = []
    for i in range(1, len(standardized_coordinates)):
        delta_x = (standardized_coordinates[i, 0] - standardized_coordinates[i - 1, 0]) * PIXEL_TO_CM
        delta_y = (standardized_coordinates[i, 1] - standardized_coordinates[i - 1, 1]) * PIXEL_TO_CM
        velocity = math.sqrt(delta_x**2 + delta_y**2)
        velocities.append(velocity)
    velocities_cm_per_s = [v * FPS for v in velocities]
    avg_velocity_cm_per_s = np.mean(velocities_cm_per_s) if velocities_cm_per_s else None

    # 10. Acceleration (using the second difference)
    accelerations = []
    for i in range(1, len(velocities)):
        delta_v = velocities[i] - velocities[i - 1]
        accelerations.append(delta_v)
    accelerations_cm_per_s2 = [a * (FPS ** 2) for a in accelerations]
    avg_acceleration_cm_per_s2 = np.mean(accelerations_cm_per_s2) if accelerations_cm_per_s2 else None

if(trajectory_type != "-1"):
    # Display results
    results = {
        'Ymax (cm)': y_max,
        'Ycatch (cm)': y_catch,
        'Ydrop (cm)': y_drop,
        'Xnet (cm)': x_net,
        'X1 (cm)': x1,
        'Theta1 (degrees)': theta1,
        'X2 (cm)': x2,
        'Xloop (cm)': x_loop,
        'Average Velocity (cm/s)': avg_velocity_cm_per_s,
        'Average Acceleration (cm/s^2)': avg_acceleration_cm_per_s2
    }

    for key, value in results.items():
        print(f"{key}: {value}")