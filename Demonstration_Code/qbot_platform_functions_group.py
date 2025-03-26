import numpy as np
import cv2
import csv
import os
import time
import math

from pal.products.qbot_platform import QBotPlatformDriver
from pal.utilities.math import Calculus
from scipy.ndimage import median_filter
from pal.utilities.math import Calculus
from pal.utilities.stream import BasicStream
from quanser.common import Timeout
from concurrent.futures import ThreadPoolExecutor

class QBPMovement():
    """ This class contains the functions for the QBot Platform such as
    Forward/Inverse Differential Drive Kinematics etc. """

    def __init__(self):
        self.WHEEL_RADIUS = QBotPlatformDriver.WHEEL_RADIUS      # radius of the wheel (meters)
        self.WHEEL_BASE = QBotPlatformDriver.WHEEL_BASE          # distance between wheel contact points on the ground (meters)
        self.WHEEL_WIDTH = QBotPlatformDriver.WHEEL_WIDTH        # thickness of the wheel (meters)
        self.ENCODER_COUNTS = QBotPlatformDriver.ENCODER_COUNTS  # encoder counts per channel
        self.ENCODER_MODE = QBotPlatformDriver.ENCODER_MODE      # multiplier for a quadrature encoder

    def diff_drive_inverse_velocity_kinematics(self, forSpd, turnSpd):
        """This function is for the differential drive inverse velocity
        kinematics for the QBot Platform. It converts provided body speeds
        (forward speed in m/s and turn speed in rad/s) into corresponding
        wheel speeds (rad/s)."""

        #------------Replace the following lines with your code---------------#
        wL = 0
        wR = 0
        #---------------------------------------------------------------------#
        return wL, wR

    def diff_drive_forward_velocity_kinematics(self, wL, wR):
        """This function is for the differential drive forward velocity
        kinematics for the QBot Platform. It converts provided wheel speeds
        (rad/s) into corresponding body speeds (forward speed in m/s and
        turn speed in rad/s)."""
        #------------Replace the following lines with your code---------------#
        forSpd = 0
        turnSpd = 0
        #---------------------------------------------------------------------#
        return forSpd, turnSpd

class QBPVision():
    def __init__(self):
        self.imageCriteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.integrator = None
        self.derivative = None
        self.prev_error = 0  # For derivative term

    def undistort_img(self,distImgs,cameraMatrix,distCoefficients):
        """
        This function undistorts a general camera, given the camera matrix and
        coefficients.
        """

        undist = cv2.undistort(distImgs,
                               cameraMatrix,
                               distCoefficients,
                               None,
                               cameraMatrix)
        return undist

    def df_camera_undistort(self, image):
        """
        This function undistorts the downward camera using the camera
        intrinsics and coefficients."""
        CSICamIntrinsics = np.array([[419.36179672, 0, 292.01381114],
                                     [0, 420.30767196, 201.61650657],
                                     [0, 0, 1]])
        CSIDistParam = np.array([-7.42983302e-01,
                                 9.24162996e-01,
                                 -2.39593372e-04,
                                 1.66230745e-02,
                                 -5.27787439e-01])
        undistortedImage = self.undistort_img(
                                                image,
                                                CSICamIntrinsics,
                                                CSIDistParam
                                                )
        return undistortedImage

    def subselect_and_threshold(self, image, rowStart, rowEnd, minThreshold, maxThreshold):
        """
        This function subselects a horizontal slice of the input image from
        rowStart to rowEnd for all columns, and then thresholds it based on the
        provided min and max thresholds. Returns the binary output from
        thresholding."""

        #------------Replace the following lines with your code---------------#
        
        subImage = None
        binary = None
        
        # subImage = image[rowStart:rowEnd,:]
        # # _, binary = cv2.threshold(subImage, minThreshold, maxThreshold, cv2.THRESH_BINARY)
        # # # Apply Gaussian blur to reduce noise
        # # Convert to grayscale (ensure it's already grayscale)
        # gray = cv2.cvtColor(subImage, cv2.COLOR_BGR2GRAY) if len(subImage.shape) == 3 else subImage

        # # Apply Gaussian blur to reduce noise
        # blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # # Apply binary thresholding
        # _, binary = cv2.threshold(blurred, minThreshold, maxThreshold, cv2.THRESH_BINARY)

        
        # #---------------------------------------------------------------------#

        # return binary
        # Ensure input is a NumPy array
        image = np.asarray(image, dtype=np.uint8)

        # Select only the specified rows
        subImage = image[rowStart:rowEnd, :]

        # Apply binary thresholding
        _, binary = cv2.threshold(subImage, minThreshold, maxThreshold, cv2.THRESH_BINARY)

        # Ensure output is also a NumPy array
        return np.asarray(binary, dtype=np.uint8)
    

    def collect_and_label_data(self, binary_image, count, storage_path, data_path):
        """
        Classifies the binary image as Line Detected (1) or No Line Detected (0).
        Ensures input is a NumPy array and correctly computes the white pixel ratio.
        """
        image_list = []
        image_list.append(binary_image)
        
        if isinstance(binary_image, list):
            if len(binary_image) == 0:
                raise ValueError("binary_image list is empty.")
            binary_image = binary_image[0]  # Extract the first image if it's a list

        if isinstance(binary_image, QBPVision):
            binary_image = np.asarray(binary_image.imageData, dtype=np.uint8)
        
        binary_image = cv2.GaussianBlur(binary_image, (5, 5), 0)  # Kernel size (5,5), standard deviation 0

        # Convert to grayscale if the image has 3 channels (e.g., RGB)
        if len(binary_image.shape) == 3:
            binary_image = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)
        
        binary_image = np.where(binary_image < 225, 0, 1)
         
        # self.storage = r"Line_detection"    
        # self.data_store = r"Data_storage.csv" 
        self.storage = storage_path
        self.data_store = data_path  
        if not os.path.exists(self.storage):
            os.makedirs(self.storage) 
        self.csv_file = os.path.join(self.storage, self.data_store)   
        # if not os.path.exists(self.csv_file):
        # Refresh data everytime for a new run
        if count == 0:
            with open(self.csv_file, "w", newline="") as file:
                csv.writer(file)
                    # writer.writerow([self.image_store, labels])
                # writer.writerow(["Training data stored"])  # Column headers  # Column headers
        
         # Convert the NumPy array to a string representation
        # binary_image_str = np.array2string(binary_image, separator=',')  # Store as a string
        # binary_image_str = np.array2string(binary_image, separator=',', threshold=np.inf, linewidth=np.inf)

        
        # # Append each image path and label to CSV
        # with open(self.csv_file, "a", newline="") as file:
        #     writer = csv.writer(file) 
        #     writer.writerow([binary_image_str])
                # Append the NumPy matrix to the CSV file (without truncation)
        with open(self.csv_file, "a", newline="") as file:
            np.savetxt(file, binary_image, fmt="%d", delimiter=",")
            file.write("\n")  # Add an empty line for separation
        
        return binary_image        
                        
        # # Compute total number of pixels
        # total_pixels = binary_image.size  # Ensure it's an integer scalar

        # # Compute the number of white pixels
        # white_pixels = np.count_nonzero(binary_image)

        # # Compute white pixel ratio as a single float value
        # white_ratio = float(white_pixels) / float(total_pixels)

        # # Return 1 if line detected, else 0
        # label = 1 if white_ratio > float(threshold) else 0
        # # # Append each image path and label to CSV
        # # for i in range(training_data):
        # #     with open(self.csv_file, "a", newline="") as file:
        # #         writer = csv.writer(file)
        # #         writer.writerow(["Line Detection", label])
            
        # return label
        
    def save_images_from_csv(self, storage, data, out_folder, labels, training, img_height, img_width):
        out_dir = storage + out_folder
        csv_data_file = os.path.join(storage, data)  # Create path joining
        csv_training_file = os.path.join(storage, training)
        
        for label in labels:
            label_path = os.path.join(storage, label)
            if not os.path.exists(label_path):
                os.makedirs(label_path, exist_ok=True)
                
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        directory = os.path.dirname(csv_training_file)
        if not os.path.exists(directory):
            print(f"Directory {directory} does not exist! Creating it.")
            os.makedirs(directory, exist_ok=True)
        else:
            print(f"Directory {directory} exists.")

        # Create the CSV file if it doesn't exist (mode='w' will create it)
        try:
            with open(csv_training_file, mode='w', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                # Write the header if the file is being created
                csv_writer.writerow(["Image Path", "label", "Error"])
                print(f"Created or opened {csv_training_file} for writing.")
        except PermissionError as e:
            print(f"PermissionError: {e}")
            return
      
        # Load the CSV file
        with open(csv_data_file, "r") as file:
            lines = file.readlines()
        
        images = []
        current_image = []

        for line in lines:
            line = line.strip()
            if not line:  # Skip empty lines
                continue

            # Convert the row into a list of integers
            row_data = [int(val) if val.strip().isdigit() else 0 for val in line.split(",")]

            if len(row_data) == img_width:
                current_image.append(row_data)

            # If we have collected exactly 320 rows, store the image and reset
            if len(current_image) == img_height:
                images.append(np.array(current_image, dtype=int))
                current_image = []  # Reset for the next image

        # Save each extracted image
        for idx, binary_image in enumerate(images):
                # Analyze the image for line position and junction detection
            center_row = binary_image[img_height // 2, :]  # Middle row
            white_cols = np.where(center_row == 1)[0]  # Columns with white pixels
            line_width = white_cols.max() - white_cols.min() if len(white_cols) > 0 else 0
            col = int(np.mean(white_cols)) if len(white_cols) > 0 else None    
            
            Center = 160
            Straight = 20
            Slight_Margin = 40
            Heavy_Margin = 80
            Junction_Left = Center - Heavy_Margin
            Junction_Right = Center + Heavy_Margin
            
           # Check for T-junction or crossroad
            is_junction = False
            branch_left = False
            branch_right = False
            vertical_line = False
            vertical_line_count = 0
            min_consecutive_rows = 1  # Adjust this value

            for row in range(img_height):
                row_data = binary_image[row, :]
                window_start = col - Straight if col is not None else Junction_Right
                window_end = col + Straight if col is not None else Junction_Left
                if col is None:
                    if np.sum(row_data[Junction_Right:Junction_Left]) > 3:
                        vertical_line_count += 1
                else:
                    if np.sum(row_data[window_start:window_end]) > 2:
                        vertical_line_count += 1
                if vertical_line_count >= min_consecutive_rows:
                    vertical_line = True
                    break
                elif np.sum(row_data[window_start:window_end]) == 0 and col is not None:
                    vertical_line_count = 0

            # Check for branches in the upper half
            total_left_pixels = 0
            total_right_pixels = 0
            for row in range(img_height):
                row_data = binary_image[row, :]
                left_half = row_data[:Junction_Left]  # Left side up to column 80
                right_half = row_data[Junction_Right:]  # Right side from column 240
                total_left_pixels += np.sum(left_half == 1)  # Count white pixels (1s)
                total_right_pixels += np.sum(right_half == 1)

            # Determine junction type based on pixel counts
            if vertical_line:
                if total_left_pixels >= 50:
                    branch_left = True
                if total_right_pixels >= 50:
                    branch_right = True
                if branch_left and branch_right:
                    is_junction = True
                    label = "Crossroad"
                elif branch_left:
                    is_junction = True
                    label = "Left_Junction"
                elif branch_right:
                    is_junction = True
                    label = "Right_Junction"
           
            

            if col is None:
                error = None
                label = "Off_Track"
  
            else:
                error = col - Center  # Calculate the error from the center column
                # print(col) # col is the center of CAMERA
                if -Straight <= error <= Straight:  # Straight Line
                    label = "On_Track"
                elif -Slight_Margin < error < -Straight:  # Slight Left Moving
                    label = "Slight_Left"
                elif Slight_Margin > error > Straight:  # Slight Right Moving
                    label = "Slight_Right"
                elif -Heavy_Margin < error < -Slight_Margin:  # Heavy Left Moving
                    label = "Heavy_Left"
                elif Heavy_Margin > error > Slight_Margin:   # Heavy Right Moving
                    label = "Heavy_Right"
    
            
            
            if binary_image.shape == (img_height, img_width):  # Ensure correct size
                 # Convert 0/1 binary data to 8-bit grayscale (0 → black, 1 → white)
                binary_image = (binary_image * 255).astype(np.uint8)
                # Add the entry to the appropriate label folder
                label_path = os.path.join(storage, label)
                img_path = os.path.join(out_dir, f"Image_{idx+1}_{label}.png")
            
                # Save the image
                cv2.imwrite(img_path, binary_image)
                print(f"Saved: {img_path} with label: {label}")
                label_img_path = os.path.join(label_path, f"image_{idx+1}.png")
                cv2.imwrite(label_img_path, binary_image)
            else:
                print(f"Skipping invalid shape {binary_image.shape} at index {idx+1}")
                
            
            
            # Write image path and label to CSV
            with open(csv_training_file, mode='a', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([img_path, label, error])
        
        # LABELS = ["straight", "left", "right", "off_line"]
        #             for label in LABELS:  
        #                 os.makedirs(os.path.join(storage_path, label), exist_ok=True)        
                
    # def save_data(self, images, labels):
        
        
    #     self.image_store = r"Line_detection"
    #     # self.data_store = r"Data_storage.csv"
    #     if not os.path.exists(self.image_store):
    #         os.makedirs(self.image_store)
        
        
    #     self.csv_file = os.path.join(self.image_store, self.data_store)
    #     # # Keep only the last `training_data` images
    #     # images = images[-training_data:]  
    #     # labels = labels[-training_data:]
        
    #     # # Refresh data everytime for a new run
    #     # if count == 0:
    #     #     with open(self.csv_file, "w", newline="") as file:
    #     #         writer = csv.writer(file)
    #     #             # writer.writerow([self.image_store, labels])
    #     #         writer.writerow(["  Image path stored", "Detection"])  # Column headers
                
        
    #     # image_filename = []
    #     # image_filenames = [
    #     #     os.path.join(self.image_store, f"image_{i:04d}.png") for i in range(len(images))]
        
    #     # # ✅ **MULTI-THREADED IMAGE SAVING** 🚀
    #     # with ThreadPoolExecutor() as executor:
    #     #     executor.map(QBPVision.save_image, image_filenames, images)
        
    #     # Append each image path and label to CSV
    #     with open(self.csv_file, "a", newline="") as file:
    #         writer = csv.writer(file)
                 
    #      # Save images and append data to CSV
    #         # for i, image in enumerate(images):
    #         for i, image in enumerate(images):    
    #             # image_filename = os.path.join(self.image_store, f"image_{i:04d}.png") 
    #             # image_filename = os.path.join(self.image_store, f"Line_{int(time.time())}.png")  
    #             image_filename = os.path.join(self.image_store, f"Line_{count:04d}.png")  
    #             cv2.imwrite(image_filename, image)  # Save image
                
    #             # # Append each image path and label to CSV
    #             # with open(self.csv_file, "a", newline="") as file:
    #             #     writer = csv.writer(file)
    #             writer.writerow([image_filename, labels[i]])
                
                
    
    # def __init__(self):
    # # def deviate(self):    
    #     self.deviation_list = []
    #     self.frame_count = 0            
    # def measure_deviation(self, binary_image):
    #     """
    #     Measures deviation from the center of the detected line.
    #     """
    #     width = 320
    #     image_center = width // 2
    #     tolerance = 16

    #     # Ensure binary image is in 0 (black) and 255 (white)
    #     # binary_image = (binary_image * 255).astype(np.uint8) if binary_image.max() == 1 else binary_image

        
    #     # Define the tolerance range
    #     lower_bound = image_center - tolerance  # 160 - 35 = 125
    #     upper_bound = image_center + tolerance  # 160 + 35 = 195

    #     # Find white pixels (where the line is)
    #     white_pixels = np.where(binary_image == 255)

    #     if len(white_pixels[1]) == 0:
    #         deviation = None  # No line detected, ignore this frame
    #     else:
    #         line_center = int(np.mean(white_pixels[1]))  # Find center of white pixels
    #         # deviation = line_center - image_center  # Calculate deviation from center
            
    #             # Compute deviation only if outside tolerance
    #         if lower_bound <= line_center <= upper_bound:
    #             deviation = 0  # Inside tolerance = perfectly following the line
    #         else:
    #             deviation = min(abs(line_center - lower_bound), abs(line_center - upper_bound))
        

    #     self.deviation_list.append(deviation)
    #     self.frame_count += 1  

    # def calculate_accuracy(self):
    #     """
    #     Calculates the accuracy as a percentage of frames where the robot stays within the allowed range.
    #     """
    #     total_frames = len(self.deviation_list)
    #     correct_frames = sum(1 for deviation in self.deviation_list if deviation == 0)
        
    #     accuracy = (correct_frames / total_frames) * 100 if total_frames > 0 else 0
    #     print(f"Accuracy: {accuracy:.2f}%")
    #     return accuracy

    # def plot_deviation(self):
    #     """
    #     Plots deviation over time with 160 as the center.
    #     """
    #     plt.figure(figsize=(10, 5))

    #     # Convert None values to NaN for plotting gaps where no line was detected
    #     deviation_array = np.array(self.deviation_list, dtype=float)
    #     deviation_array[np.isnan(deviation_array)] = np.nan  # Convert None to NaN

    #     # Adjust deviation so that 160 is the center
    #     adjusted_deviation = 160 - deviation_array  

    #     plt.plot(range(self.frame_count), adjusted_deviation, label="Deviation", color="blue")
    #     plt.axhline(160, color="r", linestyle="--", label="Center Line")  # Centered at 160
    #     plt.xlabel("Frame")
    #     plt.ylabel("Deviation from Center (pixels)")
    #     plt.title("Line Following Deviation Over Time")
    #     plt.legend()
    #     plt.savefig("Deviation.png")
    #     plt.show()
        
    # def measure_deviation(self, binary_image):
    #     """
    #     Measures deviation from the center of the detected line.
    #     """
    #     # height, width = binary_image.shape
    #     # image_center = width // 2
        
    #     width=320
    #     image_center = width//2

    #     # Find white pixels (where the line is)
    #     white_pixels = np.where(binary_image == 255)
    #     if len(white_pixels[1]) == 0:
    #         deviation = width  # No line detected (max deviation)
    #     else:
    #         line_center = int(np.mean(white_pixels[1]))
    #         deviation = line_center - image_center

    #     self.deviation_list.append(deviation)
    #     self.frame_count += 1  

    # def plot_deviation(self):
    #     """
    #     Plots deviation without blocking the main loop.
    #     """
    #     plt.figure(figsize=(10, 5))
    #     plt.plot(range(self.frame_count), self.deviation_list, label="Deviation")
    #     plt.axhline(160, color="r", linestyle="--", label="Center Line")
    #     plt.xlabel("Frame")
    #     plt.ylabel("Deviation from Center (pixels)")
    #     plt.title("Line Following Deviation Over Time")
    #     plt.legend()
    #     plt.savefig("Deviation")
    #     plt.show()  


    def image_find_objects(self, image, connectivity, minArea, maxArea):
        """
        This function implements connected component labeling on the provided
        image with the desired connectivity. From the list of blobs detected,
        it returns the first blob that fits the desired area criteria based
        on minArea and maxArea provided. Returns the column and row location
        of the blob centroid, as well as the blob's area. """

        col = 0
        row = 0
        area = 0

        #------------Replace the following lines with your code---------------#

        # (labels, ids, values, centroids) = None
        num_labels, labels, values, centroids = cv2.connectedComponentsWithStats(image, connectivity)

        #---------------------------------------------------------------------#

        #-------------Uncomment the following 12 lines of code----------------#

        for idx, val in enumerate(values):
            if val[4]>minArea and val[4] < maxArea:
                value = val
                centroid = centroids[idx]
                col = centroid[0]
                row = centroid[1]
                area = value[4]
                break
            else:
                col = None
                row = None
                area = None
        #---------------------------------------------------------------------#

        return col, row, area

    # def line_to_speed_map(self, sampleRate, saturation):

    #     integrator   = Calculus().integrator(dt = sampleRate, saturation=saturation)
    #     derivative   = Calculus().differentiator(dt = sampleRate)
    #     next(integrator)
    #     next(derivative)
    #     forSpd, turnSpd = 0, 0
    #     offset = 0

    #     img_width = 320  # Image width in pixels
    #     img_bottom_y = 200  # Bottom of the image (assuming height is 200px)
    #     focal_length = 55      # Estimated focal length for perspective (adjust if needed)

    #     while True:
    #         col, kP, kD = yield forSpd, turnSpd

    #         if col is not None:
    #             #-----------Complete the following lines of code--------------#
    #             # error = 0
    #             error = ((img_width/2) - col)
    #             # angle = np.arctan2(0, 0)
                
    #              # Compute the error angle using arctan2
    #             angle = np.arctan2(error, (focal_length))  

    #             # turnSpd = 0 * angle + 0 * derivative.send(angle)
                
    #             turnSpd = kP * angle + kD * derivative.send(angle)
    #             # forSpd = 0
    #             # forSpd = kP * np.cos(angle)
    #             # Adjust forward speed based on error (angle)
    #             forSpd = max(min(1.0, np.cos(angle) * 0.3), 0.1)  # Forward speed based on alignment

    #         else:
    #             forSpd = 0
    #             turnSpd = 0

    #             #-------------------------------------------------------------#
    #             offset = integrator.send(25*turnSpd)
    #             error += offset
    
    #Not using PID
    # def line_to_speed_map(self, prediction, sampleRate, saturation, lineFollow = False, lineLeft=False, lineRight=False):
    #     img_width = 320  # Image width in pixels
    #     img_bottom_y = 200  # Bottom of the image (assuming height is 200px)
    #     focal_length = 55  # Estimated focal length for perspective
    #     slow_spd = 0.01
    #     deg_turn = 10
    #     while True:
    #         forSpd, turnSpd = 0, 0

    #         if prediction is not None:
    #             # Map CNN predictions to speeds
    #             # print(prediction)
    #             if lineLeft:
    #                 if prediction == 0:  # Crossroad
    #                     forSpd = slow_spd   # Slow forward speed
    #                     turnSpd = deg_turn
    #                     # for i in range (15):
    #                     #     turnSpd = (-15+i)  # Sharp left turn
    #                     # print(forSpd)
    #                     # print(turnSpd)
    #                     print("Pressed Left Turn at Crossroad")
    #                 elif prediction == 3:  # Left_Junction
    #                     forSpd = slow_spd   # Slow forward speed
    #                     # for i in range (15):
    #                     #     turnSpd = (-15+i)  # Sharp left turn
    #                     turnSpd = deg_turn  # Sharp left turn
    #                     print("Left Turn at Left Junction")
    #                 elif prediction == 5:  # On_Track
    #                     forSpd = 0.1   # Steady forward speed
    #                     turnSpd = 0.0   # No turn
    #                 elif prediction == 1:  # Heavy_Left
    #                     forSpd = 0.02   # Forward
    #                     # for i in range (15):
    #                     #     turnSpd = (-15+i)  # Sharp left turn
    #                     turnSpd = 0.1   # Strong left turn
    #                 elif prediction == 7:  # Slight_Left
    #                     forSpd = 0.02   # Forward
    #                     turnSpd = 0.05  # Gentle left turn
    #                 elif prediction == 2:  # Heavy_Right
    #                     forSpd = 0.02   # Forward (ignore right turn signal)
    #                     turnSpd = 0.0   # No turn, keep moving forward
    #                 elif prediction == 8:  # Slight_Right
    #                     forSpd = 0.02   # Forward (ignore right turn signal)
    #                     turnSpd = 0.0   # No turn, keep moving forward
    #                 elif prediction == 6:  # Right_Junction
    #                     forSpd = 0.02   # Forward (ignore junction for now)
    #                     turnSpd = 0.0   # No turn unless specified
    #                 elif prediction == 4:  # Off_Track
    #                     forSpd = 0.0    # Stop
    #                     turnSpd = 0.0   # No turn
    #             elif lineRight:  # Right Arrow pressed
    #                 if prediction == 0:  # Crossroad
    #                     forSpd = slow_spd   # Slow forward speed
    #                     turnSpd = -deg_turn   # Sharp right turn
    #                     print("Right Turn at Crossroad")
    #                 elif prediction == 6:  # Right_Junction
    #                     forSpd = slow_spd   # Slow forward speed
    #                     turnSpd = -deg_turn   # Sharp right turn
    #                     print("Right Turn at Right Junction")
    #                 elif prediction == 5:  # On_Track
    #                     forSpd = 0.2   # Steady forward speed
    #                     turnSpd = 0.0   # No turn
    #                 elif prediction == 2:  # Heavy_Right
    #                     forSpd = 0.02   # Forward
    #                     turnSpd = -0.1  # Strong right turn
    #                 elif prediction == 8:  # Slight_Right
    #                     forSpd = 0.02   # Forward
    #                     turnSpd = -0.05 # Gentle right turn
    #                 elif prediction == 1:  # Heavy_Left
    #                     forSpd = 0.02   # Forward (ignore left turn signal)
    #                     turnSpd = 0.0   # No turn, keep moving forward
    #                 elif prediction == 7:  # Slight_Left
    #                     forSpd = 0.02   # Forward (ignore left turn signal)
    #                     turnSpd = 0.0   # No turn, keep moving forward
    #                 elif prediction == 3:  # Left_Junction
    #                     forSpd = 0.02   # Forward (ignore junction for now)
    #                     turnSpd = 0.0   # No turn unless specified
    #                 elif prediction == 4:  # Off_Track
    #                     forSpd = 0.0    # Stop
    #                     turnSpd = 0.0   # No turn
                
    #             elif lineFollow:  # Foward key pressed
    #                 # Default autonomous behavior
    #                 if prediction == 5:  # On_Track
    #                     forSpd = 0.1   # Move forward slowly
    #                     turnSpd = 0.0
    #                 elif prediction == 1:  # Heavy_Left
    #                     forSpd = 0.1
    #                     turnSpd = 0.1
    #                 elif prediction == 2:  # Heavy_Right
    #                     forSpd = 0.1
    #                     turnSpd = -0.1
    #                 elif prediction == 7:  # Slight_Left
    #                     forSpd = 0.1
    #                     turnSpd = 0.05
    #                 elif prediction == 8:  # Slight_Right
    #                     forSpd = 0.1
    #                     turnSpd = -0.05
    #                 elif prediction == 0:  # Crossroad
    #                     forSpd = slow_spd    # Stop by default
    #                     turnSpd = 0.0
    #                 elif prediction == 3:  # Left_Junction
    #                     forSpd = slow_spd   # Slow forward
    #                     turnSpd = 0.0   # No turn unless key pressed
    #                 elif prediction == 6:  # Right_Junction
    #                     forSpd = slow_spd   # Slow forward
    #                     turnSpd = 0.0   # No turn unless key pressed
    #                 elif prediction == 4:  # Off_Track
    #                     forSpd = 0.0    # Stop
    #                     turnSpd = 0.0        
                        
    #         yield forSpd, turnSpd  # Return speeds
                        
    # def line_to_speed_map(self, prediction, sampleRate, saturation, lineFollow=False, lineLeft=False, lineRight=False):
    #     integrator   = Calculus().integrator(dt = sampleRate, saturation=saturation)
    #     derivative   = Calculus().differentiator(dt = sampleRate)
    #     next(integrator)
    #     next(derivative)
    #     offset = 0
    #     kP = 0.1  # Proportional gain (tune as needed)
    #     kI = 0.01  # Integral gain (tune as needed)
    #     kD = 0.05  # Derivative gain (tune as needed)
        
    #     img_width = 320  # Image width in pixels
    #     img_bottom_y = 200  # Bottom of the image (assuming height is 200px)
    #     focal_length = 50  # Estimated focal length for perspective
    #     slow_spd = 0.01
    #     deg_turn = 1.5  # Angular speed in rad/s for turning (adjust as needed)

    #     # State variables for 90-degree turn
    #     turning = False  # Flag to indicate if a turn is in progress
    #     turn_direction = 0  # 1 for left, -1 for right, 0 for no turn
    #     turn_frames = 0  # Counter for frames spent turning
    #     turn_duration = int(1.57 / (deg_turn * sampleRate))  # Frames for 90 degrees (π/2 rad / (turnSpd * sampleRate))

    #     while True:
    #         forSpd, turnSpd = 0, 0

    #         # If currently turning, complete the turn before processing new predictions
    #         if turning:
    #             forSpd = slow_spd  # Slow forward speed during turn
    #             turnSpd = deg_turn * turn_direction  # Continue turning in the specified direction
    #             turn_frames += 1

    #             # Check if the turn is complete
    #             if turn_frames >= turn_duration:
    #                 turning = False  # Turn completed
    #                 turn_frames = 0  # Reset counter
    #                 turn_direction = 0  # Reset direction
    #                 print(f"Completed 90-degree {'left' if turn_direction > 0 else 'right'} turn")
    #             yield forSpd, turnSpd  # Yield turning speeds
    #             continue  # Skip to next iteration until turn is complete

    #         # Only process new predictions if not turning
    #         if prediction is not None:
    #             if lineLeft:
    #                 if prediction == 0:  # Crossroad
    #                     turning = True
    #                     turn_direction = 1  # Left turn
    #                     turn_frames = 0
    #                     forSpd = slow_spd
    #                     turnSpd = deg_turn * turn_direction
    #                     print("Initiating 90-degree left turn at Crossroad")
    #                 elif prediction == 3:  # Left_Junction
    #                     turning = True
    #                     turn_direction = 1  # Left turn
    #                     turn_frames = 0
    #                     forSpd = slow_spd
    #                     turnSpd = deg_turn * turn_direction
    #                     print("Initiating 90-degree left turn at Left Junction")
    #                 elif prediction == 5:  # On_Track
    #                     forSpd = 0.1   # Steady forward speed
    #                     turnSpd = 0.0  # No turn
    #                 elif prediction == 1:  # Heavy_Left
    #                     forSpd = 0.08
    #                     turnSpd = 0.1  # Strong left turn
    #                 elif prediction == 7:  # Slight_Left
    #                     forSpd = 0.08
    #                     turnSpd = 0.05  # Gentle left turn
    #                 elif prediction == 2:  # Heavy_Right
    #                     forSpd = 0.08
    #                     turnSpd = -0.1
    #                 elif prediction == 8:  # Slight_Right
    #                     forSpd = 0.08
    #                     turnSpd = -0.05
    #                 elif prediction == 6:  # Right_Junction
    #                     forSpd = 0.08
    #                     turnSpd = 0.0   # No turn unless specified
    #                 elif prediction == 4:  # Off_Track
    #                     forSpd = 0.05
    #                     turnSpd = 0.0   # Stop

    #             elif lineRight:  # Right Arrow pressed
    #                 if prediction == 0:  # Crossroad
    #                     turning = True
    #                     turn_direction = -1  # Right turn
    #                     turn_frames = 0
    #                     forSpd = slow_spd
    #                     turnSpd = deg_turn * turn_direction
    #                     print("Initiating 90-degree right turn at Crossroad")
    #                 elif prediction == 6:  # Right_Junction
    #                     turning = True
    #                     turn_direction = -1  # Right turn
    #                     turn_frames = 0
    #                     forSpd = slow_spd
    #                     turnSpd = deg_turn * turn_direction
    #                     print("Initiating 90-degree right turn at Right Junction")
    #                 elif prediction == 5:  # On_Track
    #                     forSpd = 0.1
    #                     turnSpd = 0.0   # No turn
    #                 elif prediction == 2:  # Heavy_Right
    #                     forSpd = 0.08
    #                     turnSpd = -0.1  # Strong right turn
    #                 elif prediction == 8:  # Slight_Right
    #                     forSpd = 0.08
    #                     turnSpd = -0.05  # Gentle right turn
    #                 elif prediction == 1:  # Heavy_Left
    #                     forSpd = 0.08
    #                     turnSpd = 0.1
    #                 elif prediction == 7:  # Slight_Left
    #                     forSpd = 0.08
    #                     turnSpd = 0.05
    #                 elif prediction == 3:  # Left_Junction
    #                     forSpd = 0.08
    #                     turnSpd = 0.0   # No turn unless specified
    #                 elif prediction == 4:  # Off_Track
    #                     forSpd = 0.05
    #                     turnSpd = 0.0   # Stop
    #             elif lineFollow:
    #                 # Map prediction to a column position (pseudo-col)
    #                 col = {
    #                     5: img_width / 2,      # On_Track (center)
    #                     1: img_width / 4,      # Heavy_Left (left side)
    #                     2: 3 * img_width / 4,  # Heavy_Right (right side)
    #                     7: 3 * img_width / 8,  # Slight_Left
    #                     8: 5 * img_width / 8,  # Slight_Right
    #                     0: img_width / 2,      # Crossroad (center)
    #                     3: img_width / 4,      # Left_Junction (left side for curve)
    #                     6: 3 * img_width / 4,  # Right_Junction (right side for curve)
    #                     4: img_width / 2       # Off_Track (center)
    #                 }.get(prediction, img_width / 2)

    #                 # Calculate error from center of image
    #                 error = (img_width / 2) - col
                    
    #                 # Amplify error for Heavy_Left (1) and Heavy_Right (2) to increase turning
    #                 if prediction in [1, 2]:  # Heavy_Left or Heavy_Right
    #                     error *= 1.5  # Increase error by 50% (adjust as needed)
                        
    #                 # Compute angle using arctan2 (angle of deviation)
    #                 angle = np.arctan2(error, focal_length)

    #                 # PID control for turning speed
    #                 proportional = kP * angle
    #                 integral = kI * integrator.send(angle)  # Cumulative error for sustained turns
    #                 derivative_term = kD * derivative.send(angle)
    #                 turnSpd = proportional + integral + derivative_term
                    
    #                 if prediction in [1, 2]:  # Heavy_Left or Heavy_Right
    #                     turnSpd *= 1.5  # Boost turnSpd by 50%

    #                 # Clamp turnSpd to prevent excessive turning
    #                 turnSpd = max(min(turnSpd, 0.3), -0.3)  # Increased limit for sharper curves

    #                 # Forward speed with a base value and slight reduction for curves
    #                 base_speed = 0.15  # Consistent base speed
    #                 speed_reduction = 0.05 * abs(angle)  # Reduce speed based on turn angle
    #                 forSpd = max(base_speed - speed_reduction, 0.05)  # Minimum speed 0.05
    #             #     if prediction == 5:  # On_Track
    #             #         forSpd = 0.2
    #             #         turnSpd = 0.0
    #             #     elif prediction == 1:  # Heavy_Left
    #             #         forSpd = 0.1
    #             #         turnSpd = 0.1
    #             #     elif prediction == 2:  # Heavy_Right
    #             #         forSpd = 0.1
    #             #         turnSpd = -0.1
    #             #     elif prediction == 7:  # Slight_Left
    #             #         forSpd = 0.1
    #             #         turnSpd = 0.05
    #             #     elif prediction == 8:  # Slight_Right
    #             #         forSpd = 0.1
    #             #         turnSpd = -0.05
    #             #     elif prediction == 0:  # Crossroad
    #             #         forSpd = slow_spd  # Slow forward by default
    #             #         turnSpd = 0.0      # No turn unless key pressed
    #             #     elif prediction == 3:  # Left_Junction
    #             #         forSpd = slow_spd
    #             #         turnSpd = 0.0      # No turn unless key pressed
    #             #     elif prediction == 6:  # Right_Junction
    #             #         forSpd = slow_spd
    #             #         turnSpd = 0.0      # No turn unless key pressed
    #             #     elif prediction == 4:  # Off_Track
    #             #         forSpd = 0.0
    #             #         turnSpd = 0.0

    #         yield forSpd, turnSpd  # Return speeds               
                            
    # def line_to_speed_map(self, sampleRate, saturation, lineFollow=False, lineLeft=False, lineRight=False):
    #     img_width = 320
    #     img_bottom_y = 200
    #     focal_length = 55
    #     slow_spd = 0.01
    #     deg_turn = 1.5

    #     integrator = Calculus().integrator(dt=sampleRate, saturation=saturation)
    #     derivative = Calculus().differentiator(dt=sampleRate)
    #     next(integrator)
    #     next(derivative)
    #     kP = 0.02
    #     kD = 0.1

    #     turning = False
    #     turn_direction = 0
    #     turn_frames = 0
    #     turn_duration = int(1.57 / (deg_turn * sampleRate))

    #     while True:
    #         forSpd, turnSpd = 0, 0

    #         if turning:
    #             forSpd = slow_spd
    #             turnSpd = deg_turn * turn_direction
    #             turn_frames += 1
    #             if turn_frames >= turn_duration:
    #                 turning = False
    #                 turn_frames = 0
    #                 turn_direction = 0
    #                 print(f"Completed 90-degree {'left' if turn_direction > 0 else 'right'} turn")
    #             yield forSpd, turnSpd
    #             continue

    #         data = yield  # Expecting a tuple (prediction, col)
    #         prediction, col = data if data is not None else (None, None)

    #         if prediction is not None:
    #             if lineLeft:
    #                 if prediction == 0:  # Crossroad
    #                     turning = True
    #                     turn_direction = 1
    #                     turn_frames = 0
    #                     forSpd = slow_spd
    #                     turnSpd = deg_turn * turn_direction
    #                     print("Initiating 90-degree left turn at Crossroad")
    #                 elif prediction == 3:  # Left_Junction
    #                     turning = True
    #                     turn_direction = 1
    #                     turn_frames = 0
    #                     forSpd = slow_spd
    #                     turnSpd = deg_turn * turn_direction
    #                     print("Initiating 90-degree left turn at Left Junction")
    #                 elif prediction == 5:  # On_Track
    #                     forSpd = 0.1
    #                     turnSpd = 0.0
    #                 elif prediction == 1 and col is not None:  # Heavy_Left
    #                     error = (img_width / 2) - col
    #                     angle = np.arctan2(error, focal_length)
    #                     forSpd = max(min(1.0, np.cos(angle) * 0.3), 0.1)
    #                     turnSpd = kP * angle + kD * derivative.send(angle)
    #                 elif prediction == 7 and col is not None:  # Slight_Left
    #                     error = (img_width / 2) - col
    #                     angle = np.arctan2(error, focal_length)
    #                     forSpd = max(min(1.0, np.cos(angle) * 0.3), 0.1)
    #                     turnSpd = kP * angle + kD * derivative.send(angle)
    #                 elif prediction == 2:  # Heavy_Right
    #                     forSpd = 0.02
    #                     turnSpd = 0.0
    #                 elif prediction == 8:  # Slight_Right
    #                     forSpd = 0.02
    #                     turnSpd = 0.0
    #                 elif prediction == 6:  # Right_Junction
    #                     forSpd = 0.02
    #                     turnSpd = 0.0
    #                 elif prediction == 4:  # Off_Track
    #                     forSpd = 0.0
    #                     turnSpd = 0.0

    #             elif lineRight:
    #                 if prediction == 0:  # Crossroad
    #                     turning = True
    #                     turn_direction = -1
    #                     turn_frames = 0
    #                     forSpd = slow_spd
    #                     turnSpd = deg_turn * turn_direction
    #                     print("Initiating 90-degree right turn at Crossroad")
    #                 elif prediction == 6:  # Right_Junction
    #                     turning = True
    #                     turn_direction = -1
    #                     turn_frames = 0
    #                     forSpd = slow_spd
    #                     turnSpd = deg_turn * turn_direction
    #                     print("Initiating 90-degree right turn at Right Junction")
    #                 elif prediction == 5:  # On_Track
    #                     forSpd = 0.2
    #                     turnSpd = 0.0
    #                 elif prediction == 2 and col is not None:  # Heavy_Right
    #                     error = (img_width / 2) - col
    #                     angle = np.arctan2(error, focal_length)
    #                     forSpd = max(min(1.0, np.cos(angle) * 0.3), 0.1)
    #                     turnSpd = kP * angle + kD * derivative.send(angle)
    #                 elif prediction == 8 and col is not None:  # Slight_Right
    #                     error = (img_width / 2) - col
    #                     angle = np.arctan2(error, focal_length)
    #                     forSpd = max(min(1.0, np.cos(angle) * 0.3), 0.1)
    #                     turnSpd = kP * angle + kD * derivative.send(angle)
    #                 elif prediction == 1:  # Heavy_Left
    #                     forSpd = 0.02
    #                     turnSpd = 0.0
    #                 elif prediction == 7:  # Slight_Left
    #                     forSpd = 0.02
    #                     turnSpd = 0.0
    #                 elif prediction == 3:  # Left_Junction
    #                     forSpd = 0.02
    #                     turnSpd = 0.0
    #                 elif prediction == 4:  # Off_Track
    #                     forSpd = 0.0
    #                     turnSpd = 0.0

    #             elif lineFollow:
    #                 if prediction == 5:  # On_Track
    #                     forSpd = 0.1
    #                     turnSpd = 0.0
    #                 elif prediction == 1 and col is not None:  # Heavy_Left
    #                     error = (img_width / 2) - col
    #                     angle = np.arctan2(error, focal_length)
    #                     forSpd = max(min(1.0, np.cos(angle) * 0.3), 0.1)
    #                     turnSpd = kP * angle + kD * derivative.send(angle)
    #                 elif prediction == 2 and col is not None:  # Heavy_Right
    #                     error = (img_width / 2) - col
    #                     angle = np.arctan2(error, focal_length)
    #                     forSpd = max(min(1.0, np.cos(angle) * 0.3), 0.1)
    #                     turnSpd = kP * angle + kD * derivative.send(angle)
    #                 elif prediction == 7 and col is not None:  # Slight_Left
    #                     error = (img_width / 2) - col
    #                     angle = np.arctan2(error, focal_length)
    #                     forSpd = max(min(1.0, np.cos(angle) * 0.3), 0.1)
    #                     turnSpd = kP * angle + kD * derivative.send(angle)
    #                 elif prediction == 8 and col is not None:  # Slight_Right
    #                     error = (img_width / 2) - col
    #                     angle = np.arctan2(error, focal_length)
    #                     forSpd = max(min(1.0, np.cos(angle) * 0.3), 0.1)
    #                     turnSpd = kP * angle + kD * derivative.send(angle)
    #                 elif prediction == 0:  # Crossroad
    #                     forSpd = slow_spd
    #                     turnSpd = 0.0
    #                 elif prediction == 3:  # Left_Junction
    #                     forSpd = slow_spd
    #                     turnSpd = 0.0
    #                 elif prediction == 6:  # Right_Junction
    #                     forSpd = slow_spd
    #                     turnSpd = 0.0
    #                 elif prediction == 4:  # Off_Track
    #                     forSpd = 0.0
    #                     turnSpd = 0.0

    #             if col is None and not turning:
    #                 forSpd = 0.0
    #                 turnSpd = 0.0

    #         yield forSpd, turnSpd               
                            
                # if prediction == 0:  # Crossroad
                #     forSpd = 0.0  # Slow down by default
                #     turnSpd = 0.0  # No turn by default
                #     if lineLeft:  # Left Arrow pressed
                #         forSpd = 0.05
                #         turnSpd = 1.5  # Turn left
                #         print("Left Pressed")
                #     elif lineRight:  # Right Arrow pressed
                #         forSpd = 0.05
                #         turnSpd = -1.5   # Turn right
                #         print("Right Pressed")
                        
                # elif prediction == 1:  # Heavy_Left
                #     if lineFollow:
                #         forSpd = 0.0   # Forward
                #         turnSpd = 0.1  # Strong left turn
                    
                # elif prediction == 2:  # Heavy_Right
                #     if lineFollow:
                #         forSpd = 0.0   # Forward
                #         turnSpd = -0.1 # Strong right turn
                    
                # elif prediction == 3:  # Left_Junction
                #     if lineFollow:
                #         forSpd = 0.01  # Slightly slower by default
                #     # turnSpd = 0.15 # Moderate left turn by default
                #     elif lineLeft:  # Left Arrow pressed
                #         forSpd = 0.05
                #         turnSpd = 1.5  # Turn left
                #         print("Left Pressed")
                #     elif lineRight:  # Right Arrow pressed
                #         forSpd = 0.05
                #         turnSpd = -1.5   # Turn right
                #         print("Right Pressed")
                        
                # elif prediction == 4:  # Off_Track
                #     forSpd = 0.0   # Stop
                #     turnSpd = 0.0  # No turn
                    
                # elif prediction == 5:  # On_Track
                #     if lineFollow:
                #         forSpd = 0.05   # Full speed ahead
                #         turnSpd = 0.0  # No turn
                    
                # elif prediction == 6:  # Right_Junction
                #     if lineFollow:
                #         forSpd = 0.01  # Slightly slower by default
                #     # turnSpd = -0.15 # Moderate right turn by default
                #     elif lineLeft:  # Left Arrow pressed
                #         forSpd = 0.05
                #         turnSpd = 1.5  # Turn left
                #         print("Left Pressed")
                #     elif lineRight:  # Right Arrow pressed
                #         forSpd = 0.05
                #         turnSpd = -1.5   # Turn right
                #         print("Right Pressed")
                        
                # elif prediction == 7:  # Slight_Left
                #     if lineFollow:
                #         forSpd = 0.03   # Forward
                #         turnSpd = 0.05 # Gentle left turn
                    
                # elif prediction == 8:  # Slight_Right
                #     if lineFollow:
                #         forSpd = 0.03   # Forward
                #         turnSpd = -0.05 # Gentle right turn
                
    # def line_to_speed_map(self, prediction, sampleRate, saturation, lineFollow=False, lineLeft=False, lineRight=False):
    #     integrator = Calculus().integrator(dt=sampleRate, saturation=saturation)
    #     derivative = Calculus().differentiator(dt=sampleRate)
    #     next(integrator)
    #     next(derivative)
    #     offset = 0
    #     kP_base = 0.15
    #     kI_base = 0.02
    #     kD_base = 0.08
        
    #     img_width = 320
    #     img_bottom_y = 200
    #     focal_length = 50
    #     slow_spd = 0.01
    #     speed_up = 0.2

    #     # Set turn duration to 60 frames (1 second at 60 FPS)
    #     if not hasattr(self, 'turn_duration'):
    #         self.turn_duration = 60
    #         print(f"Sample Rate: {sampleRate}, Turn Duration: {self.turn_duration} frames")
    #     if not hasattr(self, 'deg_turn'):
    #         self.deg_turn = 1.57 / (self.turn_duration * sampleRate)
    #         print(f"deg_turn: {self.deg_turn}")

    #     # Initialize state variables
    #     if not hasattr(self, 'turning'):
    #         self.turning = False
    #     if not hasattr(self, 'turn_direction'):
    #         self.turn_direction = 0
    #     if not hasattr(self, 'turn_frames'):
    #         self.turn_frames = 0
    #     if not hasattr(self, 'crossroad_count'):
    #         self.crossroad_count = 0
    #     if not hasattr(self, 'left_junction_count'):
    #         self.left_junction_count = 0
    #     if not hasattr(self, 'right_junction_count'):
    #         self.right_junction_count = 0
    #     if not hasattr(self, 'last_prediction'):
    #         self.last_prediction = None
    #     if not hasattr(self, 'error_history'):
    #         self.error_history = []
    #     if not hasattr(self, 'error_history_length'):
    #         self.error_history_length = 5
    #     if not hasattr(self, 'speeding_up'):
    #         self.speeding_up = False
    #     if not hasattr(self, 'speed_up_frames'):
    #         self.speed_up_frames = 0
    #     if not hasattr(self, 'speed_up_duration'):
    #         self.speed_up_duration = 30  # 0.5 seconds at 60 FPS

    #     while True:
    #         forSpd, turnSpd = 0, 0

    #         # Handle ongoing turn
    #         if self.turning:
    #             forSpd = 0.0
    #             turnSpd = self.deg_turn * self.turn_direction
    #             self.turn_frames += 1
    #             print(f"Turning: Frame {self.turn_frames}/{self.turn_duration}, forSpd={forSpd}, turnSpd={turnSpd}")
    #             if self.turn_frames >= self.turn_duration:
    #                 self.turning = False
    #                 self.turn_frames = 0
    #                 self.turn_direction = 0
    #                 self.speeding_up = True  # Start speed-up phase
    #                 self.speed_up_frames = 0
    #                 print(f"Completed 90-degree {'left' if self.turn_direction > 0 else 'right'} turn, speeding up")
    #             yield forSpd, turnSpd
    #             continue

    #         if prediction is not None:
    #             if lineLeft:
    #                 if prediction == 0:
    #                     self.turning = True
    #                     self.turn_direction = 1
    #                     self.turn_frames = 0
    #                     forSpd = 0.0
    #                     turnSpd = self.deg_turn * self.turn_direction
    #                     print("Initiating 90-degree left turn at Crossroad")
    #                 elif prediction == 1:
    #                     self.turning = True
    #                     self.turn_direction = 1
    #                     self.turn_frames = 0
    #                     forSpd = 0.0
    #                     turnSpd = self.deg_turn * self.turn_direction
    #                     print("Initiating 90-degree left turn at Left Junction")
    #                 elif prediction == 3:
    #                     forSpd = 0.1
    #                     turnSpd = 0.0
    #                 elif prediction == 5:
    #                     forSpd = 0.08
    #                     turnSpd = 0.05
    #                 elif prediction == 6:
    #                     forSpd = 0.08
    #                     turnSpd = -0.05
    #                 elif prediction == 4:
    #                     forSpd = 0.08
    #                     turnSpd = 0.0
    #                 elif prediction == 2:
    #                     forSpd = 0.05
    #                     turnSpd = 0.0

    #             elif lineRight:
    #                 if prediction == 0:
    #                     self.turning = True
    #                     self.turn_direction = -1
    #                     self.turn_frames = 0
    #                     forSpd = 0.0
    #                     turnSpd = self.deg_turn * self.turn_direction
    #                     print("Initiating 90-degree right turn at Crossroad")
    #                 elif prediction == 4:
    #                     self.turning = True
    #                     self.turn_direction = -1
    #                     self.turn_frames = 0
    #                     forSpd = 0.0
    #                     turnSpd = self.deg_turn * self.turn_direction
    #                     print("Initiating 90-degree right turn at Right Junction")
    #                 elif prediction == 3:
    #                     forSpd = 0.1
    #                     turnSpd = 0.0
    #                 elif prediction == 6:
    #                     forSpd = 0.08
    #                     turnSpd = -0.05
    #                 elif prediction == 5:
    #                     forSpd = 0.08
    #                     turnSpd = 0.05
    #                 elif prediction == 1:
    #                     forSpd = 0.08
    #                     turnSpd = 0.0
    #                 elif prediction == 2:
    #                     forSpd = 0.05
    #                     turnSpd = 0.0

    #             elif lineFollow:
    #                 # Map prediction to a column position, assuming sharper curves for Slight_Left/Right
    #                 col = {
    #                     3: img_width / 2,      # On_Track (center)
    #                     5: 1.5 * img_width / 4,  # Slight_Left 
    #                     6: 5 * img_width / 8,  # Slight_Right
    #                     0: img_width / 2,
    #                     1: 3 * img_width / 8,
    #                     4: 5 * img_width / 8,
    #                     2: img_width / 2
    #                 }.get(prediction, img_width / 2)

    #                 # Update counters for Crossroad, Left_Junction, and Right_Junction
    #                 if prediction == 0:  # Crossroad
    #                     self.crossroad_count += 1
    #                     self.left_junction_count = 0
    #                     self.right_junction_count = 0
    #                 elif prediction == 1:  # Left_Junction
    #                     self.left_junction_count += 1
    #                     self.crossroad_count = 0
    #                     self.right_junction_count = 0
    #                 elif prediction == 4:  # Right_Junction
    #                     self.right_junction_count += 1
    #                     self.crossroad_count = 0
    #                     self.left_junction_count = 0
    #                 else:
    #                     self.crossroad_count = 0
    #                     self.left_junction_count = 0
    #                     self.right_junction_count = 0

    #                 # Check for Crossroad (random turn)
    #                 if self.crossroad_count > 10 and not self.turning:
    #                     import random
    #                     self.turn_direction = random.choice([1, -1])
    #                     self.turning = True
    #                     self.turn_frames = 0
    #                     forSpd = 0.0
    #                     turnSpd = self.deg_turn * self.turn_direction
    #                     print(f"Crossroad detected {self.crossroad_count} times, initiating 90-degree {'left' if self.turn_direction > 0 else 'right'} turn")
    #                     self.crossroad_count = 0
    #                     yield forSpd, turnSpd
    #                     continue

    #                 # Check for Left_Junction (always turn left)
    #                 if self.left_junction_count > 10 and not self.turning:
    #                     self.turn_direction = 1  # Left turn
    #                     self.turning = True
    #                     self.turn_frames = 0
    #                     forSpd = 0.0
    #                     turnSpd = self.deg_turn * self.turn_direction
    #                     print(f"Left_Junction detected {self.left_junction_count} times, initiating 90-degree left turn")
    #                     self.left_junction_count = 0
    #                     yield forSpd, turnSpd
    #                     continue

    #                 # Check for Right_Junction (always turn right)
    #                 if self.right_junction_count > 10 and not self.turning:
    #                     self.turn_direction = -1  # Right turn
    #                     self.turning = True
    #                     self.turn_frames = 0
    #                     forSpd = 0.0
    #                     turnSpd = self.deg_turn * self.turn_direction
    #                     print(f"Right_Junction detected {self.right_junction_count} times, initiating 90-degree right turn")
    #                     self.right_junction_count = 0
    #                     yield forSpd, turnSpd
    #                     continue

    #                 # Skip PID control if turning
    #                 if self.turning:
    #                     continue

    #                 # Handle speed-up phase after turn
    #                 if self.speeding_up:
    #                     forSpd = speed_up
    #                     turnSpd = 0  # No turning during speed-up
    #                     self.speed_up_frames += 1
    #                     print(f"Speeding up: Frame {self.speed_up_frames}/{self.speed_up_duration}, forSpd={forSpd}, turnSpd={turnSpd}")
    #                     if self.speed_up_frames >= self.speed_up_duration:
    #                         self.speeding_up = False
    #                         self.speed_up_frames = 0
    #                         print("Speed-up phase completed, resuming normal speed")
    #                     yield forSpd, turnSpd
    #                     continue

    #                 # Calculate error from center of image
    #                 error = (img_width / 2) - col

    #                 # Smooth the error using a moving average
    #                 self.error_history.append(error)
    #                 if len(self.error_history) > self.error_history_length:
    #                     self.error_history.pop(0)
    #                 smoothed_error = sum(self.error_history) / len(self.error_history)

    #                 # Compute angle using arctan2 with smoothed error
    #                 angle = np.arctan2(smoothed_error, focal_length)

    #                 # Adjust PID gains dynamically based on error magnitude
    #                 error_magnitude = abs(smoothed_error) / (img_width / 2)
    #                 kP = kP_base * (1 + 2 * error_magnitude)
    #                 kI = kI_base * (1 + error_magnitude)
    #                 kD = kD_base * (1 + 2 * error_magnitude)

    #                 # PID control for turning speed
    #                 proportional = kP * angle
    #                 integral = kI * integrator.send(angle)
    #                 derivative_term = kD * derivative.send(angle)
    #                 turnSpd = proportional + integral + derivative_term

    #                 # More aggressive turn amplification for sharp curves
    #                 turn_amplification = 1 + 10 * (error_magnitude ** 3)
    #                 turnSpd *= turn_amplification

    #                 # Allow higher turn speeds for sharp curves
    #                 max_turn_spd = 0.5 + 0.7 * error_magnitude
    #                 turnSpd = max(min(turnSpd, max_turn_spd), -max_turn_spd)

    #                 # Forward speed with more aggressive reduction for sharp curves
    #                 base_speed = 0.15
    #                 speed_reduction = 0.1 * abs(angle)
    #                 forSpd = max(base_speed - speed_reduction, 0.03)

    #                 # Debug output to monitor behavior
    #                 print(f"Prediction: {prediction}, Error: {error:.2f}, Smoothed Error: {smoothed_error:.2f}, turnSpd: {turnSpd:.2f}, forSpd: {forSpd:.2f}")

    #         yield forSpd, turnSpd

    # def line_to_speed_map(self, prediction, sampleRate, saturation, lineFollow=False, lineLeft=False, lineRight=False):
    #     integrator = Calculus().integrator(dt=sampleRate, saturation=saturation)
    #     derivative = Calculus().differentiator(dt=sampleRate)
    #     next(integrator)
    #     next(derivative)
    #     offset = 0
    #     kP_base = 0.10  # Reduced from 0.15 for less aggressive response
    #     kI_base = 0.0   # Still disabled to prevent windup
    #     kD_base = 0.08  # Increased from 0.05 for more damping
        
    #     img_width = 320
    #     img_bottom_y = 200
    #     focal_length = 50
    #     slow_spd = 0.01
    #     speed_up = 0.2
    #     random_turning_C = 10
    #     random_turning_J = 18

    #     # Set turn duration to 60 frames (1 second at 60 FPS)
    #     if not hasattr(self, 'turn_duration'):
    #         self.turn_duration = 60
    #         print(f"Sample Rate: {sampleRate}, Turn Duration: {self.turn_duration} frames")
    #     if not hasattr(self, 'deg_turn'):
    #         self.deg_turn = 1.57 / (self.turn_duration * sampleRate)
    #         print(f"deg_turn: {self.deg_turn}")

    #     # Initialize state variables
    #     if not hasattr(self, 'turning'):
    #         self.turning = False
    #     if not hasattr(self, 'turn_direction'):
    #         self.turn_direction = 0
    #     if not hasattr(self, 'turn_frames'):
    #         self.turn_frames = 0
    #     if not hasattr(self, 'crossroad_count'):
    #         self.crossroad_count = 0
    #     if not hasattr(self, 'left_junction_count'):
    #         self.left_junction_count = 0
    #     if not hasattr(self, 'right_junction_count'):
    #         self.right_junction_count = 0
    #     if not hasattr(self, 'last_prediction'):
    #         self.last_prediction = None
    #     if not hasattr(self, 'error_history'):
    #         self.error_history = []
    #     if not hasattr(self, 'error_history_length'):
    #         self.error_history_length = 5  # Increased from 3 for smoother error
    #     if not hasattr(self, 'speeding_up'):
    #         self.speeding_up = False
    #     if not hasattr(self, 'speed_up_frames'):
    #         self.speed_up_frames = 0
    #     if not hasattr(self, 'speed_up_duration'):
    #         self.speed_up_duration = 30
    #     if not hasattr(self, 'slight_left_count'):
    #         self.slight_left_count = 0
    #     if not hasattr(self, 'slight_right_count'):
    #         self.slight_right_count = 0
    #     if not hasattr(self, 'integral_cap'):
    #         self.integral_cap = 0.5

    #     while True:
    #         forSpd, turnSpd = 0, 0

    #         # Handle ongoing turn
    #         if self.turning:
    #             forSpd = 0.0
    #             turnSpd = self.deg_turn * self.turn_direction
    #             self.turn_frames += 1
    #             print(f"Turning: Frame {self.turn_frames}/{self.turn_duration}, forSpd={forSpd}, turnSpd={turnSpd}")
    #             if self.turn_frames >= self.turn_duration:
    #                 self.turning = False
    #                 self.turn_frames = 0
    #                 self.turn_direction = 0
    #                 self.speeding_up = True
    #                 self.speed_up_frames = 0
    #                 print(f"Completed 90-degree {'left' if self.turn_direction > 0 else 'right'} turn, speeding up")
    #             yield forSpd, turnSpd
    #             continue
            
            
            
            

    #         if prediction is not None:
    #             if lineLeft:
    #                 if prediction == 0:
    #                     self.turning = True
    #                     self.turn_direction = 1
    #                     self.turn_frames = 0
    #                     forSpd = 0.0
    #                     turnSpd = self.deg_turn * self.turn_direction
    #                     print("Initiating 90-degree left turn at Crossroad")
    #                 elif prediction == 1:
    #                     self.turning = True
    #                     self.turn_direction = 1
    #                     self.turn_frames = 0
    #                     forSpd = 0.0
    #                     turnSpd = self.deg_turn * self.turn_direction
    #                     print("Initiating 90-degree left turn at Left Junction")
    #                 elif prediction == 3:
    #                     forSpd = 0.1
    #                     turnSpd = 0.0
    #                 elif prediction == 5:
    #                     forSpd = 0.08
    #                     turnSpd = 0.05
    #                 elif prediction == 6:
    #                     forSpd = 0.08
    #                     turnSpd = -0.05
    #                 elif prediction == 4:
    #                     forSpd = 0.08
    #                     turnSpd = 0.0
    #                 elif prediction == 2:
    #                     forSpd = 0.05
    #                     turnSpd = 0.0

    #             elif lineRight:
    #                 if prediction == 0:
    #                     self.turning = True
    #                     self.turn_direction = -1
    #                     self.turn_frames = 0
    #                     forSpd = 0.0
    #                     turnSpd = self.deg_turn * self.turn_direction
    #                     print("Initiating 90-degree right turn at Crossroad")
    #                 elif prediction == 4:
    #                     self.turning = True
    #                     self.turn_direction = -1
    #                     self.turn_frames = 0
    #                     forSpd = 0.0
    #                     turnSpd = self.deg_turn * self.turn_direction
    #                     print("Initiating 90-degree right turn at Right Junction")
    #                 elif prediction == 3:
    #                     forSpd = 0.1
    #                     turnSpd = 0.0
    #                 elif prediction == 6:
    #                     forSpd = 0.08
    #                     turnSpd = -0.05
    #                 elif prediction == 5:
    #                     forSpd = 0.08
    #                     turnSpd = 0.05
    #                 elif prediction == 1:
    #                     forSpd = 0.08
    #                     turnSpd = 0.0
    #                 elif prediction == 2:
    #                     forSpd = 0.05
    #                     turnSpd = 0.0

    #             elif lineFollow:
                    
    #                 # Update counters for Slight_Left and Slight_Right to estimate curve sharpness
    #                 if prediction == 5:  # Slight_Left
    #                     self.slight_left_count += 1
    #                     self.slight_right_count = 0
    #                 elif prediction == 6:  # Slight_Right
    #                     self.slight_right_count += 1
    #                     self.slight_left_count = 0
    #                 else:
    #                     self.slight_left_count = max(0, self.slight_left_count - 1)
    #                     self.slight_right_count = max(0, self.slight_right_count - 1)
                        
    #                     # Update counters for Crossroad, Left_Junction, and Right_Junction
    #                 if prediction == 0:  # Crossroad
    #                     self.crossroad_count += 1
    #                     self.left_junction_count = 0
    #                     self.right_junction_count = 0
    #                 elif prediction == 1:  # Left_Junction
    #                     self.left_junction_count += 1
    #                     self.crossroad_count = 0
    #                     self.right_junction_count = 0
    #                 elif prediction == 4:  # Right_Junction
    #                     self.right_junction_count += 1
    #                     self.crossroad_count = 0
    #                     self.left_junction_count = 0
    #                 else:
    #                     self.crossroad_count = 0
    #                     self.left_junction_count = 0
    #                     self.right_junction_count = 0

    #                 # Check for Crossroad (random turn) - Prioritize this over junctions
    #                 if self.crossroad_count > random_turning_C and not self.turning:
    #                     import random
    #                     self.turn_direction = random.choice([1, -1])
    #                     self.turning = True
    #                     self.turn_frames = 0
    #                     forSpd = 0.0
    #                     turnSpd = self.deg_turn * self.turn_direction
    #                     print(f"Crossroad detected {self.crossroad_count} times, initiating 90-degree {'left' if self.turn_direction > 0 else 'right'} turn")
    #                     self.crossroad_count = 0
    #                     yield forSpd, turnSpd
    #                     continue

    #                 # Skip junction logic if a crossroad was recently detected (e.g., within a buffer period)
    #                 if self.crossroad_count > 0:  # Crossroad is being processed or was just processed
    #                     # Adjust col position for crossroad handling (center the robot)
    #                     col = img_width / 2
    #                     print(f"Crossroad in progress (count: {self.crossroad_count}), ignoring junctions")
    #                 else:

    #                     # Adjust col position based on curve sharpness (using consecutive counts)
    #                     col_base = {
    #                         3: img_width / 2,      # On_Track (center)
    #                         5: 3 * img_width / 8,  # Slight_Left (120 pixels)
    #                         6: 5 * img_width / 8,  # Slight_Right (200 pixels)
    #                         0: img_width / 2,
    #                         1: 1 * img_width / 8,
    #                         4: 7 * img_width / 8,
    #                         2: img_width / 2
    #                     }.get(prediction, img_width / 2)

    #                     # Dynamically adjust col for Slight_Left and Slight_Right
    #                     if prediction == 5:  # Slight_Left
    #                         sharpness_factor = min(self.slight_left_count / 3.0, 1.0)  # Increased sensitivity (from 5.0 to 3.0)
    #                         col = col_base - (col_base - img_width / 4) * sharpness_factor  # Wider range (to 80 pixels)
    #                     elif prediction == 6:  # Slight_Right
    #                         sharpness_factor = min(self.slight_right_count / 3.0, 1.0)  # Increased sensitivity
    #                         col = col_base + (3 * img_width / 4 - col_base) * sharpness_factor  # Wider range (to 240 pixels)
    #                     else:
    #                         col = col_base

    #                     # Check for Left_Junction (always turn left)
    #                     if self.left_junction_count > random_turning_J and not self.turning:
    #                         self.turn_direction = 1
    #                         self.turning = True
    #                         self.turn_frames = 0
    #                         forSpd = 0.0
    #                         turnSpd = self.deg_turn * self.turn_direction
    #                         print(f"Left_Junction detected {self.left_junction_count} times, initiating 90-degree left turn")
    #                         self.left_junction_count = 0
    #                         yield forSpd, turnSpd
    #                         continue

    #                     # Check for Right_Junction (always turn right)
    #                     if self.right_junction_count > random_turning_J and not self.turning:
    #                         self.turn_direction = -1
    #                         self.turning = True
    #                         self.turn_frames = 0
    #                         forSpd = 0.0
    #                         turnSpd = self.deg_turn * self.turn_direction
    #                         print(f"Right_Junction detected {self.right_junction_count} times, initiating 90-degree right turn")
    #                         self.right_junction_count = 0
    #                         yield forSpd, turnSpd
    #                         continue

    #                 # Skip PID control if turning
    #                 if self.turning:
    #                     continue

    #                 # Handle speed-up phase after turn
    #                 if self.speeding_up:
    #                     forSpd = speed_up
    #                     turnSpd = 0
    #                     self.speed_up_frames += 1
    #                     print(f"Speeding up: Frame {self.speed_up_frames}/{self.speed_up_duration}, forSpd={forSpd}, turnSpd={turnSpd}")
    #                     if self.speed_up_frames >= self.speed_up_duration:
    #                         self.speeding_up = False
    #                         self.speed_up_frames = 0
    #                         print("Speed-up phase completed, resuming normal speed")
    #                     yield forSpd, turnSpd
    #                     continue

    #                 # Calculate error from center of image (ensure correct sign)
    #                 error = (img_width / 2) - col  # Positive for left, negative for right

    #                 # Smooth the error using a moving average
    #                 self.error_history.append(error)
    #                 if len(self.error_history) > self.error_history_length:
    #                     self.error_history.pop(0)
    #                 smoothed_error = sum(self.error_history) / len(self.error_history)

    #                 # Compute angle using arctan2 with smoothed error
    #                 angle = np.arctan2(smoothed_error, focal_length)

    #                 # Adjust PID gains dynamically based on error magnitude
    #                 error_magnitude = abs(smoothed_error) / (img_width / 2)
    #                 kP = kP_base * (1 + 2.0 * error_magnitude)  # Increased scaling for curves
    #                 kI = kI_base
    #                 kD = kD_base * (1 + 2.0 * error_magnitude)  # Increased scaling for curves
                    
    #                 # PID control for turning speed
    #                 proportional = kP * angle
    #                 integral = kI * integrator.send(angle)
    #                 integral = max(min(integral, self.integral_cap), -self.integral_cap)
    #                 derivative_term = kD * derivative.send(angle)
    #                 turnSpd = proportional + integral + derivative_term


    #             # Cumulative turn speed: High initial boost that decays
    #                 turn_boost = 1.5 - error_magnitude  # Starts at 1.5 when error is max, reduces to 0.5 when centered
    #                 turn_boost = max(0.5, min(turn_boost, 1.5))  # Clamp between 0.5 and 1.5
    #                 turnSpd *= turn_boost
                    
    #                 # Reduce turn amplification for small errors, increase for large errors
    #                 turn_amplification = 0.5 + 2.0 * error_magnitude  # Reduced base (from 1 to 0.5)
    #                 turnSpd *= turn_amplification

    #                 # Adjust maximum turn speed
    #                 max_turn_spd = 0.25 + 0.6 * error_magnitude  # Reduced base (from 0.3 to 0.25), increased scaling
    #                 turnSpd = max(min(turnSpd, max_turn_spd), -max_turn_spd)

    #                 # Stronger stability check for straight lines
    #                 if abs(smoothed_error) < 30:  # Increased threshold
    #                     turnSpd *= 0.3  # More aggressive reduction (from 0.5 to 0.3)

    #                 # Forward speed with more aggressive reduction for sharp curves
    #                 base_speed = 0.25
    #                 speed_reduction = 0.25 * abs(angle)  # Slightly increased reduction
    #                 forSpd = max(base_speed - speed_reduction, 0.01)

    #                 # Debug output to monitor behavior
    #                 # print(f"Prediction: {prediction}, Error: {error:.2f}, Smoothed Error: {smoothed_error:.2f}, turnSpd: {turnSpd:.2f}, forSpd: {forSpd:.2f}")

    #             yield forSpd, turnSpd


## Working with line Follow but no key left and right
    # def line_to_speed_map(self, prediction, sampleRate, saturation, lineFollow=False, lineLeft=False, lineRight=False):
    #     img_width = 320
    #     img_bottom_y = 200
    #     base_forSpd = 0.2
    #     base_turnSpd = 0.01
    #     slow_spd = 0.01
    #     speed_up = 0.25
    #     random_turning_C = 11
    #     random_turning_J = 100
    #     error_threshold = 5
    #     crossroad_forSpd = 0.1

    #     if not hasattr(self, 'turn_duration'):
    #         self.turn_duration = 60
    #         print(f"Sample Rate: {sampleRate}, Turn Duration: {self.turn_duration} frames")
    #     if not hasattr(self, 'deg_turn'):
    #         self.deg_turn = 1.57 / (self.turn_duration * sampleRate)
    #         print(f"deg_turn: {self.deg_turn}")
    #     if not hasattr(self, 'turning'):
    #         self.turning = False
    #     if not hasattr(self, 'turn_direction'):
    #         self.turn_direction = 0
    #     if not hasattr(self, 'turn_frames'):
    #         self.turn_frames = 0
    #     if not hasattr(self, 'crossroad_count'):
    #         self.crossroad_count = 0
    #     if not hasattr(self, 'left_junction_count'):
    #         self.left_junction_count = 0
    #     if not hasattr(self, 'right_junction_count'):
    #         self.right_junction_count = 0
    #     if not hasattr(self, 'last_prediction'):
    #         self.last_prediction = None
    #     if not hasattr(self, 'last_col'):
    #         self.last_col = img_width / 2
    #     if not hasattr(self, 'lost_line_frames'):
    #         self.lost_line_frames = 0
    #     if not hasattr(self, 'speeding_up'):
    #         self.speeding_up = False
    #     if not hasattr(self, 'speed_up_frames'):
    #         self.speed_up_frames = 0
    #     if not hasattr(self, 'speed_up_duration'):
    #         self.speed_up_duration = 30
    #     if not hasattr(self, 'slight_left_count'):
    #         self.slight_left_count = 0
    #     if not hasattr(self, 'slight_right_count'):
    #         self.slight_right_count = 0
    #     if not hasattr(self, 'last_turnSpd'):
    #         self.last_turnSpd = 0
    #     if not hasattr(self, 'last_forSpd'):
    #         self.last_forSpd = base_forSpd
    #     if not hasattr(self, 'large_error_count'):  # New attribute to track large errors
    #         self.large_error_count = 0          

    #     while True:
    #         forSpd, turnSpd = 0, 0

    #         if self.turning:
    #             forSpd = 0.0
    #             turnSpd = self.deg_turn * self.turn_direction
    #             self.turn_frames += 1
    #             print(f"Turning: Frame {self.turn_frames}/{self.turn_duration}, forSpd={forSpd}, turnSpd={turnSpd}")
    #             if self.turn_frames >= self.turn_duration:
    #                 self.turning = False
    #                 self.turn_frames = 0
    #                 self.turn_direction = 0
    #                 self.speeding_up = True
    #                 self.speed_up_frames = 0
    #                 print(f"Completed 90-degree {'left' if self.turn_direction > 0 else 'right'} turn, speeding up")
    #             yield forSpd, turnSpd
    #             continue

    #         prediction_data = yield forSpd, turnSpd
    #         if isinstance(prediction_data, tuple) and len(prediction_data) == 2:
    #             prediction, col = prediction_data
    #         else:
    #             prediction = prediction_data
    #             col = None

    #         if prediction is not None:
    #             if lineLeft:
    #                 # ... (unchanged) ...
    #                 pass
    #             elif lineRight:
    #                 # ... (unchanged) ...
    #                 pass
    #             elif lineFollow:
    #                 if prediction == 5:
    #                     self.slight_left_count += 1
    #                     self.slight_right_count = 0
    #                 elif prediction == 6:
    #                     self.slight_right_count += 1
    #                     self.slight_left_count = 0
    #                 else:
    #                     self.slight_left_count = max(0, self.slight_left_count - 1)
    #                     self.slight_right_count = max(0, self.slight_right_count - 1)

    #                 if prediction == 0:
    #                     self.crossroad_count += 1
    #                     self.left_junction_count = 0
    #                     self.right_junction_count = 0
    #                 elif prediction == 1:
    #                     self.left_junction_count += 1
    #                     self.crossroad_count = 0
    #                     self.right_junction_count = 0
    #                 elif prediction == 4:
    #                     self.right_junction_count += 1
    #                     self.crossroad_count = 0
    #                     self.left_junction_count = 0
    #                 else:
    #                     self.crossroad_count = 0
    #                     self.left_junction_count = 0
    #                     self.right_junction_count = 0

    #                 if self.crossroad_count > random_turning_C and not self.turning:
    #                     import random
    #                     self.turn_direction = random.choice([1, -1])
    #                     self.turning = True
    #                     self.turn_frames = 0
    #                     forSpd = 0.0
    #                     turnSpd = self.deg_turn * self.turn_direction
    #                     print(f"Crossroad detected {self.crossroad_count} times, initiating 90-degree turn")
    #                     self.crossroad_count = 0
    #                     continue

    #                 error = 0
    #                 if col is not None:
    #                     error = (img_width / 2) - col

    #                 if self.crossroad_count > 0:
    #                     col = img_width / 2
    #                     error = 0
    #                     print(f"Crossroad in progress (count: {self.crossroad_count}), ignoring junctions")
    #                     forSpd = crossroad_forSpd
    #                     turnSpd = 0
    #                 else:
    #                     if self.left_junction_count > random_turning_J and not self.turning:
    #                         self.turn_direction = 1
    #                         self.turning = True
    #                         self.turn_frames = 0
    #                         forSpd = 0.0
    #                         turnSpd = self.deg_turn * self.turn_direction
    #                         print(f"Left_Junction detected {self.left_junction_count} times, initiating 90-degree left turn")
    #                         self.left_junction_count = 0
    #                         continue
    #                     if self.right_junction_count > random_turning_J and not self.turning:
    #                         self.turn_direction = -1
    #                         self.turning = True
    #                         self.turn_frames = 0
    #                         forSpd = 0.0
    #                         turnSpd = self.deg_turn * self.turn_direction
    #                         print(f"Right_Junction detected {self.right_junction_count} times, initiating 90-degree right turn")
    #                         self.right_junction_count = 0
    #                         continue
                        
    #                     if prediction == 2:
    #                         decay_factor = math.exp(-5)
    #                         forSpd = self.last_forSpd * decay_factor
                           
                            
    #                     # Handle Off_Track (no blob or Prediction: 2)
    #                     if col is None:
    #                         self.lost_line_frames += 1
    #                         if self.lost_line_frames < 40:
    #                             decay_factor = math.exp(-self.lost_line_frames / 10.0)
    #                             forSpd = self.last_forSpd * decay_factor
    #                             turnSpd = self.last_turnSpd * decay_factor
    #                             if col is None:
    #                                 col = self.last_col + (img_width / 2 - self.last_col) * 0.4
    #                             print(f"Off_Track, adjusting col: {col}, frames lost: {self.lost_line_frames}, forSpd: {forSpd}, turnSpd: {turnSpd}")
    #                         else:
    #                             col = self.last_col
    #                             forSpd = 0
    #                             turnSpd = 0
    #                             print("Line lost for too long, stopping")
    #                     else:
    #                         self.last_col = col
    #                         self.lost_line_frames = 0

    #                     if self.turning:
    #                         continue

    #                     # Speed-up logic (only applies when on track and speeding up)
    #                     if self.speeding_up and self.lost_line_frames == 0:
    #                         forSpd = speed_up
    #                         turnSpd = 0
    #                         self.speed_up_frames += 1
    #                         if self.speed_up_frames >= self.speed_up_duration:
    #                             self.speeding_up = False
    #                             self.speed_up_frames = 0
    #                             print("Speed-up phase completed")
    #                         yield forSpd, turnSpd
    #                         continue

    #                     # Exponential control only when on track
    #                     if self.crossroad_count == 0 and self.lost_line_frames == 0:
    #                         error_magnitude = abs(error) / (img_width / 2)

    #                         # Update large error count
    #                         if abs(error) > 120:
    #                             self.large_error_count += 1
    #                         else:
    #                             self.large_error_count = 0

    #                         if abs(error) <= 20:
    #                             if error_magnitude > error_threshold / (img_width / 2):  # Error > 5 pixels
    #                                 # turnSpd = base_turnSpd * math.exp(0.1 * error_magnitude)
    #                                 turnSpd = 0 #base_turnSpd * error_magnitude
    #                                 max_turnSpd = 2.0
    #                                 turnSpd = turnSpd if error > 0 else -turnSpd
    #                                 turnSpd = max(min(turnSpd, max_turnSpd), -max_turnSpd)
    #                             else:
    #                                 turnSpd = 0
    #                             forSpd = 2*base_forSpd * math.exp(-2 * error_magnitude)
    #                             forSpd = max(forSpd, slow_spd * 0.5)
                                
    #                         elif abs(error) > 20 and abs(error) <= 50:
    #                             if error_magnitude > error_threshold / (img_width / 2):  # Error > 5 pixels
    #                                 # turnSpd = base_turnSpd * math.exp(0.1 * error_magnitude)
    #                                 turnSpd = base_turnSpd * error_magnitude
    #                                 max_turnSpd = 2.0
    #                                 turnSpd = turnSpd if error > 0 else -turnSpd
    #                                 turnSpd = max(min(turnSpd, max_turnSpd), -max_turnSpd)
    #                             else:
    #                                 turnSpd = 0
    #                             forSpd = base_forSpd * math.exp(-0.8 * error_magnitude)
    #                             forSpd = max(forSpd, slow_spd * 0.5)
                                
    #                         elif abs(error) > 50 and abs(error) <= 90:
    #                             if error_magnitude > error_threshold / (img_width / 2):  # Error > 5 pixels
    #                                 turnSpd = 3* base_turnSpd * math.exp(2 * error_magnitude)
    #                                 max_turnSpd = 2.0
    #                                 turnSpd = turnSpd if error > 0 else -turnSpd
    #                                 turnSpd = max(min(turnSpd, max_turnSpd), -max_turnSpd)
    #                             else:
    #                                 turnSpd = 0
    #                             forSpd = base_forSpd * math.exp(-1 * error_magnitude)
    #                             forSpd = max(forSpd, slow_spd * 0.5)
    #                         elif abs(error) > 90 and abs(error) <= 120:
    #                             if error_magnitude > error_threshold / (img_width / 2):  # Error > 5 pixels
    #                                 turnSpd = 4 * base_turnSpd * math.exp(2 * error_magnitude)
    #                                 max_turnSpd = 2.0
    #                                 turnSpd = turnSpd if error > 0 else -turnSpd
    #                                 turnSpd = max(min(turnSpd, max_turnSpd), -max_turnSpd)
    #                             else:
    #                                 turnSpd = 0
    #                             forSpd = base_forSpd * math.exp(-0.8 * error_magnitude)
    #                             forSpd = max(forSpd, slow_spd * 0.5)
    #                         elif abs(error) > 120 and abs(error) <= 130:  #self.large_error_count > 2:
    #                             if error_magnitude > error_threshold / (img_width / 2):  # Error > 5 pixels
    #                                     turnSpd = 5 * base_turnSpd * math.exp(2 * error_magnitude)
    #                                     max_turnSpd = 2.0
    #                                     turnSpd = turnSpd if error > 0 else -turnSpd
    #                                     turnSpd = max(min(turnSpd, max_turnSpd), -max_turnSpd)
    #                             else:
    #                                 turnSpd = 0
    #                             forSpd = base_forSpd * math.exp(-2 * error_magnitude)
    #                             forSpd = max(forSpd, slow_spd * 0.5)
                                
    #                         else:  # abs(error) > 140 but count <= 2, use moderate response
    #                             if error_magnitude > error_threshold / (img_width / 2):  # Error > 5 pixels
    #                                 turnSpd = 6 * base_turnSpd * math.exp(2 * error_magnitude)
    #                                 max_turnSpd = 2.0
    #                                 turnSpd = turnSpd if error > 0 else -turnSpd
    #                                 turnSpd = max(min(turnSpd, max_turnSpd), -max_turnSpd)
    #                             else:
    #                                 turnSpd = 0
    #                             forSpd = base_forSpd * math.exp(-2 * error_magnitude)
    #                             forSpd = -max(forSpd, slow_spd * 0.5)
                            
                        

                            
                            
    #                 self.last_turnSpd = turnSpd
    #                 self.last_forSpd = forSpd

    #                 print(f"Prediction: {prediction}, Col: {col}, Error: {error:.2f}, forSpd: {forSpd:.2f}, turnSpd: {turnSpd:.2f}")

    #         yield forSpd, turnSpd



    def line_to_speed_map(self, prediction, sampleRate, saturation, lineFollow=False, lineLeft=False, lineRight=False):
        img_width = 320
        img_bottom_y = 200
        base_forSpd = 0.1
        base_turnSpd = 0.018
    
        slow_spd = 0.01
        speed_up = 0.2
        random_turning_C = 2    
        random_turning_J = 100
        error_threshold = 5
        crossroad_forSpd = 0.1

        if not hasattr(self, 'turn_duration'):
            self.turn_duration = 80
            print(f"Sample Rate: {sampleRate}, Turn Duration: {self.turn_duration} frames")
        if not hasattr(self, 'deg_turn'):
            self.deg_turn = 1.57 / (self.turn_duration * sampleRate)
            print(f"deg_turn: {self.deg_turn}")
        if not hasattr(self, 'turning'):
            self.turning = False
        if not hasattr(self, 'turn_direction'):
            self.turn_direction = 0
        if not hasattr(self, 'turn_frames'):
            self.turn_frames = 0
        if not hasattr(self, 'crossroad_count'):
            self.crossroad_count = 0
        if not hasattr(self, 'left_junction_count'):
            self.left_junction_count = 0
        if not hasattr(self, 'right_junction_count'):
            self.right_junction_count = 0
        if not hasattr(self, 'last_prediction'):
            self.last_prediction = None
        if not hasattr(self, 'last_col'):
            self.last_col = img_width / 2
        if not hasattr(self, 'lost_line_frames'):
            self.lost_line_frames = 0
        if not hasattr(self, 'speeding_up'):
            self.speeding_up = False
        if not hasattr(self, 'speed_up_frames'):
            self.speed_up_frames = 0
        if not hasattr(self, 'speed_up_duration'):
            self.speed_up_duration = 30
        if not hasattr(self, 'slight_left_count'):
            self.slight_left_count = 0
        if not hasattr(self, 'slight_right_count'):
            self.slight_right_count = 0
        if not hasattr(self, 'last_turnSpd'):
            self.last_turnSpd = 0
        if not hasattr(self, 'last_forSpd'):
            self.last_forSpd = base_forSpd
        if not hasattr(self, 'large_error_count'):
            self.large_error_count = 0  
        if not hasattr(self, 'turn_cooldown'):  # New cooldown attribute
            self.turn_cooldown = 0  
        if not hasattr(self, 'turn_reason'):  # New attribute to track turn reason
            self.turn_reason = None  # Can be 'crossroad', 'junction', or None     

        while True:
            forSpd, turnSpd, target_forSpd, target_turnSpd = 0, 0, 0, 0
            if self.turning:
                forSpd = -0.2 * base_forSpd  # Keep your specified forward speed during turn
                turnSpd = self.deg_turn * self.turn_direction
                self.turn_frames += 1
                print(self.turn_reason) 
                print(f"Turning: Frame {self.turn_frames}/{self.turn_duration}, forSpd={forSpd}, turnSpd={turnSpd}")

                # Get the latest prediction data to check col during the turn
                prediction_data = yield forSpd, turnSpd
                if isinstance(prediction_data, tuple) and len(prediction_data) == 2:
                    prediction, col = prediction_data
                else:
                    prediction = prediction_data
                    col = None

                # Update last_col if col is valid
                if col is not None:
                    self.last_col = col
                    self.lost_line_frames = 0
                else:
                    self.lost_line_frames += 1

                # Check if centered and stop turning early
                center_threshold = 20  # Stop if within 20 pixels of center
                 
                # if (self.turn_reason != 'junction' and col is not None and abs(col - img_width / 2) <= center_threshold and self.turn_frames > 40):  # Minimum frames to prevent instant stop
                if (self.turn_reason != 'crossroad' and 
                col is not None and 
                abs(col - img_width / 2) <= center_threshold and 
                self.turn_frames > 75 and self.turn_cooldown == 0):
                    current_frame = self.turn_frames  # Store current frame before reset
                    self.turning = False
                    self.turn_frames = 0
                    if self.turn_direction == -1:  # Right turn (for lineRight)
                        self.crossroad_count = 0
                        self.right_junction_count = 0
                    elif self.turn_direction == 1:  # Left turn (for lineLeft)
                        self.crossroad_count = 0
                        self.left_junction_count = 0
                    self.turn_direction = 0
                    self.turn_reason = None
                    self.speeding_up = True
                    self.speed_up_frames = 0
                    self.turn_cooldown = 120
                    print(f"Line centered (col={col}), stopping turn early at frame {current_frame}, cooldown set to {self.turn_cooldown} frames")
                    continue

                # Complete turn if duration reached (fallback if line not detected)
                if self.turn_frames >= self.turn_duration:
                    self.turning = False
                    self.turn_frames = 0
                    if self.turn_direction == -1:  # Right turn (for lineRight)
                        self.crossroad_count = 0
                        self.right_junction_count = 0
                    elif self.turn_direction == 1:  # Left turn (for lineLeft)
                        self.crossroad_count = 0
                        self.left_junction_count = 0
                    self.turn_direction = 0
                    self.speeding_up = True
                    self.speed_up_frames = 0
                    self.turn_cooldown = 120  # Your fixed 2-second cooldown
                    print(f"Completed 90-degree {'left' if self.turn_direction > 0 else 'right'} turn, speeding up, cooldown set to {self.turn_cooldown} frames")
                continue



# Working with fix 60 frames turning
            # if self.turning:
            #     forSpd = -0.2*base_forSpd
            #     turnSpd = self.deg_turn * self.turn_direction
            #     self.turn_frames += 1
            #     print(f"Turning: Frame {self.turn_frames}/{self.turn_duration}, forSpd={forSpd}, turnSpd={turnSpd}")
            #     if self.turn_frames >= self.turn_duration:
            #         self.turning = False
            #         self.turn_frames = 0
            #         if self.turn_direction == -1:  # Right turn (for lineRight)
            #             self.crossroad_count = 0
            #             self.right_junction_count = 0
            #         elif self.turn_direction == 1:  # Left turn (for lineLeft)
            #             self.crossroad_count = 0
            #             self.left_junction_count = 0
            #         self.turn_direction = 0
            #         self.speeding_up = True
            #         self.speed_up_frames = 0
            #         self.turn_cooldown = 120# int(2 * sampleRate)  # 2 seconds cooldown (e.g., 60 frames at 30 FPS)
            #         print(f"Completed 90-degree {'left' if self.turn_direction > 0 else 'right'} turn, speeding up, cooldown set to {self.turn_cooldown} frames")
            #     yield forSpd, turnSpd
            #     continue
                # Decrement cooldown if active
            if self.turn_cooldown > 0:
                self.turn_cooldown -= 1
                print(f"Cooldown active: {self.turn_cooldown} frames remaining")
                
            prediction_data = yield forSpd, turnSpd
            if isinstance(prediction_data, tuple) and len(prediction_data) == 2:
                prediction, col = prediction_data
            else:
                prediction = prediction_data
                col = None

            if prediction is not None:
                # Common counters update for all modes
                if prediction == 0:
                    self.crossroad_count += 1
                    self.left_junction_count = 0
                    self.right_junction_count = 0
                elif prediction == 1:
                    self.left_junction_count += 1
                    self.crossroad_count = 0
                    self.right_junction_count = 0
                elif prediction == 4:
                    self.right_junction_count += 1
                    self.crossroad_count = 0
                    self.left_junction_count = 0
                else:
                    self.crossroad_count = 0
                    self.left_junction_count = 0
                    self.right_junction_count = 0

                if prediction == 5:
                    self.slight_left_count += 1
                    self.slight_right_count = 0
                elif prediction == 6:
                    self.slight_right_count += 1
                    self.slight_left_count = 0
                else:
                    self.slight_left_count = max(0, self.slight_left_count - 1)
                    self.slight_right_count = max(0, self.slight_right_count - 1)

                error = 0
                if col is not None:
                    error = (img_width / 2) - col

                if lineLeft:
                    if not self.turning and self.turn_cooldown == 0:
                        if self.crossroad_count > 0:
                            self.turn_direction = 1
                            self.turning = True
                            self.turn_frames = 0
                            self.turn_reason = 'junction'
                            forSpd = -0.2 * base_forSpd
                            turnSpd = self.deg_turn * self.turn_direction
                            print(f"Crossroad detected in lineLeft, initiating 90-degree left turn")
                            continue
                        elif prediction == 1 and self.left_junction_count > 0:
                            self.turn_direction = 1
                            self.turning = True
                            self.turn_frames = 0
                            self.turn_reason = 'junction'
                            forSpd = -0.2 * base_forSpd
                            turnSpd = self.deg_turn * self.turn_direction
                            print(f"Left_Junction detected in lineLeft, initiating 90-degree left turn")
                            continue
                
                    # Handle Off_Track (no blob)
                    if col is None:
                        self.lost_line_frames += 1
                        if self.lost_line_frames < 60:
                            decay_factor = math.exp(-self.lost_line_frames / 20.0)
                            forSpd = self.last_forSpd * decay_factor
                            turnSpd = self.last_turnSpd * decay_factor*1.2
                            col = self.last_col + (img_width / 2 - self.last_col) * 0.4
                            print(f"Off_Track, adjusting col: {col}, frames lost: {self.lost_line_frames}, forSpd: {forSpd}, turnSpd: {turnSpd}")
                        else:
                            col = self.last_col
                            forSpd = 0
                            turnSpd = 0
                            print("Line lost for too long, stopping")
                    else:
                        self.last_col = col
                        self.lost_line_frames = 0

                    if self.turning:
                        continue

                    if self.speeding_up and self.lost_line_frames == 0:
                        forSpd = speed_up
                        turnSpd = 0
                        self.speed_up_frames += 1
                        if self.speed_up_frames >= self.speed_up_duration:
                            self.speeding_up = False
                            self.speed_up_frames = 0
                            print("Speed-up phase completed")
                        yield forSpd, turnSpd
                        continue

                #     # Movement logic for lineLeft
                #     if self.crossroad_count == 0 and self.lost_line_frames == 0:
                #             error_magnitude = abs(error) / (img_width / 2)
                #             if abs(error) <= 20:
                #                 if error_magnitude > error_threshold / (img_width / 2):
                #                     turnSpd = 0
                #                     # target_turnSpd = base_turnSpd * error_magnitude  # Fixed bug
                #                     max_turnSpd = 2.0
                #                     turnSpd = turnSpd if error > 0 else -turnSpd
                #                     turnSpd = max(min(turnSpd, max_turnSpd), -max_turnSpd)
                #                     # target_turnSpd = target_turnSpd if error > 0 else -target_turnSpd
                #                     # target_turnSpd = max(min(target_turnSpd, max_turnSpd), -max_turnSpd)
                #                 else:
                #                     turnSpd = 0
                #                 # forSpd = 2 * base_forSpd * math.exp(-2 * error_magnitude)
                #                 # forSpd = max(forSpd, slow_spd * 0.5)
                #                 target_forSpd = 2*base_forSpd * math.exp(-2 * error_magnitude)
                #                 target_forSpd = max(target_forSpd, slow_spd * 0.5)
                                
                #             elif abs(error) > 20 and abs(error) <= 50:
                #                 if error_magnitude > error_threshold / (img_width / 2):
                #                     turnSpd = base_turnSpd * error_magnitude
                #                     # target_turnSpd = base_turnSpd * math.exp(0.5 * error_magnitude)
                #                     max_turnSpd = 2.0
                #                     turnSpd = turnSpd if error > 0 else -turnSpd
                #                     turnSpd = max(min(turnSpd, max_turnSpd), -max_turnSpd)
                #                     # target_turnSpd = target_turnSpd if error > 0 else -target_turnSpd
                #                     # target_turnSpd = max(min(target_turnSpd, max_turnSpd), -max_turnSpd)
                #                 else:
                #                     turnSpd = 0
                #                 # forSpd = base_forSpd * math.exp(-0.8 * error_magnitude)
                #                 # forSpd = max(forSpd, slow_spd * 0.5)
                #                 target_forSpd = base_forSpd * math.exp(-0.8 * error_magnitude)
                #                 target_forSpd = max(target_forSpd, slow_spd * 0.5)
                                
                #             elif abs(error) > 50 and abs(error) <= 90:
                #                 if error_magnitude > error_threshold / (img_width / 2):
                #                     turnSpd = 3 * base_turnSpd * math.exp(2 * error_magnitude)
                #                     # target_turnSpd = 2 * base_turnSpd * math.exp(2.2 * error_magnitude)
                #                     max_turnSpd = 2.0
                #                     turnSpd = turnSpd if error > 0 else -turnSpd
                #                     turnSpd = max(min(turnSpd, max_turnSpd), -max_turnSpd)
                #                     # target_turnSpd = target_turnSpd if error > 0 else -target_turnSpd
                #                     # target_turnSpd = max(min(target_turnSpd, max_turnSpd), -max_turnSpd)
                #                 else:
                #                     turnSpd = 0
                #                 # forSpd = base_forSpd * math.exp(-1 * error_magnitude)
                #                 # forSpd = max(forSpd, slow_spd * 0.5)
                #                 target_forSpd = base_forSpd * math.exp(-1 * error_magnitude)
                #                 target_forSpd = max(target_forSpd, slow_spd * 0.5)
                                
                                
                #             elif abs(error) > 90 and abs(error) <= 120:
                #                 if error_magnitude > error_threshold / (img_width / 2):
                #                     turnSpd = 5.5 * base_turnSpd * math.exp(2 * error_magnitude)
                #                     # target_turnSpd = 2 * base_turnSpd * math.exp(2.5 * error_magnitude)
                #                     max_turnSpd = 2.0
                #                     turnSpd = turnSpd if error > 0 else -turnSpd
                #                     turnSpd = max(min(turnSpd, max_turnSpd), -max_turnSpd)
                #                     # target_turnSpd = target_turnSpd if error > 0 else -target_turnSpd
                #                     # target_turnSpd = max(min(target_turnSpd, max_turnSpd), -max_turnSpd)
                #                 else:
                #                     turnSpd = 0
                #                 # forSpd = base_forSpd * math.exp(-0.8 * error_magnitude)
                #                 # forSpd = max(forSpd, slow_spd * 0.5)
                #                 target_forSpd = base_forSpd * math.exp(-2 * error_magnitude)
                #                 target_forSpd = max(target_forSpd, slow_spd * 0.5)
                                
                                
                #             elif abs(error) > 120 and abs(error) <= 130:
                #                 if error_magnitude > error_threshold / (img_width / 2):
                #                     turnSpd = 6 * base_turnSpd * math.exp(2 * error_magnitude)
                #                     # target_turnSpd = 3 * base_turnSpd * math.exp(2 * error_magnitude)
                #                     max_turnSpd = 2.0
                #                     turnSpd = turnSpd if error > 0 else -turnSpd
                #                     turnSpd = max(min(turnSpd, max_turnSpd), -max_turnSpd)
                #                     # target_turnSpd = target_turnSpd if error > 0 else -target_turnSpd
                #                     # target_turnSpd = max(min(target_turnSpd, max_turnSpd), -max_turnSpd)
                #                 else:
                #                     turnSpd = 0
                #                 # forSpd = base_forSpd * math.exp(-2 * error_magnitude)
                #                 # forSpd = max(forSpd, slow_spd * 0.5)
                #                 target_forSpd = base_forSpd * math.exp(-2 * error_magnitude)
                #                 target_forSpd = max(target_forSpd, slow_spd * 0.5)
                                
                #             elif abs(error) > 130 and abs(error) <= 140:
                #                 if error_magnitude > error_threshold / (img_width / 2):
                #                     turnSpd = 7 * base_turnSpd * math.exp(2 * error_magnitude)
                #                     # target_turnSpd = 5 * base_turnSpd * math.exp(2 * error_magnitude)
                #                     max_turnSpd = 2.0
                #                     turnSpd = turnSpd if error > 0 else -turnSpd
                #                     turnSpd = max(min(turnSpd, max_turnSpd), -max_turnSpd)
                #                     # target_turnSpd = target_turnSpd if error > 0 else -target_turnSpd
                #                     # target_turnSpd = max(min(target_turnSpd, max_turnSpd), -max_turnSpd)
                                    
                #                 else:
                #                     turnSpd = 0
                #                 # forSpd = base_forSpd * math.exp(-2 * error_magnitude)
                #                 # forSpd = max(forSpd, slow_spd * 0.5)
                #                 target_forSpd = base_forSpd * math.exp(-2 * error_magnitude)
                #                 target_forSpd = max(target_forSpd, slow_spd * 0.5)
                                
                #             else:  # abs(error) > 140
                #                 if error_magnitude > error_threshold / (img_width / 2):
                #                     turnSpd = 8 * base_turnSpd * math.exp(2 * error_magnitude)
                #                     # target_turnSpd = 4 * base_turnSpd * math.exp(2 * error_magnitude)
                #                     max_turnSpd = 2.0
                #                     turnSpd = turnSpd if error > 0 else -turnSpd
                #                     turnSpd = max(min(turnSpd, max_turnSpd), -max_turnSpd)
                #                     # target_turnSpd = target_turnSpd if error > 0 else -target_turnSpd
                #                     # target_turnSpd = max(min(target_turnSpd, max_turnSpd), -max_turnSpd)
                                    
                #                 else:
                #                     turnSpd = 0
                #                 # forSpd = base_forSpd * math.exp(-2 * error_magnitude)
                #                 # forSpd = -max(forSpd, slow_spd * 0.5)
                #                 target_forSpd = base_forSpd * math.exp(-2 * error_magnitude)
                #                 target_forSpd = max(target_forSpd, slow_spd * 0.5)
                                
                # # Apply speed-up logic first, before smoothing
                #     if self.speeding_up and self.lost_line_frames == 0:
                #         forSpd = speed_up  # Directly set to 0.25 during speed-up
                #         turnSpd = 0
                #         self.speed_up_frames += 1
                #         if self.speed_up_frames >= self.speed_up_duration:
                #             self.speeding_up = False
                #             self.speed_up_frames = 0
                #             print("Speed-up phase completed")
                #         self.last_forSpd = forSpd  # Update last_forSpd to avoid abrupt drop after speed-up
                #         print(f"Speeding up: Frame {self.speed_up_frames}/{self.speed_up_duration}, forSpd={forSpd:.2f}, turnSpd={turnSpd:.2f}")
                #         yield forSpd, turnSpd
                #         continue
                    
                #     # Handle crossroad speed
                #     if self.crossroad_count > 0:
                #         target_forSpd = crossroad_forSpd  # 0.1
                #         target_turnSpd = 0  # No turning during crossroad

                #     # Apply smoothing only when not speeding up
                #     if not self.speeding_up:
                #         alpha_forward = 0.9  # Increase alpha for faster response (was 0.1)
                #         alpha_turn = 0.7
                #         forSpd = (1 - alpha_forward) * self.last_forSpd + alpha_forward * target_forSpd
                #         forSpd = max(forSpd, 0.05)  # Set a higher minimum speed (was slow_spd * 0.5 = 0.005)
                        
                        
                    if self.crossroad_count == 0 and self.lost_line_frames == 0:
                        error_magnitude = abs(error) / (img_width / 2)
                        if abs(error) <= 20:
                            if error_magnitude > error_threshold / (img_width / 2):  # Error > 5 pixels
                                turnSpd = 0
                                max_turnSpd = 2.0
                                turnSpd = turnSpd if error > 0 else -turnSpd
                                turnSpd = max(min(turnSpd, max_turnSpd), -max_turnSpd)
                            else:
                                turnSpd = 0
                            forSpd = 2 * base_forSpd * math.exp(-2 * error_magnitude)
                            forSpd = max(forSpd, slow_spd * 0.5)
                        elif abs(error) > 20 and abs(error) <= 50:
                            if error_magnitude > error_threshold / (img_width / 2):
                                turnSpd = base_turnSpd * error_magnitude
                                max_turnSpd = 2.0
                                turnSpd = turnSpd if error > 0 else -turnSpd
                                turnSpd = max(min(turnSpd, max_turnSpd), -max_turnSpd)
                            else:
                                turnSpd = 0
                            forSpd = base_forSpd * math.exp(-0.8 * error_magnitude)
                            forSpd = max(forSpd, slow_spd * 0.5)
                        elif abs(error) > 50 and abs(error) <= 90:
                            if error_magnitude > error_threshold / (img_width / 2):
                                turnSpd = 3 * base_turnSpd * math.exp(2 * error_magnitude)
                                max_turnSpd = 2.0
                                turnSpd = turnSpd if error > 0 else -turnSpd
                                turnSpd = max(min(turnSpd, max_turnSpd), -max_turnSpd)
                            else:
                                turnSpd = 0
                            forSpd = base_forSpd * math.exp(-1 * error_magnitude)
                            forSpd = max(forSpd, slow_spd * 0.5)
                        elif abs(error) > 90 and abs(error) <= 120:
                            if error_magnitude > error_threshold / (img_width / 2):
                                turnSpd = 4 * base_turnSpd * math.exp(2 * error_magnitude)
                                max_turnSpd = 2.0
                                turnSpd = turnSpd if error > 0 else -turnSpd
                                turnSpd = max(min(turnSpd, max_turnSpd), -max_turnSpd)
                            else:
                                turnSpd = 0
                            forSpd = base_forSpd * math.exp(-0.8 * error_magnitude)
                            forSpd = max(forSpd, slow_spd * 0.5)
                        elif abs(error) > 120 and abs(error) <= 130:
                            if error_magnitude > error_threshold / (img_width / 2):
                                turnSpd = 5 * base_turnSpd * math.exp(2 * error_magnitude)
                                max_turnSpd = 2.0
                                turnSpd = turnSpd if error > 0 else -turnSpd
                                turnSpd = max(min(turnSpd, max_turnSpd), -max_turnSpd)
                            else:
                                turnSpd = 0
                            forSpd = base_forSpd * math.exp(-2 * error_magnitude)
                            forSpd = max(forSpd, slow_spd * 0.5)
                        
                        elif abs(error) > 130 and abs(error) <= 140:
                                if error_magnitude > error_threshold / (img_width / 2):
                                    turnSpd = 7 * base_turnSpd * math.exp(2 * error_magnitude)
                                    max_turnSpd = 2.0
                                    turnSpd = turnSpd if error > 0 else -turnSpd
                                    turnSpd = max(min(turnSpd, max_turnSpd), -max_turnSpd)
                                else:
                                    turnSpd = 0
                                forSpd = base_forSpd * math.exp(-2 * error_magnitude)
                                forSpd = max(forSpd, slow_spd * 0.5)
                                
                        else:  # abs(error) > 140
                            if error_magnitude > error_threshold / (img_width / 2):
                                turnSpd = 8 * base_turnSpd * math.exp(2 * error_magnitude)
                                max_turnSpd = 2.0
                                turnSpd = turnSpd if error > 0 else -turnSpd
                                turnSpd = max(min(turnSpd, max_turnSpd), -max_turnSpd)
                            else:
                                turnSpd = 0
                            forSpd = base_forSpd * math.exp(-2 * error_magnitude)
                            forSpd = -max(forSpd, slow_spd * 0.5)

                elif lineRight:
                  
                    if not self.turning and self.turn_cooldown == 0:
                        if self.crossroad_count > 0:
                            self.turn_direction = -1
                            self.turning = True
                            self.turn_frames = 0
                            self.turn_reason = 'junction'
                            forSpd = -0.2 * base_forSpd
                            turnSpd = self.deg_turn * self.turn_direction
                            print(f"Crossroad detected in lineRight, initiating 90-degree right turn")
                            continue
                        elif prediction == 4 and self.right_junction_count > 0:
                            self.turn_direction = -1
                            self.turning = True
                            self.turn_frames = 0
                            self.turn_reason = 'junction'
                            forSpd = -0.2 * base_forSpd
                            turnSpd = self.deg_turn * self.turn_direction
                            print(f"Right_Junction detected in lineRight, initiating 90-degree right turn")
                            continue

                    # Handle Off_Track (no blob)
                    if col is None:
                        self.lost_line_frames += 1
                        if self.lost_line_frames < 60:
                            decay_factor = math.exp(-self.lost_line_frames / 20.0)
                            forSpd = self.last_forSpd * decay_factor
                            turnSpd = self.last_turnSpd * decay_factor*1.2
                            col = self.last_col + (img_width / 2 - self.last_col) * 0.4
                            print(f"Off_Track, adjusting col: {col}, frames lost: {self.lost_line_frames}, forSpd: {forSpd}, turnSpd: {turnSpd}")
                        else:
                            col = self.last_col
                            forSpd = 0
                            turnSpd = 0
                            print("Line lost for too long, stopping")
                    else:
                        self.last_col = col
                        self.lost_line_frames = 0

                    if self.turning:
                        continue

                    if self.speeding_up and self.lost_line_frames == 0:
                        forSpd = speed_up
                        turnSpd = 0
                        self.speed_up_frames += 1
                        if self.speed_up_frames >= self.speed_up_duration:
                            self.speeding_up = False
                            self.speed_up_frames = 0
                            print("Speed-up phase completed")
                        yield forSpd, turnSpd
                        continue
                    
                    
                #     if self.crossroad_count == 0 and self.lost_line_frames == 0:
                #             error_magnitude = abs(error) / (img_width / 2)
                #             if abs(error) <= 20:
                #                 if error_magnitude > error_threshold / (img_width / 2):
                #                     turnSpd = 0
                #                     # target_turnSpd = base_turnSpd * error_magnitude  # Fixed bug
                #                     max_turnSpd = 2.0
                #                     turnSpd = turnSpd if error > 0 else -turnSpd
                #                     turnSpd = max(min(turnSpd, max_turnSpd), -max_turnSpd)
                #                     # target_turnSpd = target_turnSpd if error > 0 else -target_turnSpd
                #                     # target_turnSpd = max(min(target_turnSpd, max_turnSpd), -max_turnSpd)
                #                 else:
                #                     turnSpd = 0
                #                 # forSpd = 2 * base_forSpd * math.exp(-2 * error_magnitude)
                #                 # forSpd = max(forSpd, slow_spd * 0.5)
                #                 target_forSpd = 2*base_forSpd * math.exp(-2 * error_magnitude)
                #                 target_forSpd = max(target_forSpd, slow_spd * 0.5)
                                
                #             elif abs(error) > 20 and abs(error) <= 50:
                #                 if error_magnitude > error_threshold / (img_width / 2):
                #                     turnSpd = base_turnSpd * error_magnitude
                #                     # target_turnSpd = base_turnSpd * math.exp(0.5 * error_magnitude)
                #                     max_turnSpd = 2.0
                #                     turnSpd = turnSpd if error > 0 else -turnSpd
                #                     turnSpd = max(min(turnSpd, max_turnSpd), -max_turnSpd)
                #                     # target_turnSpd = target_turnSpd if error > 0 else -target_turnSpd
                #                     # target_turnSpd = max(min(target_turnSpd, max_turnSpd), -max_turnSpd)
                #                 else:
                #                     turnSpd = 0
                #                 # forSpd = base_forSpd * math.exp(-0.8 * error_magnitude)
                #                 # forSpd = max(forSpd, slow_spd * 0.5)
                #                 target_forSpd = base_forSpd * math.exp(-0.8 * error_magnitude)
                #                 target_forSpd = max(target_forSpd, slow_spd * 0.5)
                                
                #             elif abs(error) > 50 and abs(error) <= 90:
                #                 if error_magnitude > error_threshold / (img_width / 2):
                #                     turnSpd = 3 * base_turnSpd * math.exp(2 * error_magnitude)
                #                     # target_turnSpd = 2 * base_turnSpd * math.exp(2.2 * error_magnitude)
                #                     max_turnSpd = 2.0
                #                     turnSpd = turnSpd if error > 0 else -turnSpd
                #                     turnSpd = max(min(turnSpd, max_turnSpd), -max_turnSpd)
                #                     # target_turnSpd = target_turnSpd if error > 0 else -target_turnSpd
                #                     # target_turnSpd = max(min(target_turnSpd, max_turnSpd), -max_turnSpd)
                #                 else:
                #                     turnSpd = 0
                #                 # forSpd = base_forSpd * math.exp(-1 * error_magnitude)
                #                 # forSpd = max(forSpd, slow_spd * 0.5)
                #                 target_forSpd = base_forSpd * math.exp(-1 * error_magnitude)
                #                 target_forSpd = max(target_forSpd, slow_spd * 0.5)
                                
                                
                #             elif abs(error) > 90 and abs(error) <= 120:
                #                 if error_magnitude > error_threshold / (img_width / 2):
                #                     turnSpd = 5.5 * base_turnSpd * math.exp(2 * error_magnitude)
                #                     # target_turnSpd = 2 * base_turnSpd * math.exp(2.5 * error_magnitude)
                #                     max_turnSpd = 2.0
                #                     turnSpd = turnSpd if error > 0 else -turnSpd
                #                     turnSpd = max(min(turnSpd, max_turnSpd), -max_turnSpd)
                #                     # target_turnSpd = target_turnSpd if error > 0 else -target_turnSpd
                #                     # target_turnSpd = max(min(target_turnSpd, max_turnSpd), -max_turnSpd)
                #                 else:
                #                     turnSpd = 0
                #                 # forSpd = base_forSpd * math.exp(-0.8 * error_magnitude)
                #                 # forSpd = max(forSpd, slow_spd * 0.5)
                #                 target_forSpd = base_forSpd * math.exp(-2 * error_magnitude)
                #                 target_forSpd = max(target_forSpd, slow_spd * 0.5)
                                
                                
                #             elif abs(error) > 120 and abs(error) <= 130:
                #                 if error_magnitude > error_threshold / (img_width / 2):
                #                     turnSpd = 6 * base_turnSpd * math.exp(2 * error_magnitude)
                #                     # target_turnSpd = 3 * base_turnSpd * math.exp(2 * error_magnitude)
                #                     max_turnSpd = 2.0
                #                     turnSpd = turnSpd if error > 0 else -turnSpd
                #                     turnSpd = max(min(turnSpd, max_turnSpd), -max_turnSpd)
                #                     # target_turnSpd = target_turnSpd if error > 0 else -target_turnSpd
                #                     # target_turnSpd = max(min(target_turnSpd, max_turnSpd), -max_turnSpd)
                #                 else:
                #                     turnSpd = 0
                #                 # forSpd = base_forSpd * math.exp(-2 * error_magnitude)
                #                 # forSpd = max(forSpd, slow_spd * 0.5)
                #                 target_forSpd = base_forSpd * math.exp(-2 * error_magnitude)
                #                 target_forSpd = max(target_forSpd, slow_spd * 0.5)
                                
                #             elif abs(error) > 130 and abs(error) <= 140:
                #                 if error_magnitude > error_threshold / (img_width / 2):
                #                     turnSpd = 7 * base_turnSpd * math.exp(2 * error_magnitude)
                #                     # target_turnSpd = 5 * base_turnSpd * math.exp(2 * error_magnitude)
                #                     max_turnSpd = 2.0
                #                     turnSpd = turnSpd if error > 0 else -turnSpd
                #                     turnSpd = max(min(turnSpd, max_turnSpd), -max_turnSpd)
                #                     # target_turnSpd = target_turnSpd if error > 0 else -target_turnSpd
                #                     # target_turnSpd = max(min(target_turnSpd, max_turnSpd), -max_turnSpd)
                                    
                #                 else:
                #                     turnSpd = 0
                #                 # forSpd = base_forSpd * math.exp(-2 * error_magnitude)
                #                 # forSpd = max(forSpd, slow_spd * 0.5)
                #                 target_forSpd = base_forSpd * math.exp(-2 * error_magnitude)
                #                 target_forSpd = max(target_forSpd, slow_spd * 0.5)
                                
                #             else:  # abs(error) > 140
                #                 if error_magnitude > error_threshold / (img_width / 2):
                #                     turnSpd = 8 * base_turnSpd * math.exp(2 * error_magnitude)
                #                     # target_turnSpd = 4 * base_turnSpd * math.exp(2 * error_magnitude)
                #                     max_turnSpd = 2.0
                #                     turnSpd = turnSpd if error > 0 else -turnSpd
                #                     turnSpd = max(min(turnSpd, max_turnSpd), -max_turnSpd)
                #                     # target_turnSpd = target_turnSpd if error > 0 else -target_turnSpd
                #                     # target_turnSpd = max(min(target_turnSpd, max_turnSpd), -max_turnSpd)
                                    
                #                 else:
                #                     turnSpd = 0
                #                 # forSpd = base_forSpd * math.exp(-2 * error_magnitude)
                #                 # forSpd = -max(forSpd, slow_spd * 0.5)
                #                 target_forSpd = base_forSpd * math.exp(-2 * error_magnitude)
                #                 target_forSpd = max(target_forSpd, slow_spd * 0.5)
                                
                # # Apply speed-up logic first, before smoothing
                #     if self.speeding_up and self.lost_line_frames == 0:
                #         forSpd = speed_up  # Directly set to 0.25 during speed-up
                #         turnSpd = 0
                #         self.speed_up_frames += 1
                #         if self.speed_up_frames >= self.speed_up_duration:
                #             self.speeding_up = False
                #             self.speed_up_frames = 0
                #             print("Speed-up phase completed")
                #         self.last_forSpd = forSpd  # Update last_forSpd to avoid abrupt drop after speed-up
                #         print(f"Speeding up: Frame {self.speed_up_frames}/{self.speed_up_duration}, forSpd={forSpd:.2f}, turnSpd={turnSpd:.2f}")
                #         yield forSpd, turnSpd
                #         continue
                    
                #     # Handle crossroad speed
                #     if self.crossroad_count > 0:
                #         target_forSpd = crossroad_forSpd  # 0.1
                #         target_turnSpd = 0  # No turning during crossroad

                #     # Apply smoothing only when not speeding up
                #     if not self.speeding_up:
                #         alpha_forward = 0.9  # Increase alpha for faster response (was 0.1)
                #         alpha_turn = 0.7
                #         forSpd = (1 - alpha_forward) * self.last_forSpd + alpha_forward * target_forSpd
                #         forSpd = max(forSpd, 0.05)  # Set a higher minimum speed (was slow_spd * 0.5 = 0.005)
                        

                    # Movement logic for lineRight
                    if self.crossroad_count == 0 and self.lost_line_frames == 0:
                        error_magnitude = abs(error) / (img_width / 2)
                        if abs(error) <= 20:
                            if error_magnitude > error_threshold / (img_width / 2):  # Error > 5 pixels
                                turnSpd = 0
                                max_turnSpd = 2.0
                                turnSpd = turnSpd if error > 0 else -turnSpd
                                turnSpd = max(min(turnSpd, max_turnSpd), -max_turnSpd)
                            else:
                                turnSpd = 0
                            forSpd = 2 * base_forSpd * math.exp(-2 * error_magnitude)
                            forSpd = max(forSpd, slow_spd * 0.5)
                        elif abs(error) > 20 and abs(error) <= 50:
                            if error_magnitude > error_threshold / (img_width / 2):
                                turnSpd = base_turnSpd * error_magnitude
                                max_turnSpd = 2.0
                                turnSpd = turnSpd if error > 0 else -turnSpd
                                turnSpd = max(min(turnSpd, max_turnSpd), -max_turnSpd)
                            else:
                                turnSpd = 0
                            forSpd = base_forSpd * math.exp(-0.8 * error_magnitude)
                            forSpd = max(forSpd, slow_spd * 0.5)
                        elif abs(error) > 50 and abs(error) <= 90:
                            if error_magnitude > error_threshold / (img_width / 2):
                                turnSpd = 3 * base_turnSpd * math.exp(2 * error_magnitude)
                                max_turnSpd = 2.0
                                turnSpd = turnSpd if error > 0 else -turnSpd
                                turnSpd = max(min(turnSpd, max_turnSpd), -max_turnSpd)
                            else:
                                turnSpd = 0
                            forSpd = base_forSpd * math.exp(-1 * error_magnitude)
                            forSpd = max(forSpd, slow_spd * 0.5)
                        elif abs(error) > 90 and abs(error) <= 120:
                            if error_magnitude > error_threshold / (img_width / 2):
                                turnSpd = 4 * base_turnSpd * math.exp(2 * error_magnitude)
                                max_turnSpd = 2.0
                                turnSpd = turnSpd if error > 0 else -turnSpd
                                turnSpd = max(min(turnSpd, max_turnSpd), -max_turnSpd)
                            else:
                                turnSpd = 0
                            forSpd = base_forSpd * math.exp(-0.8 * error_magnitude)
                            forSpd = max(forSpd, slow_spd * 0.5)
                        elif abs(error) > 120 and abs(error) <= 130:
                            if error_magnitude > error_threshold / (img_width / 2):
                                turnSpd = 5 * base_turnSpd * math.exp(2 * error_magnitude)
                                max_turnSpd = 2.0
                                turnSpd = turnSpd if error > 0 else -turnSpd
                                turnSpd = max(min(turnSpd, max_turnSpd), -max_turnSpd)
                            else:
                                turnSpd = 0
                            forSpd = base_forSpd * math.exp(-2 * error_magnitude)
                            forSpd = max(forSpd, slow_spd * 0.5)
                            
                        elif abs(error) > 130 and abs(error) <= 140:
                                if error_magnitude > error_threshold / (img_width / 2):
                                    turnSpd = 7 * base_turnSpd * math.exp(2 * error_magnitude)
                                    max_turnSpd = 2.0
                                    turnSpd = turnSpd if error > 0 else -turnSpd
                                    turnSpd = max(min(turnSpd, max_turnSpd), -max_turnSpd)
                                else:
                                    turnSpd = 0
                                forSpd = base_forSpd * math.exp(-2 * error_magnitude)
                                forSpd = max(forSpd, slow_spd * 0.5)
                                
                                
                        else:  # abs(error) > 140
                            if error_magnitude > error_threshold / (img_width / 2):
                                turnSpd = 8 * base_turnSpd * math.exp(2 * error_magnitude)
                                max_turnSpd = 2.0
                                turnSpd = turnSpd if error > 0 else -turnSpd
                                turnSpd = max(min(turnSpd, max_turnSpd), -max_turnSpd)
                            else:
                                turnSpd = 0
                            forSpd = base_forSpd * math.exp(-2 * error_magnitude)
                            forSpd = -max(forSpd, slow_spd * 0.5)

                elif lineFollow:
                    # if self.crossroad_count > random_turning_C and not self.turning:
                    #     import random
                    #     self.turn_direction = random.choice([1, -1])
                    #     self.turning = True
                    #     self.turn_frames = 0
                    #     forSpd = 0.0
                    #     turnSpd = self.deg_turn * self.turn_direction
                    #     print(f"Crossroad detected {self.crossroad_count} times, initiating 90-degree turn")
                    #     self.crossroad_count = 0
                    #     continue
                    if (self.crossroad_count > random_turning_C and not self.turning and 
                        self.turn_cooldown == 0):  # Add cooldown check here
                        import random
                        self.turn_direction = random.choice([1, -1])
                        self.turning = True
                        self.turn_frames = 0
                        self.turn_reason = 'junction'  # Track reason for debugging
                        forSpd = 0.0
                        turnSpd = self.deg_turn * self.turn_direction
                        print(f"Crossroad detected {self.crossroad_count} times, initiating 90-degree turn")
                        self.crossroad_count = 0
                        continue
                        
        
                    if self.crossroad_count > 0:
                        col = img_width / 2
                        error = 0
                        print(f"Crossroad in progress (count: {self.crossroad_count}), ignoring junctions")
                        forSpd = crossroad_forSpd
                        # forSpd = self.last_forSpd
                        turnSpd = 0
                    else:
                        if self.left_junction_count > random_turning_J and not self.turning:
                            self.turn_direction = 1
                            self.turning = True
                            self.turn_frames = 0
                            self.turn_reason = 'junction'  # Track reason for debugging
                            forSpd = 0.0
                            turnSpd = self.deg_turn * self.turn_direction
                            print(f"Left_Junction detected {self.left_junction_count} times, initiating 90-degree left turn")
                            self.left_junction_count = 0
                            continue
                        if self.right_junction_count > random_turning_J and not self.turning:
                            self.turn_direction = -1
                            self.turning = True
                            self.turn_frames = 0
                            self.turn_reason = 'junction'  # Track reason for debugging
                            forSpd = 0.0
                            turnSpd = self.deg_turn * self.turn_direction
                            print(f"Right_Junction detected {self.right_junction_count} times, initiating 90-degree right turn")
                            self.right_junction_count = 0
                            continue
                        
                        if prediction == 2:
                            decay_factor = math.exp(-5)
                            forSpd = self.last_forSpd * decay_factor

                        if col is None:
                            self.lost_line_frames += 1
                            if self.lost_line_frames < 60:
                                decay_factor = math.exp(-self.lost_line_frames / 20.0)
                                forSpd = self.last_forSpd * decay_factor
                                turnSpd = self.last_turnSpd * decay_factor*1.2
                                col = self.last_col + (img_width / 2 - self.last_col) * 0.4
                                print(f"Off_Track, adjusting col: {col}, frames lost: {self.lost_line_frames}, forSpd: {forSpd}, turnSpd: {turnSpd}")
                            else:
                                col = self.last_col
                                forSpd = 0
                                turnSpd = 0
                                print("Line lost for too long, stopping")
                        else:
                            self.last_col = col
                            self.lost_line_frames = 0

                        if self.turning:
                            continue

                        if self.speeding_up and self.lost_line_frames == 0:
                            forSpd = speed_up
                            turnSpd = 0
                            self.speed_up_frames += 1
                            if self.speed_up_frames >= self.speed_up_duration:
                                self.speeding_up = False
                                self.speed_up_frames = 0
                                print("Speed-up phase completed")
                            yield forSpd, turnSpd
                            continue

                        if self.crossroad_count == 0 and self.lost_line_frames == 0:
                            error_magnitude = abs(error) / (img_width / 2)
                            if abs(error) <= 20:
                                if error_magnitude > error_threshold / (img_width / 2):
                                    turnSpd = 0
                                    # target_turnSpd = base_turnSpd * error_magnitude  # Fixed bug
                                    max_turnSpd = 2.0
                                    turnSpd = turnSpd if error > 0 else -turnSpd
                                    turnSpd = max(min(turnSpd, max_turnSpd), -max_turnSpd)
                                    # target_turnSpd = target_turnSpd if error > 0 else -target_turnSpd
                                    # target_turnSpd = max(min(target_turnSpd, max_turnSpd), -max_turnSpd)
                                else:
                                    turnSpd = 0
                                # forSpd = 2 * base_forSpd * math.exp(-2 * error_magnitude)
                                # forSpd = max(forSpd, slow_spd * 0.5)
                                target_forSpd = 2*base_forSpd * math.exp(-2 * error_magnitude)
                                target_forSpd = max(target_forSpd, slow_spd * 0.5)
                                
                            elif abs(error) > 20 and abs(error) <= 50:
                                if error_magnitude > error_threshold / (img_width / 2):
                                    turnSpd = base_turnSpd * error_magnitude
                                    # target_turnSpd = base_turnSpd * math.exp(0.5 * error_magnitude)
                                    max_turnSpd = 2.0
                                    turnSpd = turnSpd if error > 0 else -turnSpd
                                    turnSpd = max(min(turnSpd, max_turnSpd), -max_turnSpd)
                                    # target_turnSpd = target_turnSpd if error > 0 else -target_turnSpd
                                    # target_turnSpd = max(min(target_turnSpd, max_turnSpd), -max_turnSpd)
                                else:
                                    turnSpd = 0
                                # forSpd = base_forSpd * math.exp(-0.8 * error_magnitude)
                                # forSpd = max(forSpd, slow_spd * 0.5)
                                target_forSpd = base_forSpd * math.exp(-0.8 * error_magnitude)
                                target_forSpd = max(target_forSpd, slow_spd * 0.5)
                                
                            elif abs(error) > 50 and abs(error) <= 90:
                                if error_magnitude > error_threshold / (img_width / 2):
                                    turnSpd = 3 * base_turnSpd * math.exp(2 * error_magnitude)
                                    # target_turnSpd = 2 * base_turnSpd * math.exp(2.2 * error_magnitude)
                                    max_turnSpd = 2.0
                                    turnSpd = turnSpd if error > 0 else -turnSpd
                                    turnSpd = max(min(turnSpd, max_turnSpd), -max_turnSpd)
                                    # target_turnSpd = target_turnSpd if error > 0 else -target_turnSpd
                                    # target_turnSpd = max(min(target_turnSpd, max_turnSpd), -max_turnSpd)
                                else:
                                    turnSpd = 0
                                # forSpd = base_forSpd * math.exp(-1 * error_magnitude)
                                # forSpd = max(forSpd, slow_spd * 0.5)
                                target_forSpd = base_forSpd * math.exp(-1 * error_magnitude)
                                target_forSpd = max(target_forSpd, slow_spd * 0.5)
                                
                                
                            elif abs(error) > 90 and abs(error) <= 120:
                                if error_magnitude > error_threshold / (img_width / 2):
                                    turnSpd = 5.5 * base_turnSpd * math.exp(2 * error_magnitude)
                                    # target_turnSpd = 2 * base_turnSpd * math.exp(2.5 * error_magnitude)
                                    max_turnSpd = 2.0
                                    turnSpd = turnSpd if error > 0 else -turnSpd
                                    turnSpd = max(min(turnSpd, max_turnSpd), -max_turnSpd)
                                    # target_turnSpd = target_turnSpd if error > 0 else -target_turnSpd
                                    # target_turnSpd = max(min(target_turnSpd, max_turnSpd), -max_turnSpd)
                                else:
                                    turnSpd = 0
                                # forSpd = base_forSpd * math.exp(-0.8 * error_magnitude)
                                # forSpd = max(forSpd, slow_spd * 0.5)
                                target_forSpd = base_forSpd * math.exp(-2 * error_magnitude)
                                target_forSpd = max(target_forSpd, slow_spd * 0.5)
                                
                                
                            elif abs(error) > 120 and abs(error) <= 130:
                                if error_magnitude > error_threshold / (img_width / 2):
                                    turnSpd = 6 * base_turnSpd * math.exp(2 * error_magnitude)
                                    # target_turnSpd = 3 * base_turnSpd * math.exp(2 * error_magnitude)
                                    max_turnSpd = 2.0
                                    turnSpd = turnSpd if error > 0 else -turnSpd
                                    turnSpd = max(min(turnSpd, max_turnSpd), -max_turnSpd)
                                    # target_turnSpd = target_turnSpd if error > 0 else -target_turnSpd
                                    # target_turnSpd = max(min(target_turnSpd, max_turnSpd), -max_turnSpd)
                                else:
                                    turnSpd = 0
                                # forSpd = base_forSpd * math.exp(-2 * error_magnitude)
                                # forSpd = max(forSpd, slow_spd * 0.5)
                                target_forSpd = base_forSpd * math.exp(-2 * error_magnitude)
                                target_forSpd = max(target_forSpd, slow_spd * 0.5)
                                
                            elif abs(error) > 130 and abs(error) <= 140:
                                if error_magnitude > error_threshold / (img_width / 2):
                                    turnSpd = 7 * base_turnSpd * math.exp(2 * error_magnitude)
                                    # target_turnSpd = 5 * base_turnSpd * math.exp(2 * error_magnitude)
                                    max_turnSpd = 2.0
                                    turnSpd = turnSpd if error > 0 else -turnSpd
                                    turnSpd = max(min(turnSpd, max_turnSpd), -max_turnSpd)
                                    # target_turnSpd = target_turnSpd if error > 0 else -target_turnSpd
                                    # target_turnSpd = max(min(target_turnSpd, max_turnSpd), -max_turnSpd)
                                    
                                else:
                                    turnSpd = 0
                                # forSpd = base_forSpd * math.exp(-2 * error_magnitude)
                                # forSpd = max(forSpd, slow_spd * 0.5)
                                target_forSpd = base_forSpd * math.exp(-2 * error_magnitude)
                                target_forSpd = max(target_forSpd, slow_spd * 0.5)
                                
                            else:  # abs(error) > 140
                                if error_magnitude > error_threshold / (img_width / 2):
                                    turnSpd = 8 * base_turnSpd * math.exp(2 * error_magnitude)
                                    # target_turnSpd = 4 * base_turnSpd * math.exp(2 * error_magnitude)
                                    max_turnSpd = 2.0
                                    turnSpd = turnSpd if error > 0 else -turnSpd
                                    turnSpd = max(min(turnSpd, max_turnSpd), -max_turnSpd)
                                    # target_turnSpd = target_turnSpd if error > 0 else -target_turnSpd
                                    # target_turnSpd = max(min(target_turnSpd, max_turnSpd), -max_turnSpd)
                                    
                                else:
                                    turnSpd = 0
                                # forSpd = base_forSpd * math.exp(-2 * error_magnitude)
                                # forSpd = -max(forSpd, slow_spd * 0.5)
                                target_forSpd = base_forSpd * math.exp(-2 * error_magnitude)
                                target_forSpd = max(target_forSpd, slow_spd * 0.5)
                                
                # Apply speed-up logic first, before smoothing
                    if self.speeding_up and self.lost_line_frames == 0:
                        forSpd = speed_up  # Directly set to 0.25 during speed-up
                        turnSpd = 0
                        self.speed_up_frames += 1
                        if self.speed_up_frames >= self.speed_up_duration:
                            self.speeding_up = False
                            self.speed_up_frames = 0
                            print("Speed-up phase completed")
                        self.last_forSpd = forSpd  # Update last_forSpd to avoid abrupt drop after speed-up
                        print(f"Speeding up: Frame {self.speed_up_frames}/{self.speed_up_duration}, forSpd={forSpd:.2f}, turnSpd={turnSpd:.2f}")
                        yield forSpd, turnSpd
                        continue
                    
                    # Handle crossroad speed
                    if self.crossroad_count > 0:
                        target_forSpd = crossroad_forSpd  # 0.1
                        target_turnSpd = 0  # No turning during crossroad

                    # Apply smoothing only when not speeding up
                    if not self.speeding_up:
                        alpha_forward = 0.8   # Increase alpha for faster response (was 0.1)
                        alpha_turn = 0.7
                        forSpd = (1 - alpha_forward) * self.last_forSpd + alpha_forward * target_forSpd
                        forSpd = max(forSpd, 0.05)  # Set a higher minimum speed (was slow_spd * 0.5 = 0.005)
                        # max_turnSpd = 2.0
                        # turnSpd = (1 - alpha_turn) * self.last_turnSpd + alpha_turn * target_turnSpd
                        # turnSpd = max(min(turnSpd, max_turnSpd), -max_turnSpd)  # Keep clamping
                        
                    
                    # print(f"forSpd: {forSpd:.2f}, turnSpd: {turnSpd:.2f}")
                    
                self.last_turnSpd = turnSpd
                self.last_forSpd = forSpd

                print(f"Prediction: {prediction}, Col: {col}, Error: {error:.2f}, forSpd: {forSpd:.2f}, turnSpd: {turnSpd:.2f}")

            yield forSpd, turnSpd






# class VisionSystem:
#     def __init__(self, dataset_path="dataset"):
#         self.dataset_path = dataset_path

#         # Create dataset folder if not exists
#         if not os.path.exists(self.dataset_path):
#             os.makedirs(self.dataset_path)

#         # CSV file to store labels
#         self.csv_file = os.path.join(self.dataset_path, "labels.csv")
#         if not os.path.exists(self.csv_file):
#             with open(self.csv_file, "w", newline="") as file:
#                 writer = csv.writer(file)
#                 writer.writerow(["image_path", "label"])  # CSV Header

#     def collect_and_label_data(self, binary_image, threshold=0.1):
#         """Processes an image and returns a label (1 for line detected, 0 for no line)."""
#         # Convert input to NumPy array
#         if isinstance(binary_image, QBPVision):
#             binary_image = np.asarray(binary_image.imageData, dtype=np.uint8)

#         # Convert to grayscale if needed
#         if len(binary_image.shape) == 3:
#             binary_image = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)

#         total_pixels = binary_image.size  
#         white_pixels = np.count_nonzero(binary_image)
#         white_ratio = float(white_pixels) / float(total_pixels)

#         # Label: 1 = line detected, 0 = no line
#         return 1 if white_ratio > float(threshold) else 0

#     def save_data(self, images):
#         """Saves multiple images and their corresponding labels."""
#         for i, image in enumerate(images):
#             # Collect label for each image
#             label = self.collect_and_label_data(image)

#             # Save image
#             img_filename = os.path.join(self.dataset_path, f"image_{i:04d}.png")
#             cv2.imwrite(img_filename, image)

#             # Store label in CSV
#             with open(self.csv_file, "a", newline="") as file:
#                 writer = csv.writer(file)
#                 writer.writerow([img_filename, label])

import threading
import cv2
import numpy as np
import matplotlib.pyplot as plt

# class LineFollowerPerformance:
#     def __init__(self):
#     # def deviate(self):    
#         self.deviation_list = []
#         self.frame_count = 0
    
#     def measure_deviation(self, binary_image):
#         """
#         Measures deviation from the center of the detected line.
#         """
#         # height, width = binary_image.shape
#         # image_center = width // 2
        
#         width=320
#         image_center = width//2

#         # Find white pixels (where the line is)
#         white_pixels = np.where(binary_image == 255)
#         if len(white_pixels[1]) == 0:
#             deviation = width  # No line detected (max deviation)
#         else:
#             line_center = int(np.mean(white_pixels[1]))
#             deviation = line_center - image_center

#         self.deviation_list.append(deviation)
#         self.frame_count += 1  

#     def plot_deviation(self):
#         """
#         Plots deviation without blocking the main loop.
#         """
#         plt.figure(figsize=(10, 5))
#         plt.plot(range(self.frame_count), self.deviation_list, label="Deviation")
#         plt.axhline(0, color="r", linestyle="--", label="Center Line")
#         plt.xlabel("Frame")
#         plt.ylabel("Deviation from Center (pixels)")
#         plt.title("Line Following Deviation Over Time")
#         plt.legend()
#         plt.show()

import numpy as np
import matplotlib.pyplot as plt
import cv2

class LineFollowerPerformance:
    def __init__(self):
        self.deviation_list = []
        self.frame_count = 0  

    def measure_deviation(self, binary_image, image_center):
        """
        Measures deviation from the center of the detected line.
        """
        self.image_center = image_center
        # width = 320
        # image_center = width // 2
        # tolerance = 16

        # Ensure binary image is in 0 (black) and 255 (white)
        # binary_image = (binary_image * 255).astype(np.uint8) if binary_image.max() == 1 else binary_image

        
        # # Define the tolerance range
        # lower_bound = image_center - tolerance  # 160 - 35 = 125
        # upper_bound = image_center + tolerance  # 160 + 35 = 195

        # # Find white pixels (where the line is)
        # white_pixels = np.where(binary_image == 255)

        # if len(white_pixels[1]) == 0:
        #     deviation = None  # No line detected, ignore this frame
        # else:
        #     line_center = int(np.mean(white_pixels[1]))  # Find center of white pixels
        #     # deviation = line_center - image_center  # Calculate deviation from center
            
        #         # Compute deviation only if outside tolerance
        #     if lower_bound <= line_center <= upper_bound:
        #         deviation = 0  # Inside tolerance = perfectly following the line
        #     else:
        #         deviation = min(abs(line_center - lower_bound), abs(line_center - upper_bound))
        
        
        # Ensure binary image is in 0 (black) and 255 (white)
        # binary_image = (binary_image * 255).astype(np.uint8) if binary_image.max() == 1 else binary_image

        # Find white pixels (where the line is)
        # binary_image = np.where(binary_image < 225, 0, 1)
        # white_pixel_indices = np.where(binary_image == 1, 255, 0)
        white_pixel_indices = np.column_stack(np.where(binary_image == 1))
        # white_pixels = white_pixels_ori[:,:]
        # for row in range(white_pixels.shape[0]):  # Iterate over 320 columns
        #     row_data = white_pixels[row,:]  # Extract column-wise data
        #     # print(f"Column {row}: {row_data}")

        # if len(white_pixels[1]) == 0:
        #     deviation = None  # No line detected, ignore this frame
        # else:
        #     line_center = int(np.mean(white_pixels[1]))  # Find center of white pixels
        #     deviation = line_center - image_center  # Calculate deviation from center
        # # print(line_center)    
        
        
        if white_pixel_indices.size == 0:
            deviation = None  # No white pixels detected, ignore this frame
        else:
            # Compute the average column index of all white pixels
            line_center = int(np.mean(white_pixel_indices[:,1]))  # Column indices (axis 1)
            deviation = line_center #- image_center# - line_center  # Calculate deviation from center
            
        # print(deviation)
        self.deviation_list.append(deviation)
        self.frame_count += 1  
        
    def calculate_accuracy(self):
        """
        Calculates the accuracy as a percentage of frames where the robot stays within ±20 pixels of the center.
        """
        # print(self.deviation_list)
        total_frames = len(self.deviation_list)
        # print(total_frames)
        # Count frames where deviation is within [-20, 20]
        
        correct_frames = sum(1 for deviation in self.deviation_list if deviation is not None)# and (deviation-(160-deviation)) <= deviation <= (deviation+(160-deviation)))
        # print(correct_frames)
        accuracy = (correct_frames / total_frames) * 100 if total_frames > 0 else 0
        # print(f"Accuracy: {accuracy:.2f}%")
        return accuracy


    def plot_deviation(self):
        """
        Plots deviation over time with 160 as the center.
        """
        plt.figure(figsize=(10, 5))

        # # Convert None values to NaN for plotting gaps where no line was detected
        # deviation_array = np.array(self.deviation_list, dtype=float)
        # deviation_array[np.isnan(deviation_array)] = np.nan  # Convert None to NaN
        
        deviation_array = np.array([d if d is not None else np.nan for d in self.deviation_list], dtype=float)


        # Adjust deviation so that 160 is the center
        offset = (self.image_center-deviation_array)/2
        adjusted_deviation = deviation_array + offset
        print(deviation_array)
        print(adjusted_deviation)
        plt.plot(range(self.frame_count), adjusted_deviation, label="Deviation", color="blue")
        plt.axhline(self.image_center, color="r", linestyle="--", label="Center Line of camera")
        

        # plt.plot(range(self.frame_count), adjusted_deviation, label="Deviation", color="blue")
        # plt.axhline(160, color="r", linestyle="--", label="Center Line")  # Centered at 160
        plt.xlabel("Frame")
        plt.ylabel("Deviation from Center (pixels)")
        plt.title("Line Following Deviation Over Time")
        plt.legend()
        plt.savefig("Deviation.png")
        plt.show()

    
    
class QBPRanging():
    def __init__(self):
        pass

    def adjust_and_subsample(self, ranges, angles,end=-1,step=4):

        # correct angles data
        angles_corrected = -1*angles + np.pi/2
        # return every 4th sample
        return ranges[0:end:step], angles_corrected[0:end:step]

    def correct_lidar(self, lidarPosition, ranges, angles):

        # Convert lidar data from polar into cartesian, and add lidar position
        # Then Convert back into polar coordinates

        #-------Replace the following line with your code---------#
        # Determine the start of the focus region 
        ranges_c=None
        angles_c=None
        #---------------------------------------------------------#

        return ranges_c, angles_c

    def detect_obstacle(self, ranges, angles, forSpd, forSpeedGain, turnSpd, turnSpeedGain, minThreshold, obstacleNumPoints):

        halfNumPoints = 205
        quarterNumPoints = round(halfNumPoints/2)

        # Grab the first half of ranges and angles representing 180 degrees
        frontRanges = ranges[0:halfNumPoints]
        frontAngles = angles[0:halfNumPoints]

        # Starting index in top half          1     West
        # Mid point in west quadrant         51     North-west
        # Center index in top half          102     North
        # Mid point in east quadrant     51+102     North-east
        # Ending index in top half          205     East

        ### Section 1 - Dynamic Focus Region ###
        
        #-------Replace the following line with your code---------#
        # Determine the start of the focus region 
        startingIndex = 0
        #---------------------------------------------------------#

        # Setting the upper and lower bound such that the starting index 
        # is always in the first quarant
        if startingIndex < 0:
            startingIndex = 0
        elif startingIndex > 102:
            startingIndex = 102

        # Pick quarterNumPoints in ranges and angles from the front half
        # this will be the region you monitor for obstacles
        monitorRanges = frontRanges[startingIndex:startingIndex+quarterNumPoints]
        monitorAngles = frontAngles[startingIndex:startingIndex+quarterNumPoints]

        ### Section 2 - Dynamic Stop Distance ###

        #-------Replace the following line with your code---------#
        # Determine safetyThreshold based on Forward Speed 
        safetyThreshold = 1
        
        #---------------------------------------------------------#

        
        # At angles corresponding to monitorAngles, pick uniform ranges based on
        # a safety threshold
        safetyAngles = monitorAngles
        safetyRanges = safetyThreshold*monitorRanges/monitorRanges

  
        ### Section 3 - Obstacle Detection ###

        
        #-------Replace the following line with your code---------#
        # Total number of obstacles detected between 
        # minThreshold & safetyThreshold
        # Then determine obstacleFlag based on obstacleNumPoints

        obstacleFlag = 0
        
        #---------------------------------------------------------#


        # Lidar Ranges and Angles for plotting (both scan & safety zone)
        plottingRanges = np.append(monitorRanges, safetyRanges)
        plottingAngles = np.append(monitorAngles, safetyAngles)

        return plottingRanges, plottingAngles, obstacleFlag
