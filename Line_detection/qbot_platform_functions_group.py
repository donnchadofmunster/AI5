import numpy as np
import cv2
import csv
import os
import time

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
        
    def save_images_from_csv(self, storage, data, out_dir, labels, training, img_height, img_width):
        out_dir = storage + "/Full_Extracted_images"
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
 

        # with open(csv_training_file, mode='w', newline='') as csvfile:
        #         csv_writer = csv.writer(csvfile)
        #         # Write headers
        #         csv_writer.writerow(["Image Path", "label, Error"])
                    
        # Load the CSV file
        with open(csv_data_file, "r") as file:
            lines = file.readlines()
        
        images = []
        current_image = []

        # for line in lines:
        #     line = line.strip()
        #     if not line:  # Skip empty lines
        #         continue

        #     # Convert the row into a list of integers, handling potential non-numeric values
        #     row_data = [int(val) if val.strip().isdigit() else 0 for val in line.split(",")]

        #     if len(row_data) == img_width:
        #         current_image.append(row_data)

        #     # If we have collected exactly 50 rows, store the image and reset
        #     if len(current_image) == img_height:
        #         images.append(np.array(current_image, dtype=int))
        #         current_image = []  # Reset for the next image

        # # Save each extracted image
        # for idx, binary_image in enumerate(images):
        #     if binary_image.shape == (img_height, img_width):  # Ensure correct size
        #         # Convert 0/1 binary data to 8-bit grayscale (0 → black, 1 → white)
        #         binary_image = (binary_image * 255).astype(np.uint8)

        #         # Save the image
        #         img_path = os.path.join(out_dir, f"image_{idx+1}.png")
        #         cv2.imwrite(img_path, binary_image)
        #         print(f"Saved: {img_path}")
        #     else:
        #         print(f"Skipping invalid shape {binary_image.shape} at index {idx+1}")
                
                
        labels_dict = {"On_Track": [], "Left": [], "Right": [], "Off_Track": []}

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
            # Assuming that the column at index 160 corresponds to the center of the line
            error = None
            label = "off_line"

            # Find the center column (e.g., 160) in each image and determine label based on its value
            col = None
            for i in range(img_width):
                if binary_image[img_height // 2, i] == 1:  # Check center row for active pixels (line)
                    col = i
                    break
                
            Margin = 20
                        
            if col is None:
                error = None
                label = "Off_Track"
            else:
                error = col - 160  # Calculate the error from the center column

                if -Margin <= error <= Margin:  # Straight Line
                    label = "On_Track"
                elif -200 < error < -Margin:  # Left Turn
                    label = "Left"
                elif 200 > error > Margin:  # Right Turn
                    label = "Right"

            
            
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

    def line_to_speed_map(self, sampleRate, saturation):

        integrator   = Calculus().integrator(dt = sampleRate, saturation=saturation)
        derivative   = Calculus().differentiator(dt = sampleRate)
        next(integrator)
        next(derivative)
        forSpd, turnSpd = 0, 0
        offset = 0

        img_width = 320  # Image width in pixels
        img_bottom_y = 200  # Bottom of the image (assuming height is 200px)
        focal_length = 55      # Estimated focal length for perspective (adjust if needed)

        while True:
            col, kP, kD = yield forSpd, turnSpd

            if col is not None:
                #-----------Complete the following lines of code--------------#
                # error = 0
                error = ((img_width/2) - col)
                # angle = np.arctan2(0, 0)
                
                 # Compute the error angle using arctan2
                angle = np.arctan2(error, (focal_length))  

                # turnSpd = 0 * angle + 0 * derivative.send(angle)
                
                turnSpd = kP * angle + kD * derivative.send(angle)
                # forSpd = 0
                # forSpd = kP * np.cos(angle)
                # Adjust forward speed based on error (angle)
                forSpd = max(min(1.0, np.cos(angle) * 0.3), 0.1)  # Forward speed based on alignment

            # else:
            #     forSpd = 0
            #     turnSpd = 0

                #-------------------------------------------------------------#
                offset = integrator.send(25*turnSpd)
                error += offset



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
