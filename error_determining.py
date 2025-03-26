# ANALYSIS CODE - Put inside the line_following.py file
# This code will be used to determine the error in the robot's position

from datetime import datetime
import os

import cv2

from qbot_platform_functions import QBPVision
from pal.products.qbot_platform import QBotPlatformDriver,Keyboard,\
    QBotPlatformCSICamera, QBotPlatformRealSense, QBotPlatformLidar

DATA_DIR = "error_data"
CSV_PATH = os.path.join(DATA_DIR, "error_data.csv")
os.makedirs(DATA_DIR, exist_ok=True)

def add_error_metric(error):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df = pd.DataFrame([[timestamp, error]], columns=["timestamp", "error"])
    df.to_csv(CSV_PATH, mode='a', header=False, index=False)

def determine_error(input):
    # Your code here
    vision = QBPVision()
        
    # Classificaiton
    binary = vision.subselect_and_threshold(input, 99, 101, 225, 255) # Replace input with camera image for camera data
    col, row, area = vision.image_find_objects(binary, 8, 5, 200)
    if col is None:  # Handle the case when no objects are found
        error = 160
    else:
        error = abs(col - 160) # Maximum error = 160
    return error

def get_robot_image():
    vision = QBPVision()
    frameRate, sampleRate = 60.0, 1/60.0
    downCam = QBotPlatformCSICamera(frameRate=frameRate, exposure = 39.0, gain=17.0)
    undistorted = vision.df_camera_undistort(downCam.imageData)
    camera_image = cv2.resize(undistorted, (320, 200))
    return camera_image

def analyse_robot_image(image):
    add_error_metric(determine_error(image))
    
# END OF ANALYSIS - After this error_data.csv will be created with the error data
# This error data can be plotted inside error_plotter.ipynb