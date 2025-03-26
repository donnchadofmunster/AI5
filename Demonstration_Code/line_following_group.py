#-----------------------------------------------------------------------------#
#------------------Skills Progression 1 - Task Automation---------------------#
#-----------------------------------------------------------------------------#
#----------------------------Lab 3 - Line Following---------------------------#
#-----------------------------------------------------------------------------#

# Imports
from pal.products.qbot_platform import QBotPlatformDriver,Keyboard,\
    QBotPlatformCSICamera, QBotPlatformRealSense, QBotPlatformLidar
from qbot_platform_functions_group import QBPVision
# from qbot_platform_functions_editted import VisionSystem
from qbot_platform_functions_group import LineFollowerPerformance
from quanser.hardware import HILError
from pal.utilities.probe import Probe
from pal.utilities.gamepad import LogitechF710
import time
import numpy as np
import cv2
from qlabs_setup import setup
import csv
import os
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F
from datetime import datetime
import os

import pandas as pd
# import datetime
import cv2

    
class LineFollowerCNN(nn.Module):
    def __init__(self):
        super(LineFollowerCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)  # Add dropout
        self.fc1 = nn.Linear(64 * 6 * 40, 128)  # Adjust based on input size
        self.fc2 = nn.Linear(128, 7)  # 7 classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LineFollowerCNN().to(device)
path_to_state = "line_follower_cnn_improve.pth"  # Update if path differs
model.load_state_dict(torch.load(path_to_state, map_location=device, weights_only=True))
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((50, 320)),  # Match your model's expected input size
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Adjust for 1 channel
])

class_names = [
    "Crossroad",
    "Left_Junction",
    "Off_Track",
    "On_Track",
    "Right_Junction",
    "Slight_Left",
    "Slight_Right"
]


# Section A - Setup
# Initialize time tracking
last_save_time = 0
save_interval = 0.01  # Save 10 frame per second
count = 0

setup(locationQBotP=[-1.35, 0.3, 0.05], rotationQBotP=[0, 0, 0], verbose=True)
time.sleep(2)
ipHost, ipDriver = 'localhost', 'localhost'
commands, arm, noKill = np.zeros((2), dtype = np.float64), 0, True
frameRate, sampleRate = 60.0, 1/60.0
counter, counterDown = 0, 0
endFlag, offset, forSpd, turnSpd = False, 0, 0, 0
startTime = time.time()
def elapsed_time():
    return time.time() - startTime
timeHIL, prevTimeHIL = elapsed_time(), elapsed_time() - 0.017


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
    
    
    
try:
    # Section B - Initialization
    myQBot       = QBotPlatformDriver(mode=1, ip=ipDriver)
    downCam      = QBotPlatformCSICamera(frameRate=frameRate, exposure = 39.0, gain=17.0)
    keyboard     = Keyboard()
    vision       = QBPVision()
    eval_perform = LineFollowerPerformance()
    # vision2      = VisionSystem()
    probe        = Probe(ip = ipHost)
    probe.add_display(imageSize = [200, 320, 1], scaling = True, scalingFactor= 2, name='Raw Image')
    probe.add_display(imageSize = [50, 320, 1], scaling = False, scalingFactor= 2, name='Binary Image')
    # probe.add_display(imageSize = [100, 320, 1], scaling = False, scalingFactor= 2, name='Binary Image_1')
    # line2SpdMap = vision.line_to_speed_map(sampleRate=sampleRate, saturation=75)
    # line2SpdMap = vision.line_to_speed_map(prediction, sampleRate=sampleRate, saturation=75)

    # next(line2SpdMap)
    startTime = time.time()
    time.sleep(0.5)

    # Main loop
    while noKill and not endFlag:
        t = elapsed_time()

        if not probe.connected:
            probe.check_connection()

        if probe.connected:

            # Keyboard Driver
            newkeyboard = keyboard.read()
            if newkeyboard:
                arm = keyboard.k_space
                # lineFollow = keyboard.k_7
                lineFollow = keyboard.k_up
                lineReverse = keyboard.k_down               
                lineLeft = keyboard.k_left         # New Left key
                lineRight = keyboard.k_right        # New Right key
                
                keyboardComand = keyboard.bodyCmd
                if keyboard.k_u:
                    noKill = False
            
            # # Section C - toggle line following
            # if not lineFollow:
            #     commands = np.array([keyboardComand[0], keyboardComand[1]], dtype = np.float64) # robot spd command
            # else:
            #     commands = np.array([forSpd, turnSpd], dtype = np.float64) # robot spd command
                
                
                # Section C - Toggle line following with manual overrides
                
            if lineFollow or lineLeft or lineRight:
                commands = np.array([forSpd, turnSpd], dtype=np.float64)
            else:
                commands = np.array([keyboardComand[0], keyboardComand[1]], dtype=np.float64)
                
                if lineReverse:  # Reverse
                    commands = np.array([-0.2, 0.0], dtype=np.float64)  # Reverse
                    
            # if not lineFollow:
            #     # Manual control using keyboard
            #     commands = np.array([keyboardComand[0], keyboardComand[1]], dtype=np.float64)
            #     # if lineLeft:  # Manual sharp left turn
            #     #     commands = np.array([0.1, 0.2], dtype=np.float64)  # Forward + left turn
            #     # elif lineRight:  # Manual sharp right turn
            #     #     commands = np.array([0.1, -0.2], dtype=np.float64)  # Forward + right turn
            #     if lineReverse:  # Reverse
            #         commands = np.array([-0.2, 0.0], dtype=np.float64)  # Reverse
                
            #     # elif class_names[prediction] in ["Crossroad", "Left_Junction", "Right_Junction"]:
            #     # elif lineLeft:
            #     #     commands = np.array([0.1, 0.2], dtype=np.float64)  # Manual left
            #     # elif lineRight:
            #     #     commands = np.array([0.1, -0.2], dtype=np.float64)   # Manual right

            # else:
            #     # Autonomous line following with intersection handling
            #     commands = np.array([forSpd, turnSpd], dtype=np.float64)
                # # Hybrid approach
               
                # if class_names[prediction] == "Crossroad":
                #     if lineLeft:
                #         commands = np.array([0.1, -0.2], dtype=np.float64)  # Manual left
                #     elif lineRight:
                #         commands = np.array([0.1, 0.2], dtype=np.float64)   # Manual right
                # elif class_names[prediction] == "Left_Junction":
                #     if lineLeft:
                #         commands = np.array([0.1, -0.2], dtype=np.float64)  # Manual override
                #     else:
                #         commands = np.array([0.1, -0.2], dtype=np.float64)  # Auto left
                # elif class_names[prediction] == "Right_Junction":
                #     if lineRight:
                #         commands = np.array([0.1, 0.2], dtype=np.float64)   # Manual override
                #     else:
                #         commands = np.array([0.1, 0.2], dtype=np.float64)   # Auto right
                # else:
                #     commands = np.array([forSpd, turnSpd], dtype=np.float64)  # Default line following
            # QBot Hardware
            newHIL = myQBot.read_write_std(timestamp = time.time() - startTime,
                                            arm = arm,
                                            commands = commands)
            if newHIL:
                timeHIL = time.time()
                newDownCam = downCam.read()
                if newDownCam:
                    counterDown += 1

                    # Section D - Image processing 
                    
                    # Section D.1 - Undistort and resize the image
                    undistorted = vision.df_camera_undistort(downCam.imageData)
                    gray_sm = cv2.resize(undistorted, (320, 200))
                    analyse_robot_image(gray_sm)
                    #-------Replace the following line with your code---------#
                    binary = vision.subselect_and_threshold(gray_sm, rowStart=50,rowEnd=100,minThreshold=225,maxThreshold=255)
                    # binary_1 = vision.subselect_and_threshold(gray_sm, rowStart=0,rowEnd=100,minThreshold=225,maxThreshold=255)

                    # # # Ensure binary is a NumPy array
                    # if isinstance(binary, QBPVision):
                    #     binary = np.asarray(binary.imageData, dtype=np.uint8)

                    # print("Binary Type:", type(binary))  
                    # print("Binary Shape:", binary.shape)
                    # print([binary])
                    # training_data= 100
                    # label = vision.collect_and_label_data(binary, training_data)
                    # label = vision.collect_and_label_data(binary)
                    # image_list = []
                    # label_list = []
                    storage_path = r"Line_detection_Group"    
                    data_path = r"Data_storage.csv"
                    if timeHIL - last_save_time >= save_interval:
                        # image_list.append(binary)
                        labelled_data = vision.collect_and_label_data(binary, count, storage_path, data_path)
                        last_save_time = timeHIL # Reset timer
                        count += 1

                    eval_perform.measure_deviation(labelled_data, image_center=160)
                    eval_perform.calculate_accuracy()
                    
                    # Blob Detection via Connected Component Labeling
                    # # col, row, area = 0, 0, 0
                    # col, row, area = vision.image_find_objects(binary, connectivity=8, minArea=500, maxArea=2000)

                    # Section D.2 - Speed command from blob information
                    # forSpd, turnSpd = line2SpdMap.send((col, 1, 8))
                    # forSpd, turnSpd = line2SpdMap.send((col, 3, 5))
                    
                    
                    
                    ##
                    # # Red point in the interception
                    # forSpd, turnSpd = line2SpdMap.send((col,0, 0)) << waiting for the next command either left or right or continue to move forward
                    #     >> in the waiting press space + 6 or 8, while pressing then turn 90 deg then stop when it reach the center of line then move forward
                    # Red point in the interception
                    # forSpd, turnSpd = line2SpdMap.send((col,0, 0)) << waiting for the next command either left or right or continue to move forward
                    # Convert binary image to PIL Image, then apply transforms
                    # CNN Prediction
                    binary_pil = Image.fromarray(binary)
                    binary_tensor = transform(binary_pil).unsqueeze(0).to(device)
                    # print("Binary shape :", binary.shape)
                    # print("Binary_pil shape :", binary_pil.shape)
                    # print("Binary_tensor shape :", binary_tensor.shape)
                    with torch.no_grad():
                        output = model(binary_tensor)
                        prediction = torch.argmax(output, dim=1).item()
                        print(class_names[prediction])
                    # print("Output shape :", output.shape)
                    # print("Prediction shape :", prediction.shape)   
                    
                    col, row, area = vision.image_find_objects(binary, connectivity=8, minArea=100, maxArea=5000)
                   
                    # line2SpdMap = vision.line_to_speed_map(prediction, sampleRate=sampleRate, saturation=75)
                    line2SpdMap = vision.line_to_speed_map(prediction, sampleRate=sampleRate, saturation=75, lineFollow=lineFollow,
                                                          lineLeft=lineLeft, lineRight=lineRight)
                    next(line2SpdMap)
                    forSpd, turnSpd = line2SpdMap.send((prediction, col))
                    # print(forSpd)
                    # print(turnSpd)
                    # # Get speeds with detailed debugging
                    # try:
                    #     print("Sending to generator...")
                    #     result = line2SpdMap.send(prediction)
                    #     print(f"Generator result: {result}")
                    #     if result is None:
                    #         print("Error: Generator returned None")
                    #         break
                    #     forSpd, turnSpd = result
                    #     print(f"Speeds: {forSpd}, {turnSpd}")
                    # except StopIteration:
                    #     print("Generator stopped unexpectedly")
                    #     break
                    # except Exception as e:
                    #     print(f"Generator error: {e}")
                    #     break
                    # print(class_names[prediction])
                    if lineFollow:
                        commands = np.array([forSpd, turnSpd], dtype=np.float64) 
                    elif lineLeft:  
                        commands = np.array([forSpd, turnSpd], dtype=np.float64)   # Forward + left turn
                    elif lineRight:
                        commands = np.array([forSpd, turnSpd], dtype=np.float64)   # Forward + right turn 
                    # print(forSpd)
                    # print(turnSpd)
                    # print("Condition triggered: Heavy_Left, turnSpd =", turnSpd)
                    #---------------------------------------------------------#

                if counterDown%4 == 0:
                    sending = probe.send(name='Raw Image', imageData=gray_sm)
                    sending = probe.send(name='Binary Image', imageData=binary)
                    # sending = probe.send(name='Binary Image_1', imageData=binary_1)
                prevTimeHIL = timeHIL

except KeyboardInterrupt:
    print('User interrupted.')
except HILError as h:
    print(h.get_error_message())
finally:
    downCam.terminate()
    myQBot.terminate()
    probe.terminate()
    keyboard.terminate()
    # vision.plot_deviation()
    # eval_perform.plot_deviation()
    image_height = 50
    image_width = 320
    # csv_path = os.path.join(storage_path, data_path)  # Correct path joining
    #
    # output_folder = "/Full_Extracted_images"
    # # labels = ["On_Track", "Left", "Right", "Off_Track"]
    # labels = ["On_Track", "Slight_Left", "Slight_Right", "Heavy_Left", "Heavy_Right", "Off_Track", "Crossroad", "Left_Junction", "Right_Junction"]
    # training_path = r"Training_data.csv"
    # vision.save_images_from_csv(storage_path, data_path, output_folder, labels, training_path, image_height, image_width)