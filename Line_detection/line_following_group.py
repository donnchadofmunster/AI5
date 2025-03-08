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

# Section A - Setup
# Initialize time tracking
last_save_time = 0
save_interval = 0.1  # Save 10 frame per second
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
    line2SpdMap = vision.line_to_speed_map(sampleRate=sampleRate, saturation=75)
    next(line2SpdMap)
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
                lineFollow = keyboard.k_7
                keyboardComand = keyboard.bodyCmd
                if keyboard.k_u:
                    noKill = False
            
            # Section C - toggle line following
            if not lineFollow:
                commands = np.array([keyboardComand[0], keyboardComand[1]], dtype = np.float64) # robot spd command
            else:
                commands = np.array([forSpd, turnSpd], dtype = np.float64) # robot spd command

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
                    
                    #-------Replace the following line with your code---------#
                    binary = vision.subselect_and_threshold(gray_sm, rowStart=50,rowEnd=100,minThreshold=225,maxThreshold=255)
                    
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
                    # col, row, area = 0, 0, 0
                    col, row, area = vision.image_find_objects(binary, connectivity=8, minArea=500, maxArea=2000)

                    # Section D.2 - Speed command from blob information
                    forSpd, turnSpd = line2SpdMap.send((col, 1, 8))
                    #---------------------------------------------------------#

                if counterDown%4 == 0:
                    sending = probe.send(name='Raw Image', imageData=gray_sm)
                    sending = probe.send(name='Binary Image', imageData=binary)
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
    output_folder = "/Full_Extracted_images"
    labels = ["On_Track", "Left", "Right", "Off_Track"]
    training_path = r"Training_data.csv"
    vision.save_images_from_csv(storage_path, data_path, output_folder, labels, training_path, image_height, image_width)