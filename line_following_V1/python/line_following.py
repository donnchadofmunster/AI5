#-----------------------------------------------------------------------------#
#------------------Skills Progression 1 - Task Automation---------------------#
#-----------------------------------------------------------------------------#
#----------------------------Lab 3 - Line Following---------------------------#
#-----------------------------------------------------------------------------#

# Imports
from pal.products.qbot_platform import QBotPlatformDriver,Keyboard,\
    QBotPlatformCSICamera, QBotPlatformRealSense, QBotPlatformLidar
from hal.content.qbot_platform_functions import QBPVision,QBPDataCollector
from quanser.hardware import HILError
from pal.utilities.probe import Probe
from pal.utilities.gamepad import LogitechF710
import time
import numpy as np
import cv2
import torch
from qlabs_setup import setup
# ✅ Load the Trained CNN Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LineFollowerCNN(torch.nn.Module):
    def __init__(self):
        super(LineFollowerCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = torch.nn.Linear(32 * 32 * 32, 128)
        self.fc2 = torch.nn.Linear(128, 4)  # 4 steering outputs

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)  # No activation, CrossEntropyLoss applies softmax
        return x

# ✅ Initialize and Load Model
model = LineFollowerCNN().to(device)
model.load_state_dict(torch.load("line_follower_cnn.pth", map_location=device))
model.eval()
print("✅ CNN Model Loaded!")

# Section A - Setup
# Change the robot initial position and orientation to test it in the outer line of the track
setup(locationQBotP=[-1.508, -0.41, 0.003], rotationQBotP=[0, 0, 90], verbose=True)
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
    probe        = Probe(ip = ipHost)
    probe.add_display(imageSize = [200, 320, 1], scaling = True, scalingFactor= 2, name='Raw Image')
    probe.add_display(imageSize = [50, 320, 1], scaling = False, scalingFactor= 2, name='Binary Image')
    line2SpdMap = vision.line_to_speed_map(sampleRate=sampleRate, saturation=75)
    next(line2SpdMap)
    startTime = time.time()
    time.sleep(0.5)

    # Initialize the data collector
    data_collector = QBPDataCollector(output_dir="training_data")
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
                    # Subselect a part of the image and perform thresholding
                    binary = vision.subselect_and_threshold(gray_sm, 50, 100, 225, 255)
                    
                    # Blob Detection via Connected Component Labeling
                    col, row, area = vision.image_find_objects(binary, 8, 500, 2000)
                    # Section D.2 - Speed command from blob information
                    forSpd, turnSpd = line2SpdMap.send(gray_sm)  # Send the grayscale image to CNN
                    #---------------------------------------------------------#

                if counterDown%4 == 0:
                    label = {'line_detected': col is not None, 'centroid': (col, row) if col else None}
                    sending = probe.send(name='Raw Image', imageData=gray_sm)
                    sending = probe.send(name='Binary Image', imageData=binary)
                    data_collector.collect_and_label_data(gray_sm, label)
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