import numpy as np
# Imports
from pal.products.qbot_platform import QBotPlatformDriver,Keyboard,\
    QBotPlatformCSICamera

class LineFollowingEnv(gym.Env):
    def __init__(self, cameraSrc, C1=1.0, C2=1.0):
        super(LineFollowingEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)  # angular velocity
        self.observation_space = spaces.Box(low=np.array([-1.0, -180.0]), high=np.array([1.0, 180.0]), dtype=np.float32)  # [d, theta]
        
        self.C1 = C1
        self.C2 = C2
        self.dt = 0.05
        self.forward_speed = 0.3
        self.max_steps = 1000
        self.current_step = 0

    def reset(self):
        # Randomly initialize distance and angle
        self.d = np.random.uniform(-1, 1)      # Normalized distance
        self.theta = np.random.uniform(-180, 180)  # Angle in degrees
        
        self.current_step = 0
        return np.array([self.d, self.theta], dtype=np.float32)

    def step(self, action, image):
        self.current_step += 1
        
        # Apply the action (update theta based on the steering angle)
        angular_velocity = action[0]
        delta_theta = angular_velocity * self.dt
        self.theta += np.degrees(delta_theta)
        
        # Simulate the effect on the line position
        self.d += np.sin(np.radians(self.theta)) * 0.05  # Change 0.05 to tune motion sensitivity

        # Calculate reward
        reward = -1 * (self.C1 * (self.d ** 2) + self.C2 * (self.theta ** 2))

        # Check if the episode is done
        done = bool(
            abs(self.d) < 0.05 or self.current_step >= self.max_steps  # Out of bounds or max steps reached
        )

        # Return step information
        obs = np.array([self.d, self.theta], dtype=np.float32)
        return obs, reward, done, {}

    def render(self, mode='console'):
        print(f"Step: {self.current_step}, d: {self.d}, theta: {self.theta}")
        
    def detect_centerline(image):
        """Detects the centerline of a white line in a black background and returns (x, y) points."""
        height, width = image.shape[:2]
        #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Region of interest (ROI) is the top half of the image as the robot only moves forward
        roi = image[:int(height/2)+1, :] 
        
        # Threshold to isolate the white line
        _, binary = cv2.threshold(roi, 200, 255, cv2.THRESH_BINARY)
        line_center = []
        
        # Identify the center of the white line at different heights (every 10 rows)
        for i in range(0, int(height/2) , 10):
        # Find white pixels (line)
            row = binary[i, :]
            white_pixels = np.where(row > 0)
            try:
                center_y = int (np.mean(white_pixels))
            except: 
                continue
            if len(white_pixels) > 0:
                # Compute line center
                line_center.append((i, center_y ))
        return line_center

    def calculate_angle(centerline):
        """Calculates the angle of the white line with respect to the vertical axis."""
        
        if len(centerline) < 2:
            return None  # Not enough points to calculate angle

        # Fit a line using linear regression
        x_coords = np.array([p[0] for p in centerline])
        y_coords = np.array([p[1] for p in centerline])
        
        # Fit a line: y = mx + b
        m, b = np.polyfit(x_coords, y_coords, 1)  # Slope and intercept

        # Calculate angle from vertical axis
        angle_rad = np.arctan(m)
        angle_deg = np.degrees(angle_rad)

        return angle_deg
    
    def set_state(self, d, theta):
        self.d = d
        self.theta = theta