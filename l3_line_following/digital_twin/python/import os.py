import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

# ✅ Update path
CSV_FILE = r"C:\Users\abdal\Documents\Quanser\Mobile_Robotics_Student\src\Mobile_Robotics\sp1_task_automation\l3_line_following\digital_twin\python\training_data\training_data.csv"

# ✅ Load Data
def load_data(csv_file):
    if not os.path.exists(csv_file):
        print(f"❌ CSV file not found at: {csv_file}")
        return None

    df = pd.read_csv(csv_file)
    
    if df.empty:
        print("❌ No data found in CSV.")
        return None

    return df

# ✅ Compute MAE and Accuracy
def compute_performance_metrics(centroids_x):
    if centroids_x.size == 0:  # Fix ValueError issue
        print("❌ No valid centroid data found.")
        return None, None

    ground_truth_center = 160  # Image center
    errors = np.abs(centroids_x - ground_truth_center)

    mae = np.mean(errors)  # Mean Absolute Error
    accuracy = 100 - (mae / ground_truth_center * 100)

    return mae, accuracy

# ✅ Plot Function
def plot_centroids(csv_file):
    df = load_data(csv_file)
    if df is None:
        return
    
    frame_numbers = np.arange(len(df))
    centroids_x = df["centroid_x"].to_numpy()

    # ✅ Handle missing data (NaN handling)
    centroids_x = np.nan_to_num(centroids_x, nan=160)

    # ✅ Apply minor smoothing
    centroids_x_smooth = uniform_filter1d(centroids_x, size=5, mode='nearest')

    # ✅ Compute Performance Metrics
    mae, accuracy = compute_performance_metrics(centroids_x_smooth)
    
    # ✅ Plot
    plt.figure(figsize=(10, 5))
    plt.plot(frame_numbers, centroids_x_smooth, 'bo-', label="Centroid X Position (Smoothed)", alpha=0.7)
    plt.axhline(y=160, color='r', linestyle='--', label="Image Center (160px)")
    
    plt.xlabel("Frame Number")
    plt.ylabel("Centroid X Position")
    plt.title(f"Line Following Centroid Tracking Over Time\nMAE: {mae:.2f} pixels, Accuracy: {accuracy:.2f}%")
    plt.legend()
    plt.grid()
    plt.show()

# ✅ Run
plot_centroids(CSV_FILE)
