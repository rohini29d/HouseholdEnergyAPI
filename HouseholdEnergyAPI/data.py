import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Start time
start_time = datetime(2007, 1, 1, 0, 0)

# Generate 5,000 timestamps at 1-minute intervals
timestamps = [start_time + timedelta(minutes=i) for i in range(5000)]
dates = [dt.strftime('%d-%m-%Y') for dt in timestamps]
times = [dt.strftime('%I:%M-%S %p') for dt in timestamps]

# Set seed for reproducibility
np.random.seed(42)

# Generate synthetic energy data
global_active_power = np.random.normal(loc=2.5, scale=0.1, size=5000).round(3)
global_reactive_power = np.random.normal(loc=0.1, scale=0.02, size=5000).round(3)
voltage = np.random.normal(loc=241.5, scale=0.5, size=5000).round(2)
global_intensity = (global_active_power * 1000 / voltage).round(1)
sub_metering_1 = np.random.randint(0, 2, size=5000)
sub_metering_2 = np.random.randint(0, 2, size=5000)
sub_metering_3 = np.random.randint(0, 2, size=5000)

# Combine into a DataFrame
data = {
    "Date": dates,
    "Time": times,
    "Global_active_power": global_active_power,
    "Global_reactive_power": global_reactive_power,
    "Voltage": voltage,
    "Global_intensity": global_intensity,
    "Sub_metering_1": sub_metering_1,
    "Sub_metering_2": sub_metering_2,
    "Sub_metering_3": sub_metering_3,
}

df = pd.DataFrame(data)

# Save to CSV
df.to_csv("energy_data.csv", index=False)
