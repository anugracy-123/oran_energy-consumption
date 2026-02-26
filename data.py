import pandas as pd
import numpy as np

rows = []

np.random.seed(42)

for t in range(3200):              # time steps
    for cell in range(13):         # H3 cells
        user_count = np.random.randint(1, 25)
        avg_sinr = np.random.uniform(10, 35)
        rx_power = np.random.uniform(-90, -60)
        bs_id = np.random.randint(1, 6)

        # ORAN-aware power mode selection
        if user_count < 5:
            power_mode = "PS"
            power_factor = 0.8
        elif user_count < 15:
            power_mode = "PM"
            power_factor = 0.95
        else:
            power_mode = "PF"
            power_factor = 1.1

        # Throughput calculation (enhanced network scenario)
        avg_throughput = (
            power_factor *
            np.log2(1 + avg_sinr) *
            np.random.uniform(6, 9)
        )

        rows.append([
            t,
            f"h3_{cell}",
            user_count,
            round(avg_sinr, 2),
            round(rx_power, 2),
            bs_id,
            power_mode,
            round(avg_throughput, 2)
        ])

df = pd.DataFrame(rows, columns=[
    "time_step",
    "h3_cell",
    "user_count",
    "avg_sinr",
    "rx_power",
    "bs_id",
    "power_mode",
    "avg_throughput"
])

df.to_csv("throughput_prediction_dataset.csv", index=False)

print("Dataset created:", df.shape)