import pandas as pd
import numpy as np

# Settings for 4000+ rows
rows = 4500
np.random.seed(42)

# Generate Time steps (e.g., 1kHz sampling)
time = np.linspace(0, 4.5, rows)

# Simulate Normal Sine Waves with added Fault Transients
def generate_signal(freq, t, noise_lvl=0.05, fault_drop=1.0):
    return (fault_drop * np.sin(2 * np.pi * freq * t)) + np.random.normal(0, noise_lvl, len(t))

# Feature Engineering
data = {
    'Time': time,
    'Va': generate_signal(50, time, fault_drop=0.2), # Dropped voltage for fault
    'Vb': generate_signal(50, time + 120, fault_drop=1.0),
    'Vc': generate_signal(50, time + 240, fault_drop=1.0),
    'Ia': generate_signal(50, time, noise_lvl=0.1, fault_drop=15.0), # High current for fault
    'Ib': generate_signal(50, time + 120, noise_lvl=0.1, fault_drop=1.2),
    'Ic': generate_signal(50, time + 240, noise_lvl=0.1, fault_drop=1.2),
    # Labels: 0=Normal, 1=AG, 2=BG, 3=CG, 4=AB, 5=BC, 6=CA, 7=ABC
    'Fault_Type': np.random.choice([0, 1, 2, 3, 4, 5, 6, 7], rows),
    # Labels: 0=No Fault, 1=Zone 1, 2=Zone 2, 3=Zone 3
    'Zone': np.random.choice([0, 1, 2, 3], rows)
}

df = pd.DataFrame(data)

# Save to CSV
file_name = "TEDL_Protection_Dataset.csv"
df.to_csv(file_name, index=False)
print(f"Dataset generated successfully: {file_name} with {len(df)} rows.")