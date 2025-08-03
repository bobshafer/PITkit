import pandas as pd
import matplotlib.pyplot as plt

# Define the column names for the SPARC data files
COLUMN_NAMES = [
    'Radius',      # kiloparsecs (kpc)
    'V_obs',       # Observed velocity (km/s)
    'Err_V_obs',   # Error in V_obs (km/s)
    'V_gas',       # Gas contribution (km/s)
    'V_disk',      # Stellar disk contribution (km/s)
    'V_bulge',     # Bulge contribution (km/s)
    'SB_disk',     # Disk surface brightness
    'SB_bulge'     # Bulge surface brightness
]

# Specify the path to a galaxy data file
galaxy_file = 'data/NGC3198_rotmod.dat'

# Read the data using pandas
# We skip the first 3 lines of comments and use whitespace as the delimiter
galaxy_data = pd.read_csv(
    galaxy_file,
    delim_whitespace=True,
    comment='#',
    header=None,
    names=COLUMN_NAMES
)

# Create a simple plot of the rotation curve
plt.figure(figsize=(10, 6))
plt.errorbar(
    galaxy_data['Radius'],
    galaxy_data['V_obs'],
    yerr=galaxy_data['Err_V_obs'],
    fmt='o',
    capsize=3,
    label='Observed Rotation Curve'
)
plt.xlabel('Radius (kpc)')
plt.ylabel('Velocity (km/s)')
plt.title('Rotation Curve of NGC 3198')
plt.legend()
plt.grid(True)
plt.show()
