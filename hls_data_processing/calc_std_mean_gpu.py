import rasterio
from glob import glob
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
import torch

import rasterio
from glob import glob
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
import torch

def calc_mean_std(i_band, band, files_to_process):
    clip_val = 1.5
    
    # Define GPU device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Band:", band)
    
    # Initialize an empty tensor on the GPU
    vals_all = torch.empty(0, device=device)
    
    # Process only the files listed in the CSV
    for file in tqdm(files_to_process, desc=f"Processing Band {band}"):
        with rasterio.open(file) as src:
            vals = src.read(int(band)).flatten()  # Ensure band is converted to int
            
            # Convert vals to a PyTorch tensor and move to GPU
            vals_tensor = torch.tensor(vals, dtype=torch.float32, device=device)
            vals_all = torch.cat((vals_all, vals_tensor))  # Concatenate on GPU

    # Compute mean and std on GPU
    band_mean = vals_all.mean().item()
    band_std = vals_all.std().item()

    # Perform clipping on GPU
    vals_all_clipped = torch.clamp(vals_all, 
                                   torch.quantile(vals_all, clip_val / 100),
                                   torch.quantile(vals_all, 1 - clip_val / 100))

    # Compute min and max on GPU
    band_min = vals_all_clipped.min().item()
    band_max = vals_all_clipped.max().item()
    
    # Save results to a file
    with open("band_values_2022_" + str(i_band) + ".txt", "w") as f:
        f.write("Mean: %f\r\n" % (band_mean))
        f.write("Std: %f\r\n" % (band_std))
        f.write("Min: %f\r\n" % (band_min))
        f.write("Max: %f\r\n" % (band_max))
    
    # Save tensor values back to the CPU in a pickle file
    with open("band_values_2022_" + str(i_band) + '.list', 'wb') as file:
        pickle.dump(vals_all.cpu().numpy(), file)

    del vals_tensor, vals_all_clipped, vals_all  # Clear memory

# Load the CSV file with identifiers
chip_csv = '/home/npatel23/gokhale_user/Crop Classification Project/Crop_Segmentation_Project/chip_summary_diff_2022_testing.csv'
chip_df = pd.read_csv(chip_csv)

# Extract all file paths from the folder
all_files = glob("/home/npatel23/gokhale_user/Crop Classification Project/Crop Classification Project/HLS_CDL_Data/filtered_chips/*_false_color.tif")

# Filter files based on the CSV identifiers
filtered_files = [file for file in all_files if any(chip_id in file for chip_id in chip_df['identifier'])]
print(len(filtered_files))
i_band = range(1, 4)  # Bands 1, 2, 3
band_ids = [1, 2, 3]  # Simplified to just 3 bands
df = pd.DataFrame({"i_band": i_band, "band": band_ids})

# Process each band using a for loop
for _, row in tqdm(df.iterrows(), desc="Processing All Bands", total=df.shape[0]):
    calc_mean_std(row['i_band'], row['band'], filtered_files)
