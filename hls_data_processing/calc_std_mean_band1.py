import rasterio
from glob import glob
import numpy as np
import pandas as pd
import pickle
import os
import multiprocessing as mp
from tqdm import tqdm

def calc_mean_std(i_band, band, files_to_process):
    clip_val = 1.5
    
    chips = glob("/home/npatel23/gokhale_user/Crop Classification Project/Crop Classification Project/HLS_CDL_Data/sample_chips/*_false_color.tif")
    means = []
    stds = []
    
    print("Band:", band)
    vals_all = np.ndarray([0])
    for file in tqdm(files_to_process, desc=f"Processing Band {band}"):
        # file = chips[k]
        with rasterio.open(file) as src:
            vals = src.read(int(band)).flatten()  # Ensure band is converted to int
            vals_all = np.concatenate([vals_all, vals])
    band_mean = vals_all.mean()
    band_std = vals_all.std()
    vals_all_clipped = np.clip(vals_all, np.nanpercentile(vals_all, clip_val), 
                               np.nanpercentile(vals_all, 100 - clip_val))
    band_min = np.min(vals_all_clipped)
    band_max = np.max(vals_all_clipped)
    f = open("band_values_" + str(i_band) + ".txt","w+")
    f.write("Mean: %f\r\n" % (band_mean))
    f.write("Std: %f\r\n" % (band_std))
    f.write("Min: %f\r\n" % (band_min))
    f.write("Max: %f\r\n" % (band_max))
    f.close()
    
    with open("band_values_" + str(i_band) + '.list', 'wb') as file:
        pickle.dump(vals_all, file)

if __name__ == "__main__":
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
    print('Successfully Completed Process')