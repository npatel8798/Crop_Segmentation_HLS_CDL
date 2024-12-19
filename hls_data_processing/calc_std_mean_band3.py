import rasterio
from glob import glob
import numpy as np
import pandas as pd
import pickle
import os
import multiprocessing as mp
from tqdm import tqdm

def calc_mean_std(i_band, band):
    clip_val = 1.5
    
    chips = glob("/home/npatel23/gokhale_user/Crop Classification Project/Crop Classification Project/HLS_CDL_Data/sample_chips/*_false_color.tif")
    means = []
    stds = []
    
    print("Band:", band)
    vals_all = np.ndarray([0])
    for k in tqdm(range(len(chips)), desc=f"Processing Band {band}"):
        file = chips[k]
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
    i_band = list(range(1, 2))  # Now 1, 2, 3
    band_ids = [3]  # Simplified to just 3 bands Only selecting Band 1
    df = pd.DataFrame({"i_band": i_band, "band": band_ids})
    
    # Process each band using a for loop
    for _, row in tqdm(df.iterrows(), desc="Processing All Bands", total=df.shape[0]):
        calc_mean_std(row['i_band'], row['band'])
    print('Successfully Completed Process')