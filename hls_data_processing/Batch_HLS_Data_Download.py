import geopandas
import json
import xarray
import rasterio
import rioxarray
import os
import shlex
import fiona
import urllib.request as urlreq
import pandas as pd
import numpy as np
import requests
import xmltodict
import shutil
import datetime
import boto3
import pyproj
import multiprocessing as mp
from shapely.geometry import shape, Polygon, box

from pystac_client import Client 
from collections import defaultdict
from glob import glob
from rasterio.enums import Resampling
import matplotlib.pyplot as plt
from subprocess import Popen, PIPE
from tqdm import tqdm
from netrc import netrc
from platform import system
from getpass import getpass
from rasterio.session import AWSSession
from pathlib import Path
import earthaccess

import json
import shutil
from pathlib import Path
import shlex
import subprocess


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


cloud_thres = 25 # percent cloud cover for tile level query

root_path = "/home/npatel23/gokhale_user/Crop Classification Project/"
req_path = "/home/npatel23/gokhale_user/Crop Classification Project/HLS_CDL_Data/"
extra_files = "/home/npatel23/gokhale_user/Crop Classification Project/"

chip_dir = 'Crop Classification Project/HLS_CDL_Data/chips/'
tile_dir = 'Crop Classification Project/HLS_CDL_Data/tiles'
chip_dir_filt = 'Crop Classification Project/HLS_CDL_Data/chips_filtered'
chip_fmask_dir = 'Crop Classification Project/HLS_CDL_Data/chips_fmask'

chip_file =  root_path + "chip_bbox_task_1.geojson"
chipping_json = root_path + "chip_bbox_task_1_5070.geojson"
chip_csv = root_path + "chip_tracker.csv"
kml_file = extra_files + 'sentinel_tile_grid.kml'
# tile_tracker_csv = req_path + "tile_tracker.csv"
cdl_file = extra_files + "/2023_30m_cdls/2023_30m_cdls.tif"
cdl_reclass_csv = root_path + "cdl_total_dst.csv"

earthaccess.login(persist=True)

selected_tiles = pd.read_csv("selected_tiles_2022.csv")

print(selected_tiles.head())

import shutil

def tile_download(table, from_csv=True):
    """
    Downloading tiles by reading from the metadata information gathered earlier

    Args:
        table: A pandas dataframe that generated previously
        from_csv: If the tile information is from a csv, then True
    """
    info_list = []
    bands = ["B02", "B04", "B8A", "Fmask"]
    accept_tiles = np.unique(table.tile_id)
    for tile in tqdm(accept_tiles):
        temp_tb = table[table.tile_id == tile]
        for i, j in temp_tb.iterrows():
            if from_csv:
                bands_dict = json.loads(j.http_links.replace("'", '"'))
            else:
                bands_dict = j.s3_links
            for band in bands:
                temp_key = bands_dict[band]
                os.chdir('/umbc/rs/gokhale/users/npatel23/Crop Classification Project/HLS_CDL_Data/tiles/')
                temp_sav_path = Path(f"{bands_dict[band].split('/')[4]}/2022/{os.path.basename(temp_key)}")

                # Create directory if it doesn't exist
                temp_sav_path.parent.mkdir(parents=True, exist_ok=True)

                # Check if the file already exists
                if not temp_sav_path.exists():
                    command = f"wget -N -q --read-timeout=5 --tries=0 -O {shlex.quote(str(temp_sav_path))} {shlex.quote(temp_key)}"
                    result = subprocess.run(command, shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    
                    if result.returncode != 0:
                        print(f"Download failed for: {temp_key} with error {result.stderr}")
                    else:
                        print('File Downloaded Successfully')
                else:
                    print(f"File already exists: {temp_sav_path}, skipping download.")

            temp_dict = {"tile":tile, "timestep":i, "date":j.date, "save_path":f"{bands_dict[band].split('/')[4]}/2022/", "filename":bands_dict["B02"].split('/')[6].replace(".B02.tif","")}
            info_list.append(temp_dict)

    return pd.DataFrame(info_list)


track_df = tile_download(selected_tiles, from_csv=True)

track_df.to_csv(req_path + "track_df_2022.csv", index=False)
