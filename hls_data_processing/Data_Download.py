import geopandas
import sys
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
import shlex
import subprocess
import multiprocessing as mp
import logging
from shapely.geometry import shape, Polygon, box


from pystac_client import Client 
from collections import defaultdict
from glob import glob
from rasterio.enums import Resampling
from rasterio.mask import mask
import matplotlib.pyplot as plt
from subprocess import Popen, PIPE
from tqdm import tqdm
from netrc import netrc
from platform import system
from getpass import getpass
from rasterio.session import AWSSession
from pathlib import Path
import earthaccess

def find_tile(tile_x, x, tile_y, y, tile_name):
    """
    Identify closest tile
    """
    
    s = (tile_x - x)**2+(tile_y - y)**2
    tname = tile_name[np.argmin(s)]
    return(tname)

def spatial_filtering (dataframe):
    """
        Using spatial coverage percentage to filter chips

        Args:
            dataframe: A pandas dataframe that generated previously
    """
    cover_list = [100, 90, 80, 70, 60, 50]
    tile_list_ft = []
    tile_list = dataframe.tile_id.unique().tolist()
    
    for tile in tqdm(tile_list):
        temp_df = dataframe[dataframe.tile_id == tile]
        for cover_pct in cover_list:
            
            temp_df_filtered = temp_df[temp_df.spatial_cover >= cover_pct]
            if len(temp_df_filtered) >= 3:
                for i in range(len(temp_df_filtered)):
                    tile_list_ft.append(temp_df_filtered.iloc[i])
                break
    
    tile_df_filtered = pd.DataFrame(tile_list_ft)
    return tile_df_filtered

def tile_download(table, tile_dir, from_csv=True):
    """
    Downloading tiles by reading from the metadata information gathered earlier

    Args:
        table: A pandas dataframe that generated previously
        from_csv: If the tile information is from a csv, then True
    """
    logging.info('Starting Tile Downloading Function')
    info_list = []
    bands = ["LB02", "LB03", "LB04", "LB05", "LB06", "Fmask"]
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
                # os.chdir('/umbc/rs/gokhale/users/npatel23/Crop Classification Project/HLS_CDL_Data/tiles/')
                temp_sav_path = Path(tile_dir + os.path.basename(temp_key))

                # Create directory if it doesn't exist
                temp_sav_path.parent.mkdir(parents=True, exist_ok=True)

                # Check if the file already exists
                if not temp_sav_path.exists():
                    command = f"wget -N -q --read-timeout=5 --tries=0 -O {shlex.quote(str(temp_sav_path))} {shlex.quote(temp_key)}"
                    result = subprocess.run(command, shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    print(f'Downloaded file {temp_sav_path}')
                    logging.info(f'Downloaded file {temp_sav_path}')
                    if result.returncode != 0:
                        print(f"Download failed for: {temp_key} with error {result.stderr}")
                        logging.info(f"Download failed for: {temp_key} with error {result.stderr}")
                else:
                    print(f"File already exists: {temp_sav_path}, skipping download.")
                    logging.info(f"File already exists: {temp_sav_path}, skipping download.")

            temp_dict = {"tile":tile, "timestep":i, "date":j.date, "save_path":tile_dir, "filename":bands_dict["LB02"].split('/')[6].replace(".B02.tif","")}
            info_list.append(temp_dict)

    return pd.DataFrame(info_list)

def point_transform(coor, src_crs, target_crs=5070):
    proj = pyproj.Transformer.from_crs(src_crs, target_crs, always_xy=True)
    projected_coor = proj.transform(coor[0], coor[1])
    return [projected_coor[0], projected_coor[1]]

def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def reproject_hls(tile_path,
                  cdl_ds,
                  target_crs ="EPSG:5070", 
                  remove_original = True, 
                  resampling_method = Resampling.bilinear):
    
    """
    This function receives the path to a specific HLS tile and reproject it to the targeting crs_ds.
    The option of removing the raw HLS tile is provided
    
    Assumptions:
    - tile_path is a full path that end with .tif
    - cdl_ds is a rioxarray dataset that is opened with `cache=False` setting.
    
    
    Inputs:
    - tile_path: The full path to a specific HLS tile
    - target_crs: The crs that you wish to reproject the tile to, default is EPSG 4326
    - remove_original: The option to remove raw HLS tile after reprojecting, default is True
    - resampling_method: The method that rioxarray use to reproject, default is bilinear
    """
    logging.info('Inside reproject HLS function')
    xds = rioxarray.open_rasterio(tile_path)
    half_scene_len = np.abs(np.round((xds.x.max().data - xds.x.min().data) / 2))
    coor_min = point_transform([xds.x.min().data - half_scene_len, xds.y.min().data - half_scene_len], xds.rio.crs)
    coor_max = point_transform([xds.x.max().data + half_scene_len, xds.y.max().data + half_scene_len], xds.rio.crs)
    x0 = find_nearest(cdl_ds.x.data, coor_min[0])
    y0 = find_nearest(cdl_ds.y.data, coor_min[1])
    x1 = find_nearest(cdl_ds.x.data, coor_max[0])
    y1 = find_nearest(cdl_ds.y.data, coor_max[1])
    
    cdl_for_reprojection = cdl_ds.rio.slice_xy(x0, y0, x1, y1)
    
    xds_new = xds.rio.reproject_match(cdl_for_reprojection, resampling = resampling_method)
    if remove_original:
        if Path(tile_path).is_file():
            os.remove(tile_path)
        xds_new.rio.to_raster(raster_path = tile_path.replace(".tif", ".reproject.tif"))
    else:
        xds_new.rio.to_raster(raster_path = tile_path.replace(".tif", ".reproject.tif"))

def hls_process(req_path, state_name_year, cdl_file):
    logging.info('Inside HLS Process function')
    track_df = pd.read_csv(req_path + '/' + f"track_df_{state_name_year}.csv")
    track_df["cdl_file"] = cdl_file
    track_df.loc[:, "bands"] = '["B02", "B03", "B04", "B05", "B06", "Fmask"]'

    print(f"Total {len(track_df)} Tiles need to reproj")

    remove_original = False
    not_process_tiles = []
    for index, track in tqdm(track_df.iterrows()):

        save_path = track["save_path"]
        filename= track["filename"]
        bands = json.loads(track["bands"])
        cdl_file = track["cdl_file"]
        
        cdl_ds = rioxarray.open_rasterio(cdl_file, cache=False)
        
        for band in bands:
            tile_path_reproj = f"{save_path}{filename}.{band}.reproject.tif"
            try:
                if not Path(tile_path_reproj).is_file():
                    tile_path = f"{save_path}{filename}.{band}.tif"
                    if Path(tile_path).is_file():
                        if band == "Fmask":
                            print(tile_path)
                            logging.info(f'Processing HLS for tile {tile_path}')
                            reproject_hls(tile_path, cdl_ds, remove_original, resampling_method = Resampling.nearest)
                        else :
                            print(tile_path)
                            logging.info(f'Processing HLS for tile {tile_path}')
                            reproject_hls(tile_path, cdl_ds, remove_original)
                    else:
                        print('File is not available')
                        logging.info(f'File {tile_path} is not available')
                else:
                    print(f'File {filename}.{band}.reproject.tif Already Exist')
                    logging.info(f'File {filename}.{band}.reproject.tif Already Exist')
            except Exception as e:
                print(e)
                not_process_tiles.append(tile_path)
                break


def create_chip_df_tiles(req_path, chip_file, kml_file, state_name_year, cloud_thres):
    logging.info('Inside tiles collection function')    
    with open(chip_file, "r") as file:
        chips = json.load(file)
        # print(chips)
    print(f'Reading File {chip_file} Successfully')
    logging.info(f'File {chip_file} is Successfully Opened')

    # Create lists about chip information to find tiles corresponding to it later
    chip_ids = []
    chip_x = []
    chip_y = []
    for item in chips['features']:
        chip_ids.append(item['properties']['id'])
        chip_x.append(item['properties']['center'][0])
        chip_y.append(item['properties']['center'][1])
    
    with open(req_path + '/' + f"chip_ids_{state_name_year}.json", "w") as f:
        json.dump(chip_ids, f, indent=2)

    print(f'Chip ids chip_ids_{state_name_year}.json file created Successfully')
    logging.info(f'Chip ids chip_ids_{state_name_year}.json file created Successfully')

    fiona.drvsupport.supported_drivers['KML'] = 'rw'
    tile_src = geopandas.read_file(kml_file, driver='KML')
    print('Reading Sentinel KML file')
    logging.info('Sentinel KML file is opened successfully')
    # Create table containing information about sentinel tiles
    tile_name = []
    tile_x = []
    tile_y = []
    for tile_ind in range(tile_src.shape[0]):
        tile_name.append(tile_src.iloc[tile_ind].Name)
        tile_x.append(tile_src.iloc[tile_ind].geometry.centroid.x)
        tile_y.append(tile_src.iloc[tile_ind].geometry.centroid.y)
    tile_name = np.array(tile_name)
    tile_x = np.array(tile_x)
    tile_y = np.array(tile_y)
    tile_src = pd.concat([tile_src, tile_src.bounds], axis = 1)

    chip_df = pd.DataFrame({"chip_id" : chip_ids, "chip_x" : chip_x, "chip_y" : chip_y})
    chip_df['tile'] = chip_df.apply(lambda row : find_tile(tile_x, row['chip_x'], tile_y, row['chip_y'], tile_name), axis = 1)
    # chip_df.tail(5)
    print('Tiles list is collected from Chip ID')
    logging.info('Tiles list is collected from Chip ID')
    file_name = req_path + '/' + f"chip_df_{state_name_year}.csv"
    chip_df.to_csv(file_name, index=False)
    print(f'File {file_name} is Saved Successfully')
    logging.info(f'File {file_name} is Saved Successfully')



    chip_df = pd.read_csv(req_path + '/' + f"chip_df_{state_name_year}.csv")
    tiles = chip_df.tile.unique().tolist()
    print(f'File read successfully chip_df_{state_name_year}')
    logging.info(f'File read successfully chip_df_{state_name_year}')

    print(f"There are a total of {len(tiles)} tiles")
    logging.info(f"There are a total of {len(tiles)} tiles")
    tile_list = []
    tile_iter = 0
    for current_tile in tqdm(tiles):
        print(current_tile)
        logging.info(f'Processing {current_tile}')

        chip_df_filt = chip_df.loc[chip_df.tile == current_tile]#.reset_index()
        first_chip_id = chip_df_filt.chip_id.iloc[0]
        first_chip_index_in_json = chip_ids.index(first_chip_id)
        roi = chips['features'][first_chip_index_in_json]['geometry']

        polygon = shape(roi)
        
        temporal = (f"{year}-01-01T00:00:00", f"{year}-12-31T23:59:59")

        results = earthaccess.search_data(
        short_name=['HLSL30'],
        bounding_box=polygon.bounds,
        temporal=temporal,
        count=100
        )

        print('Total Number of Granules', len(results))
        logging.info(f'Total Number of Granules {len(results)}') 
           
        tile_name = "T" + current_tile
        iter_items = 0
        for i in results:
            if i['umm']['GranuleUR'].split('.')[2] == tile_name:
                if int(i['umm']['AdditionalAttributes'][1]['Values'][0]) <= cloud_thres:
                    for url in i.data_links():
                        # print(url.split('/')[4])
                        if 'B02' in url:
                            Band_B02 = url

                        if 'B03' in url:
                            Band_B03 = url

                        if 'B04' in url:
                            Band_B04 = url

                        if 'B05' in url:
                            Band_B05 = url

                        if 'B06' in url:
                            Band_B06 = url

                        if 'Fmask' in url:
                            Band_Fmask = url
                        
                    temp_dict = {"tile_id": tile_name, "cloud_cover": int(i['umm']['AdditionalAttributes'][1]['Values'][0]),
                                    "date": datetime.datetime.strptime(i['umm']['TemporalExtent']['RangeDateTime']['BeginningDateTime'].split('T')[0], "%Y-%m-%d"),
                                    "spatial_cover": int(i['umm']['AdditionalAttributes'][3]['Values'][0]),
                                    "http_links": {"LB02": Band_B02, "LB03": Band_B03, "LB04": Band_B04, "LB05": Band_B05, "LB06": Band_B06, "Fmask":Band_Fmask}}
                    tile_list.append(temp_dict)
                    iter_items += 1
                    
        tile_iter += 1
        
    tile_df = pd.DataFrame(tile_list)
    file_name = req_path + '/' + f"tile_df_{state_name_year}.csv"
    tile_df.to_csv(file_name, index=False)
    print(f'File {file_name} is Saved Successfully')
    logging.info(f'File {file_name} is Saved Successfully')

def tiles_downloading(req_path, state_name_year, tile_dir):

    tile_df = pd.read_csv(req_path + '/' + f"tile_df_{state_name_year}.csv")

    print(f'There are a total of {len(tile_df)} tiles')
    logging.info(f'There are a total of {len(tile_df)} tiles')

    selected_tiles = spatial_filtering(tile_df)

    print(f'Filtered {len(selected_tiles)} Tiles based on Spatial Cover')
    logging.info(f'Filtered {len(selected_tiles)} Tiles based on Spatial Cover')

    file_name = req_path + '/' + f"selected_tiles_{state_name_year}.csv"

    selected_tiles.to_csv(file_name, index=False)
    print(f'File {file_name} is Saved Successfully')
    logging.info(f'File {file_name} is Saved Successfully')

    track_df = tile_download(selected_tiles, tile_dir, from_csv=True)

    file_name = req_path + '/' + f"track_df_{state_name_year}.csv"
    track_df.to_csv(file_name, index=False)
    print(f'File {file_name} is Saved Successfully')
    logging.info(f'File {file_name} is Saved Successfully')



def check_qa(qa_path, shape,  valid_qa = [0, 4, 32, 36, 64, 68, 96, 100, 128, 132, 160, 164, 192, 196, 224, 228]):
    
    """
    This function receives a path to a qa file, and a geometry. It clips the QA file to the geometry. 
    It returns the number of valid QA pixels in the geometry, and the clipped values.
    
    Assumptions: The valid_qa values are taken from Ben Mack's post:
    https://benmack.github.io/nasa_hls/build/html/tutorials/Working_with_HLS_datasets_and_nasa_hls.html
    
    Inputs:
    - qa_path: full path to reprojected QA tif file
    - shape: 'geometry' property of single polygon feature read by fiona
    - valid_qa: list of integer values that are 'valid' for QA band.
    

    
    """
    with rasterio.open(qa_path) as src:
        out_image, out_transform = rasterio.mask.mask(src, shape, crop=True)
        vals = out_image.flatten()
        unique, counts = np.unique(vals, return_counts=True)
        qa_df = pd.DataFrame({"qa_val" : unique, "counts" : counts})
        qa_df
        qa_df[~ qa_df.qa_val.isin(valid_qa)].sort_values(['counts'], ascending = False)
        qa_df['pct'] = (100 *qa_df['counts'])/(224.0 * 224.0)
        
        bad_qa = qa_df[~ qa_df.qa_val.isin(valid_qa)].sort_values(['counts'], ascending = False)
        if len(bad_qa) > 0:
            highest_invalid_percent = bad_qa.pct.tolist()[0]
        else: 
            highest_invalid_percent = 0
        # ncell = len(vals)
        valid_count = sum(x in valid_qa for x in vals)
        return(valid_count, highest_invalid_percent, out_image[0])

def tiles_to_chips(req_path, state_name_year, chip_file, chipping_json, cdl_reclass_csv, chip_csv, cdl_file, chip_dir, chip_fmask_dir):

    # Getting all saved dataframes and json
    chip_df = pd.read_csv(req_path + '/' + f"chip_df_{state_name_year}.csv")
    with open(req_path + '/' + f"chip_ids_{state_name_year}.json", 'r') as f: 
        chip_ids = json.load(f)
    track_df = pd.read_csv(req_path + '/' + f"track_df_{state_name_year}.csv")
    with open(chip_file, "r") as file:
        chips = json.load(file)

    tiles_to_chip = track_df.tile.unique().tolist()
    with open(chipping_json, "r") as file_chip:
        chipping_js = json.load(file_chip)

    cdl_class_df = pd.read_csv(cdl_reclass_csv)
    crop_dict = dict(zip(cdl_class_df.old_class_value, cdl_class_df.new_class_value))
    def crop_multi(x):
        return(crop_dict[x])
    c_multi = np.vectorize(crop_multi)

    failed_tiles = []
    temp_lst = []
    chip_df_new = pd.DataFrame()
    print("Total Chips to process", len(tiles_to_chip))
    logging.info(f"Total Chips to process {len(tiles_to_chip)}")

    for tile in tqdm(tiles_to_chip):
        print(len(tile))
        logging.info(f'Total Tiles {len(tile)}') 
        chips_to_process = chip_df[chip_df.tile == tile[1:]].reset_index(drop = True)
        for k in range(len(chips_to_process)):
            current_id = chips_to_process.chip_id[k]
            chip_tile = chips_to_process.tile[k]
            chip_index = chip_ids.index(current_id)

            chip_feature = chipping_js['features'][chip_index]

            shape = [chip_feature['geometry']]
            full_tile_name = "T" + chip_tile
            # try:

            tile_info_df = track_df[track_df.tile == full_tile_name]
            selected_image_folders = tile_info_df.save_path.to_list()
            all_date_images = []
            all_date_qa = []
            
            for i in range(len(tile_info_df)):
                image_date = tile_info_df.iloc[i].date
                all_date_images.append(tile_info_df.iloc[i].save_path + f"{tile_info_df.iloc[i].filename}")
                all_date_qa.append(tile_info_df.iloc[i].save_path + f"{tile_info_df.iloc[i].filename}.Fmask.reproject.tif")
            # print(f'There are total {len(all_date_images)} to process')
            for i in all_date_images: 
                try:
                    chip_id_val, chip_x, chip_y, tile, valid_first_val, bad_pct_first_val, image_date, na_count_evi, na_count_ndvi = process_chip(current_id, full_tile_name, shape, track_df, cdl_file, chip_df, i, tile_info_df, chip_dir, chip_fmask_dir, c_multi)
                    
                    temp_dict = {'chip_id' : chip_id_val,
                    'chip_x' : chip_x,
                    'chip_y' : chip_y,
                    'tile' : tile,
                    'valid_first' : valid_first_val,
                    'bad_pct_first' : bad_pct_first_val,
                    'image_date' : image_date,
                    'na_count_evi' : na_count_evi,
                    'na_count_ndvi' : na_count_ndvi,
                                }
                    temp_lst.append(temp_dict)
                
                except Exception as e:
                    print(e)
                    failed_tiles.append(tile)
                    break

    chip_df_new = pd.DataFrame(temp_lst)
    chip_df_new.to_csv(chip_csv, index=False)


def calc_evi(nir, red, blue):
    """Calculate the Enhanced Vegetation Index (EVI) from NIR, Red, and Blue bands."""
    return 2.5 * (nir.astype(float) - red.astype(float)) / (nir.astype(float) + 6 * red.astype(float) - 7.5 * blue.astype(float) + 1)

def calc_ndvi(nir, red):
    """Calculate the Normalized Difference Vegetation Index (NDVI)."""
    return (nir.astype(float) - red.astype(float)) / (nir.astype(float) + red.astype(float))


def process_chip(chip_id, 
                 chip_tile,
                 shape,
                 track_csv,
                 cdl_file,
                 chip_df,
                 i,
                 tile_info_df,
                 chip_dir,
                 chip_fmask_dir,
                 c_multi,
                 bands = ["B02", "B03", "B04", "B05", "B06"]):
    
    """
    This function receives a chip id, HLS tile, chip geometry, and a list of bands to process. 
    
    Assumptions:
    
    Inputs:
    - chip_id: string of chip id, e.g. '000_001'
    - chip_tile: string of HLS tile , e.g. '15ABC'
    - shape: 'geometry' property of single polygon feature read by fiona
    
    The function writes out a multi-date TIF containing the bands for each of the three image dates for an HLS tile. 
    The function writes out a multi-date TIF containing the QA bands of each date.
    The function writes out a chipped version of CDL. 
    The function calls check_qa(), which makes assumptions about what QA pixels are valid.
    The function returns the number of valid QA pixels at each date, as a tuple.
    
    """
    np.seterr(divide='ignore', invalid='ignore')
    blue_img = i + '.B02.reproject.tif'
    green_img = i + '.B03.reproject.tif'
    red_img = i + '.B04.reproject.tif'
    nir_img = i + '.B06.reproject.tif'
    swir_img = i + '.B06.reproject.tif'
    mask_img = i + '.Fmask.reproject.tif'
    with rasterio.open(swir_img) as swir_src, \
         rasterio.open(nir_img) as nir_src, \
         rasterio.open(red_img) as red_src, \
         rasterio.open(green_img) as green_src, \
         rasterio.open(blue_img) as blue_src:
        
        # Mask data with the given shape and crop to extent
        nir_data, transform = mask(nir_src, shape, crop=True)
        swir_data, _ = mask(swir_src, shape, crop=True)
        red_data, _ = mask(red_src, shape, crop=True)
        green_data, _ = mask(green_src, shape, crop=True)
        blue_data, _ = mask(blue_src, shape, crop=True)

        # Calculate EVI
        evi_data = calc_evi(nir_data[0], red_data[0], blue_data[0])

        # Calculate NDVI
        ndvi_data = calc_ndvi(nir_data[0], red_data[0])
        
        out_bands_evi = np.array(evi_data)
        
        # Setup metadata for output file
        out_meta = nir_src.meta
        out_meta.update({
            "driver": "GTiff",
            "dtype": 'float32',
            "height": out_bands_evi.shape[0],
            "width": out_bands_evi.shape[1],
            "count": 1,  # Single band for EVI
            "transform": transform,
            "compress": 'LZW'
        })
        
        na_count_evi = sum(evi_data.flatten() == -1000)
        na_count_ndvi = sum(ndvi_data.flatten() == -1000)

        # reclass negative HLS values to 0
        evi_data = np.clip(evi_data, 0, None)
        ndvi_data = np.clip(ndvi_data, 0, None)

        # Define the output filename based on input or other logic
        date_string = tile_info_df.loc[tile_info_df['filename']== i.split("/")[-1]].date.item().replace("/", "_")
        print(date_string)
        
        evi_output_filename = chip_dir + str(chip_id) + '_' + str(date_string) + "_evi.tif"
        ndvi_output_filename = chip_dir + str(chip_id) + '_' + str(date_string) + "_ndvi.tif" 
        false_output_filename = chip_dir + str(chip_id) + '_' + str(date_string) + "_false_color.tif"
        agri_output_filename = chip_dir + str(chip_id) + '_' + str(date_string) + "_ag_band.tif"
        veg_output_filename = chip_dir + str(chip_id) + '_' + str(date_string) + "_vg_band.tif"
        health_output_filename = chip_dir + str(chip_id) + '_' + str(date_string) + "_hv_band.tif"
        # Write the EVI data to a new TIFF file
        with rasterio.open(evi_output_filename, "w", **out_meta) as dest:
            dest.write(evi_data.astype(rasterio.float32), 1)
            print(f"EVI image saved: {evi_output_filename}")
            logging.info(f"EVI image saved: {evi_output_filename}")
        
        with rasterio.open(ndvi_output_filename, "w", **out_meta) as dest:
            dest.write(ndvi_data.astype(rasterio.float32), 1)
            print(f"NDVI image saved: {ndvi_output_filename}")
            logging.info(f"NDVI image saved: {ndvi_output_filename}")


        out_meta.update({
            "driver": "GTiff",
            "dtype": 'float32',
            "height": out_bands_evi.shape[0],
            "width": out_bands_evi.shape[1],
            "count": 3,
            "transform": transform,
            "compress": 'LZW'
        })

        with rasterio.open(false_output_filename, "w", **out_meta) as dest_f:
            dest_f.write(nir_data[0].astype(rasterio.float32), 1)
            dest_f.write(red_data[0].astype(rasterio.float32), 2)
            dest_f.write(green_data[0].astype(rasterio.float32), 3)
            print(f"False Color image saved: {false_output_filename}")
            logging.info(f"False Color image saved: {false_output_filename}")

        with rasterio.open(agri_output_filename, "w", **out_meta) as dest_a:
            dest_a.write(swir_data[0].astype(rasterio.float32), 1)
            dest_a.write(nir_data[0].astype(rasterio.float32), 2)
            dest_a.write(blue_data[0].astype(rasterio.float32), 3)
            print(f"Agriculture Color image saved: {agri_output_filename}")
            logging.info(f"Agriculture Color image saved: {agri_output_filename}")


        with rasterio.open(veg_output_filename, "w", **out_meta) as dest_v:
            dest_v.write(swir_data[0].astype(rasterio.float32), 1)
            dest_v.write(nir_data[0].astype(rasterio.float32), 2)
            dest_v.write(red_data[0].astype(rasterio.float32), 3)
            print(f"Vegetation Color image saved: {veg_output_filename}")
            logging.info(f"Vegetation Color image saved: {veg_output_filename}")

        with rasterio.open(health_output_filename, "w", **out_meta) as dest_h:
            dest_h.write(nir_data[0].astype(rasterio.float32), 1)
            dest_h.write(swir_data[0].astype(rasterio.float32), 2)
            dest_h.write(blue_data[0].astype(rasterio.float32), 3)
            print(f"Health Vegetation Color image saved: {health_output_filename}")
            logging.info(f"Health Vegetation Color image saved: {health_output_filename}")

        valid_first, bad_pct_first, qa_first = check_qa(mask_img, shape)
        date = tile_info_df.loc[tile_info_df['filename'] == i.split("/")[-1]].date.item()

        out_meta.update({"driver": "GTiff",
                         "dtype": 'float32',
                     "height": qa_first.shape[0],
                     "width": qa_first.shape[1],
                     "count": 1,
                     "transform": transform,
                     "compress": 'LZW'})
        mask_output_filename = chip_fmask_dir + str(chip_id) + '_' + str(date_string) + "_Fmask.tif"
        with rasterio.open(mask_output_filename, "w", **out_meta) as dest:
            dest.write(qa_first, 1)
            print(f"mask_Label image saved: {mask_output_filename}")
            logging.info(f"mask_Label image saved: {mask_output_filename}")


        with rasterio.open(cdl_file) as src:
            out_image, out_transform = rasterio.mask.mask(src, shape, crop=True)
            out_meta = src.meta
            colormap = src.colormap(1)
        out_meta.update({"driver": "GTiff",
                     "height": out_image.shape[1],
                     "width": out_image.shape[2],
                     "transform": transform,
                     "compress": 'LZW'})
        out_image_multi = c_multi(out_image).astype(np.uint8)
        cdl_mask_output_filename = chip_dir + str(chip_id) + '_' + str(date_string) + ".mask.tif"
        with rasterio.open(cdl_mask_output_filename, "w", **out_meta) as dest:
            dest.write(out_image_multi)
            dest.write_colormap(1, colormap)
            print(f"CDL_Label image saved: {cdl_mask_output_filename}")
            logging.info(f"CDL_Label image saved: {cdl_mask_output_filename}")

        print(chip_id)
        print(i.split('.')[2][1:])
        chip_val = chip_df[(chip_df['chip_id'] == chip_id) & (chip_df['tile'] == i.split('.')[2][1:])]

        chip_id_val = chip_id
        chip_x = chip_val['chip_x'].values[0]
        chip_y = chip_val['chip_y'].values[0]
        tile = i.split('.')[2]
        valid_first_val = valid_first
        bad_pct_first_val = bad_pct_first
        image_date = date
        na_count_evi = na_count_evi
        na_count_ndvi = na_count_ndvi

    return chip_id_val, chip_x, chip_y, tile, valid_first_val, bad_pct_first_val, image_date, na_count_evi, na_count_ndvi


def main(state_name, year):
   
    logging.info(f'Starting HLS Data Downloading and Processing Pipeline for State {state_name} and Year {year}')
    auth = earthaccess.login(persist=True)
    if auth:
        print('Successfully Login to Earth Data')
        logging.info('Successfully Login to Earth Data')

    cloud_thres = 25 # percent cloud cover for tile level query
    logging.info(f'Cloud Threshold is {cloud_thres}')

    state_name_year = state_name + '_' + year

    root_path = "/home/npatel23/gokhale_user/Crop Classification Project/"
    req_path = f"/home/npatel23/gokhale_user/Crop Classification Project/HLS_CDL_Data/{state_name}/{state_name_year}"
    req_path_state = f"/home/npatel23/gokhale_user/Crop Classification Project/HLS_CDL_Data/{state_name}/"
    extra_files = root_path

    chip_dir = req_path + '/chips/'
    tile_dir = req_path + '/tiles/'
    chip_dir_filt = req_path + '/chips_filtered/'
    chip_fmask_dir = req_path + '/chips_fmask/'

    if not os.path.exists(req_path):
        os.makedirs(req_path)
        print(f"Created directory: {req_path}")
        logging.info(f"Created directory: {req_path}")
    else:
        print(f"Directory already exists: {req_path}")
        logging.info(f"Directory already exists: {req_path}")

    if not os.path.exists(chip_dir):
        os.makedirs(chip_dir)
        print(f"Created directory: {chip_dir}")
        logging.info(f"Created directory: {chip_dir}")
    else:
        print(f"Directory already exists: {chip_dir}")
        logging.info(f"Directory already exists: {chip_dir}")

    if not os.path.exists(tile_dir):
        os.makedirs(tile_dir)
        print(f"Created directory: {tile_dir}")
        logging.info(f"Created directory: {tile_dir}")
    else:
        print(f"Directory already exists: {tile_dir}")
        logging.info(f"Directory already exists: {tile_dir}")

    if not os.path.exists(chip_dir_filt):
        os.makedirs(chip_dir_filt)
        print(f"Created directory: {chip_dir_filt}")
        logging.info(f"Created directory: {chip_dir_filt}")
    else:
        print(f"Directory already exists: {chip_dir_filt}")
        logging.info(f"Directory already exists: {chip_dir_filt}")

    if not os.path.exists(chip_fmask_dir):
        os.makedirs(chip_fmask_dir)
        print(f"Created directory: {chip_fmask_dir}")
        logging.info(f"Created directory: {chip_fmask_dir}")
    else:
        print(f"Directory already exists: {chip_fmask_dir}")  
        logging.info(f"Directory already exists: {chip_fmask_dir}")


    chip_file =  req_path_state + f"chip_bbox_task_{state_name}_4326.geojson"          #User Input
    chipping_json = req_path_state + f"chip_bbox_task_{state_name}_5070.geojson"       #User Input
    chip_csv = req_path + f"chip_tracker_{state_name_year}.csv"
    # chip_csv_new = req_path_state + "chip_tracker_new_cali.csv"
    kml_file = extra_files + 'sentinel_tile_grid.kml'                               #User Input
    # tile_tracker_csv = req_path + "tile_tracker.csv"
    state_name_upper = state_name.upper()
    cdl_file = req_path + '/' + f"CDL_{year}_{state_name_upper}.tif"                #User Input
    cdl_reclass_csv = root_path + "cdl_total_dst_main.csv"                          #User Input

    create_chip_df_tiles(req_path, chip_file, kml_file, state_name_year, cloud_thres)

    tiles_downloading(req_path, state_name_year, tile_dir)

    hls_process(req_path, state_name_year, cdl_file)

    tiles_to_chips(req_path, state_name_year, chip_file, chipping_json, cdl_reclass_csv, chip_csv, cdl_file, chip_dir, chip_fmask_dir)

if __name__ == "__main__":
    state_name = sys.argv[1]
    year = sys.argv[2]
    logging.basicConfig(filename=f"batch_downloading_{state_name}_{year}.log",
                level=logging.INFO,
                format='%(asctime)s %(message)s',
                filemode='a')
    main(state_name, year)
