import rioxarray 
import rasterio
import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import json
from shapely.geometry import Polygon
import os
import sys
from tqdm import tqdm
import requests
import re


def calc_standard_deviation(chips_df):
    # Calculate mean and standard deviation

    values = chips_df['sum'].tolist()
    # print(len(values))

    mean = np.mean(values)
    std_dev = np.std(values)
    print(f"Mean: {mean}")
    print(f"Standard Deviation: {std_dev}")

    # Transform the values to standard normal (z-scores)
    standardized_values = [(x - mean) / std_dev for x in values]
    # print("Original Values:", values)
    # print("Standardized Values (Z-scores):", standardized_values)
    new_mean = np.mean(standardized_values)
    new_std_dev = np.std(standardized_values)
    print(f"Mean: {new_mean}")
    print(f"Standard Deviation: {new_std_dev}")

    # Define thresholds
    lower_bound = new_mean - 1 * new_std_dev
    upper_bound = new_mean + 1 * new_std_dev

    print(f"Lower Bound (μ - σ): {lower_bound}")
    print(f"Upper Bound (μ + σ): {upper_bound}")

    # Filter values within the range [μ - 2σ, μ + 2σ]
    filtered_values = [x for x in standardized_values if lower_bound <= x <= upper_bound]
    print(len(filtered_values))
    print(len(standardized_values))

    indices_dict = {element: [i for i, value in enumerate(standardized_values) if value == element] for element in filtered_values}
    indicess=[]
    for element, indices in indices_dict.items():
        
        indicess.extend(indices)
    
    filtered_df = chips_df.iloc[~chips_df.index.isin(indicess)]
    
    return filtered_df

def code_to_name(filename, state_mapping):
    # Change download file name from code to Name
    parts = filename.split('_')
    code = parts[2].replace(".tif", '')
    if code in state_mapping:
        parts[2] = state_mapping[code]
        filename = '_'.join(parts)
        filename = filename + '.tif'
    return filename

# Function to get the number by state name
def get_state_number(state_name, state_mapping):
    state_name_to_number = {v: k for k, v in state_mapping.items()}
    return state_name_to_number.get(state_name, "State not found")

def download_cdl_state(cdl_year, state_number, state_mapping):
    # Download CDL Data based on US State
    for i in cdl_year:
        url = f'https://nassgeodata.gmu.edu/axis2/services/CDLService/GetCDLFile?year={str(i)}&fips={str(state_number)}'
        response = requests.get(url)
        response.raise_for_status()
        data = response.text
        match = re.search(r'<returnURL>(.*?)</returnURL>', data)
        if match:
            print(match.group(1))
            url_filename = match.group(1)
        else:
            print("URL not found")
        
        filename = url_filename.split('/')[-1]
        filename = code_to_name(filename, state_mapping)
        if os.path.exists(filename):
            print(f"File already Exists {filename}")
        else:
            file_response = requests.get(url_filename)
            file_response.raise_for_status()  # Ensure the download is successful
            with open(filename, 'wb') as file:
                file.write(file_response.content)
            print(f"File downloaded successfully and saved as {filename}.")


def generate_bbox(cdl_year, state_name):
    

    cdl_class_df = pd.read_csv("/home/npatel23/gokhale_user/Crop Classification Project/Bounding_Box/cdl_classes.csv", encoding = "ISO-8859-1")
    # print(cdl_class_df)
    
    # Selecting only Class containing Crop and Open-Water
    classes_to_select = [1, 2, 3, 4, 5, 6, 10, 12, 21, 22, 23, 24, 27, 28, 29, 36, 45, 75]
    # classes_to_select = [1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60]

    # Filter the DataFrame to exclude rows with the specified classes
    cdl_class_df = cdl_class_df[cdl_class_df['class'].isin(classes_to_select)]

    cdl_class_valid = cdl_class_df[~cdl_class_df['value'].isna()]["class"].values

    columns_to_extract = ["chip_id", "chip_coordinate_x", "chip_coordinate_y", "Crop Per", "Non-Crop Per"]
    final_df = None

    for year in cdl_year:


        cdl_file = f"CDL_{year}_{state_name}.tif"
        xds = rioxarray.open_rasterio(cdl_file, cache=False)

        # Define chip specs
        chip_dim_x = 224
        chip_dim_y = 224
        x0 = xds.x.data[0]
        y0 = xds.y.data[0]
        res = 30 # meters

        df_columns = ["chip_id"]
        # df_columns.append(0)
        for i in cdl_class_df["class"]:
            df_columns.append(i)
        df_columns.append("chip_coordinate_x")
        df_columns.append("chip_coordinate_y")
        df_columns.append("Crop Per")
        df_columns.append("Non-Crop Per")
        chips_df = pd.DataFrame(columns = df_columns)

        relevant_classes = chips_df.columns[1:-4]  # Adjust indexing as per columns in chips_df
        # relevant_classes = [col for col in chips_df.columns if col.isdigit()]
        relevant_class_indices = [int(class_label) for class_label in relevant_classes]  # Convert class names to integer indices

        for idx in tqdm(range(0, int(np.floor(xds.shape[2] / chip_dim_x)))):
            for idy in range(0, int(np.floor(xds.shape[1] / chip_dim_y))):
                chip_id = f"chip_{str(idy).zfill(3)}_{str(idx).zfill(3)}"
                
                chip_x0 = x0 + idx * chip_dim_x * res
                chip_y0 = y0 - idy * chip_dim_y * res

                chip = xds.rio.slice_xy(chip_x0, chip_y0 - (chip_dim_y - 1) * res, chip_x0 + (chip_dim_x - 1) * res, chip_y0)
                classes, class_counts = np.unique(chip.data, return_counts=True)
                
                if 0 not in classes:
                    counts = np.zeros(len(relevant_class_indices))
                    relevant_sum = 0  # Initialize sum for relevant classes
                    irrelevant_sum = 0  # Initialize sum for irrelevant classes
                    relevant_per = 0
                    irrelevant_per = 0
                    for i, cls in enumerate(classes):
                        if cls in relevant_class_indices:
                            relevant_index = relevant_class_indices.index(cls)  # Get the index for relevant class
                            counts[relevant_index] = class_counts[i]  # Update count for this class
                            # print("classes----->",classes, "counts----->", class_counts)
                            relevant_sum += class_counts[i]
                            relevant_per = round(relevant_sum / 50176 * 100, 2)
                        else:
                            irrelevant_sum += class_counts[i]
                            irrelevant_per = round(irrelevant_sum / 50176 * 100, 2)
                    # print(f"Sum of relevant classes: {relevant_sum}")
                    # print(f"Sum of irrelevant classes: {irrelevant_sum}")
                    
                    chips_df.loc[len(chips_df.index)] = [chip_id] + counts.tolist() + [chip_x0, chip_y0] + [relevant_per] + [irrelevant_per]

        chips_df['sum'] = chips_df.loc[:, classes_to_select].sum(axis=1)

        filtered_df = calc_standard_deviation(chips_df)

        integer_columns = [col for col in filtered_df.columns if isinstance(col, int)]
        non_integer_columns = [col for col in filtered_df.columns if not isinstance(col, int)]

        renumbered_columns = {old: new for new, old in enumerate(sorted(integer_columns), start=1)}
        filtered_df.rename(columns=renumbered_columns, inplace=True)
        print(f'File cdl_total_{state_name}_{year}.csv saved sucessfully')
        filtered_df.to_csv(f'cdl_total_{state_name.lower()}_{year}.csv', index=False)

        df_year_subset = filtered_df[columns_to_extract].copy()
        df_year_subset.rename(
            columns={
                "Crop Per": f"Crop Per {year}",
                "Non-Crop Per": f"Non-Crop Per {year}"
            },
            inplace=True,
        )
        
        if final_df is None:
            # For the first year, initialize final_df with data from the current year
            final_df = df_year_subset
        else:
            # Merge the current year data into final_df
            final_df = final_df.merge(
                df_year_subset,
                on=["chip_id", "chip_coordinate_x", "chip_coordinate_y"],
                how="outer",
                suffixes=("", f"_{year}"),
            )

        # Handle missing values after merging
        final_df.fillna(0, inplace=True)
    
    final_df.to_csv(f'cdl_{state_name.lower()}.csv', index=False)
    print(f'File cdl_{state_name.lower()}.csv saved sucessfully')


def main(state_name):
    print('Starting')
    print(state_name)
    root_path = '/home/npatel23/gokhale_user/Crop Classification Project/Bounding_Box'

    state_mapping = {
        "01": "ALABAMA", "04": "ARIZONA", "05": "ARKANSAS", "06": "CALIFORNIA",
        "08": "COLORADO", "09": "CONNECTICUT", "10": "DELAWARE", "11": "DISTRICT_OF_COLUMBIA",
        "12": "FLORIDA", "13": "GEORGIA", "16": "IDAHO", "17": "ILLINOIS",
        "18": "INDIANA", "19": "IOWA", "20": "KANSAS", "21": "KENTUCKY",
        "22": "LOUISIANA", "23": "MAINE", "24": "MARYLAND", "25": "MASSACHUSETTS",
        "26": "MICHIGAN", "27": "MINNESOTA", "28": "MISSISSIPPI", "29": "MISSOURI",
        "30": "MONTANA", "31": "NEBRASKA", "32": "NEVADA", "33": "NEW_HAMPSHIRE",
        "34": "NEW_JERSEY", "35": "NEW_MEXICO", "36": "NEW_YORK", "37": "NORTH_CAROLINA",
        "38": "NORTH_DAKOTA", "39": "OHIO", "40": "OKLAHOMA", "41": "OREGON",
        "42": "PENNSYLVANIA", "44": "RHODE_ISLAND", "45": "SOUTH_CAROLINA",
        "46": "SOUTH_DAKOTA", "47": "TENNESSEE", "48": "TEXAS", "49": "UTAH",
        "50": "VERMONT", "51": "VIRGINIA", "53": "WASHINGTON", "54": "WEST_VIRGINIA",
        "55": "WISCONSIN", "56": "WYOMING"
    }

    fip_cdl_state = ["01", "04", "05", "06", "08", "09", "10", "11", "12", "13", 
                    "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", 
                    "26", "27", "28", "29", "30", "31", "32", "33", "34", "35", 
                    "36", "37", "38", "39", "40", "41", "42", "44", "45", "46", 
                    "47", "48", "49", "50", "51", "53", "54", "55", "56"]

    cdl_year = [2023, 2022, 2021, 2020, 2019, 2018, 2017, 2016, 2015, 2014, 2013]

    # state_name = "CALIFORNIA"  # Replace with the input state name

    if not os.path.exists(state_name.lower()):
        os.makedirs(state_name.lower())
        os.chdir(root_path + '/' + state_name.lower())
    else:
        os.chdir(root_path + '/' + state_name.lower())

    state_name = state_name.upper()
    state_number = get_state_number(state_name, state_mapping)

    download_cdl_state(cdl_year, state_number, state_mapping)

    generate_bbox(cdl_year, state_name)

    

if __name__ == "__main__":
    state_name = sys.argv[1]
    main(state_name)





