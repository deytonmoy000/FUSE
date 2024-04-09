
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

import collections
from datetime import datetime, timedelta
import random
import json
import h5py

import concurrent.futures
import os

os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1"
from queue import PriorityQueue
from collections import defaultdict
nDSV = 500
dsv_loc_val = -1000 # value to signify current loc of dsv
invalid_loc_val = -((2**31)-1)
max_valid_value = 720
size_dict = {1: [1, 1], 2: [1, 2], 4:[2, 2], 8: [2, 4], 16: [4, 4], 32:[4, 8]}
gridSize_info_dict = {3000:(13, 30, 390, 13, 28, 364), 1000:(41, 91, 3731, 39, 84, 3276), 500:(83, 182, 15106, 78, 168, 13104)}

ALPHA = 0.2
BETA = 2
GAMMA = 1
cur_min_LONG = 113.735223
cur_max_LONG = 114.649835
cur_min_LAT = 22.426765
cur_max_LAT = 22.845040


def gen_granular_data(grid_size):
    print("BEGINNING DATA Generation...")

    grid_size /= 1000
    block_size = 0.01  # 1 km in degrees
    block_size *= grid_size

    n_rows = int((cur_max_LAT - cur_min_LAT)//block_size)
    n_cols = int((cur_max_LONG - cur_min_LONG)//block_size)

    df = pd.read_csv('data/data_new.csv')

    df = df[~((df['Long_id'] > 83) | (df['Lat_id'] > 38))]

    df['Long_id'] = ((df['Longitude'] - cur_min_LONG) / block_size).astype(int)
    df['Lat_id'] = ((cur_max_LAT - df['Latitude']) / block_size).astype(int)

    df['Position'] = df['Lat_id']*n_cols + df['Long_id']

    fname = f'data/data_new_{str(int(grid_size*1000))}m.csv'
    df.to_csv(fname, header=True, index=False)

    n_rows_valid, n_cols_valid = df.Lat_id.max()+1, df.Long_id.max()+1

    print("SAVED DATAFRAME TO ", fname)
    print(n_rows, n_cols, n_rows_valid, n_cols_valid)
    return n_rows, n_cols, n_rows_valid, n_cols_valid


def update_grids_info(df, grid_size):
    block_size = 0.01 # 1km
    block_size *= grid_size/1000

    n_rows, n_cols = int((cur_max_LAT - cur_min_LAT) // block_size), int((cur_max_LONG - cur_min_LONG) // block_size)
    nPosns = n_cols * n_rows

    n_rows_valid, n_cols_valid = df.Lat_id.max()+1, df.Long_id.max()+1
    nPosns_valid = n_rows_valid * n_cols_valid

    grid_size_fname = str(grid_size)+'m'
    print(f"n_rows:{n_rows}, n_cols:{n_cols};  n_rows_valid:{n_rows_valid}, n_cols_valid:{n_cols_valid}")

    return n_rows, n_cols, nPosns, n_rows_valid, n_cols_valid, nPosns_valid


def create_grid_json_updated(grid_size=1):
    grid_size /= 1000
    block_size = 0.01  # 1 km in degrees
    block_size *= grid_size
    min_lat = cur_min_LAT
    min_lon = cur_min_LONG
    max_lat = min_lat + block_size
    max_lon = min_lon + block_size
    id_counter = 1
    polygons = []
    while min_lat < cur_max_LAT:
        while min_lon < cur_max_LONG:
            polygon_coords = [
                (min_lon, min_lat),
                (min_lon, max_lat),
                (max_lon, max_lat),
                (max_lon, min_lat)
            ]
            polygon = {
                "geo_array": polygon_coords,
                "get_id": id_counter
            }
            polygons.append(polygon)
            min_lon = max_lon
            max_lon = min_lon + block_size
            id_counter += 1
        min_lon = cur_min_LONG
        max_lon = min_lon + block_size
        min_lat = max_lat
        max_lat = min_lat + block_size
    map_data = {
        "out_edge": polygons,
        "minmax": {
            "min_x": cur_min_LAT,
            "min_y": cur_min_LONG,
            "max_x": cur_max_LAT,
            "max_y": cur_max_LONG
        }
    }
    map_json = json.dumps(map_data, indent=4)
    file_path = f"data/shenzen_map_{str(int(grid_size*1000))}m.json"
    with open(file_path, "w") as json_file:
        json.dump(map_data, json_file, indent=4)

    print(f"Map data has been saved to {file_path}")


def get_valid_grid(df, grid_size):
    grid_size *= 1000
    fname = f'data/data_new_{str(int(grid_size))}m.csv'

    df = pd.read_csv(fname)
    subset_df = df[['Long_id', 'Lat_id']]
    unique_combinations = subset_df.drop_duplicates()

    visited_cells = set(map(tuple, subset_df.values))
    fname_valid_grids = f'data/map/valid_grids_{str(int(grid_size))}m.txt'
    with open(fname_valid_grids, "w") as file:
        for cell in visited_cells:
            x, y = cell
            line = str(x)+','+str(y)+'\n'
            file.write(line)
    file.close()
    return visited_cells


def csv_to_grid_updated(grid_size):

    grid_size /= 1000
    block_size = 0.01  # 1 km in degrees
    block_size *= grid_size

    fname = f'data/data_new_{str(int(grid_size*1000))}m.csv'

    fname_json = f"data/shenzen_map_{str(int(grid_size*1000))}m.json"
    df = pd.read_csv(fname)

    n_rows, n_cols, nPosns, n_rows_valid, n_cols_valid, nPosns_valid = update_grids_info(df, grid_size*1000)

    print(f"csv_to_grid_updated --> n_rows:{n_rows}, n_cols:{n_cols};  n_rows_valid:{n_rows_valid}, n_cols_valid:{n_cols_valid}")
    print("DATAFRAME LOADED FROM CSV")
    print(df.describe())
    print(df.head())
    with open(fname_json, "r") as json_file:
        map_data = json.load(json_file)
    map_image = Image.open("figs/shenzen-map_new.png")
    x_scale = map_image.width / (cur_max_LONG - cur_min_LONG)
    y_scale = map_image.height / (cur_max_LAT - cur_min_LAT)
    grid_ids = {}
    max_LONG_id, max_LAT_id = 0, 0
    max_LONG_coord, max_LAT_coord = cur_min_LONG, cur_min_LAT
    final_max_LONG, final_max_LAT = 0, 0

    for polygon in map_data["out_edge"]:
        geo_array = polygon["geo_array"]
        min_LONG, min_LAT = (geo_array[0][0] - cur_min_LONG) * x_scale, (cur_max_LAT - geo_array[0][1]) * y_scale
        max_LONG, max_LAT = (geo_array[2][0] - cur_min_LONG) * x_scale, (cur_max_LAT - geo_array[2][1]) * y_scale
        Long_id, Lat_id = (geo_array[0][0] - cur_min_LONG)/block_size, (cur_max_LAT - geo_array[0][1])/block_size
        Long_id, Lat_id = int(Long_id), int(Lat_id)
        grid_ids[(Long_id, Lat_id)] = [[geo_array[0][0], geo_array[2][0], geo_array[0][1], geo_array[2][1]], [min_LONG, max_LONG, min_LAT, max_LAT]]
        if Long_id > max_LONG_id:
            max_LONG_id = Long_id
            max_LONG_coord = geo_array[2][0]
            final_max_LONG = max_LONG
        if Lat_id > max_LAT_id:
            max_LAT_id = Lat_id
            max_LAT_coord = geo_array[2][1]
            final_max_LAT = max_LAT
    fname_grids = f'data/map/grids_new_{str(int(grid_size*1000))}m.txt'
    with open(fname_grids, "w") as file:
        for key, val in grid_ids.items():
            x, y = key
            x1, x2, y1, y2 = val[0]
            cx1, cx2, cy1, cy2 = val[1]
            line = [x, y, cx1, cx2, cy1, cy2, x1, x2, y1, y2]
            line = ','.join(map(str, line)) + '\n'
            file.write(line)
    file.close()

    print("GRIDS CREATED -- MAX LONG ID: {}({})  -- MAX LAT ID: {}({})".format(max_LONG_id, max_LONG_coord, max_LAT_id, max_LAT_coord))
    print("MAPPING VEHICLE LOCATIONS...")

    visited_cells = get_valid_grid(df, grid_size)
    print("Generated Visited Cells Len():  ", len(visited_cells), " GRIDS")
    print(visited_cells)

    cells_to_remove = set()

    directions = [[1, 0], [0,1], [-1, 0], [0, -1]]
    for cell in visited_cells:
        flag = 0
        x, y = cell

        for dx, dy in directions:
            if (x+dx >= 0 and x+dx <= n_cols_valid) and (y+dy >= 0 and y+dy <= n_rows_valid) and ((x+dx, y+dy) in visited_cells):
                flag += 1
                if flag > 1:
                    break
        if x == 15 and y == 39:
            print(flag, cell)
        if flag < 2:
            cells_to_remove.add((x, y)) 
    for cell in cells_to_remove:
        visited_cells.remove(cell)

    print("REMOVED ", len(cells_to_remove), " ISOLATED GRIDS")

    invalid_grids = set()
    for key, val in grid_ids.items():
        x, y = key
        if (x, y) not in visited_cells:
            invalid_grids.add((x, y))
    fname_valid_grids = f'data/map/valid_grids_{str(int(grid_size*1000))}m.txt'
    with open(fname_valid_grids, "w") as file:
        for cell in visited_cells:
            x, y = cell
            line = str(x)+','+str(y)+'\n'
            file.write(line)
    file.close()

    print(f"UPDATED VISITED GRID ({fname_valid_grids}) FILE")

    fname_invalid_grids = f'data/map/invalid_grids_{str(int(grid_size*1000))}m.txt'

    with open(fname_invalid_grids, "w") as file:
        for cell in invalid_grids:
            x, y = cell
            line = str(x)+','+str(y)+'\n'
            file.write(line)
    file.close()
    print(f"UPDATED INVALID GRID ({fname_invalid_grids}) FILE")


def grid_to_map_updated(grid_size):

    map_image = Image.open("figs/shenzen-map_new.png")
    fig, ax = plt.subplots(1, figsize=(16, 16))  # Adjust the figure size for a larger image
    ax.imshow(map_image)
    final_max_LONG, final_max_LAT = 0, 0
    max_LONG_id, max_LAT_id = 0, 0
    final_max_LONG_coord, final_max_LAT_coord = cur_min_LONG, cur_min_LAT

    grid_ids = {}
    fname_grids = f'data/map/grids_new_{str(int(grid_size))}m.txt'
    with open(fname_grids, "r") as file:
        for line in file:
            lon_id, lat_id, min_LONG_coord, max_LONG_coord, min_LAT_coord, max_LAT_coord, min_LONG, max_LONG, min_LAT, max_LAT  = line.split(',')

            val2 = [min_LONG_coord, max_LONG_coord, min_LAT_coord, max_LAT_coord]
            val1 = [min_LONG, max_LONG, min_LAT, max_LAT]

            val1 = list(map(float, val1))
            val2 = list(map(float, val2))

            grid_ids[int(lon_id), (int(lat_id))] = [val1, val2]

    file.close()
    for key, value in grid_ids.items():
        min_LONG, max_LONG, min_LAT, max_LAT = value[1]
        min_LONG_coord, max_LONG_coord, min_LAT_coord, max_LAT_coord = value[0]
        Long_id, Lat_id = key
        if Long_id > max_LONG_id:
            final_max_LONG = max_LONG
            max_LONG_id = Long_id
            final_max_LONG_coord = max_LONG_coord
        if Lat_id > max_LAT_id:
            final_max_LAT = max_LAT
            max_LAT_id = Lat_id
            final_max_LAT_coord = max_LAT_coord

        grid_rect = patches.Rectangle(
            (min_LONG, min_LAT),  # (x, y)
            max_LONG - min_LONG,  # width
            max_LAT - min_LAT,  # height
            linewidth=0.2,
            edgecolor='k',  # red color
            facecolor='none'
        )
        ax.add_patch(grid_rect)

    print("GRIDS CREATED -- MAX LONG ID: {}({})  -- MAX LAT ID: {}({})".format(max_LONG_id, final_max_LONG_coord, max_LAT_id, final_max_LAT_coord))
    visited_cells = set()
    fname_valid_grids = f'data/map/valid_grids_{str(int(grid_size))}m.txt'
    with open(fname_valid_grids, "r") as file:
        for line in file:
            x, y = map(int, line.split(','))
            visited_cells.add((x, y))
    file.close()
    for cell in visited_cells:
        grid_x, grid_y = cell
        grid_values = grid_ids[(grid_x, grid_y)]
        min_LONG, max_LONG, min_LAT, max_LAT = grid_values[1]

        min_LONG_coord, max_LONG_coord, min_LAT_coord, max_LAT_coord = grid_values[0]

        visited_grid_rect = patches.Rectangle(
            (min_LONG, min_LAT),  # (x, y)
            max_LONG - min_LONG,  # width
            max_LAT - min_LAT,  # height
            linewidth=0.5,
            edgecolor='k',  # red color
            facecolor='g'
        )
        ax.add_patch(visited_grid_rect)
    ax.set_aspect('equal')
    x_axis = list(range(0, np.ceil(final_max_LONG).astype(int), 100))
    y_axis = list(range(0, np.ceil(final_max_LAT).astype(int), 100))
    block_size = 0.01 # 1km
    block_size *= grid_size/1000

    n_rows, n_cols = int((cur_max_LAT - cur_min_LAT) // block_size), int((cur_max_LONG - cur_min_LONG) // block_size)
    conversion_factor = [n_cols/final_max_LONG, n_rows/final_max_LAT]
    desired_x_axis = [str(int(x * conversion_factor[0])) for x in x_axis]
    desired_y_axis = [str(int(y * conversion_factor[1])) for y in y_axis]
    plt.xticks(x_axis, desired_x_axis)
    plt.yticks(y_axis, desired_y_axis)

    image_fname = f"figs_paper/map_with_visited_grid_cells_{str(int(grid_size))}m.pdf"

    plt.savefig(image_fname, dpi=300)  # Adjust the DPI for a higher resolution image

def angle_to_direction(angle):
    if -180 <= angle <= 180:
        directions = ['North', 'NorthEast', 'East', 'SouthEast', 'South', 'SouthWest', 'West', 'NorthWest']
        angle = (int(angle) + 360) % 360  # Ensure angle is in the range [0, 360)
        index = int(int((angle + 22.5)) % 360 // 45)
        return directions[index]
    return None


def prepare_dataframe_coord(data, n_steps=5):
    data = data.copy()
    data = data.sort_values(by=['ID', 'rounded_Time'])
    vids = data.ID.unique()

    modified_data = pd.DataFrame()

    for id in range(TaxiNewtmp.ID.unique().shape[0]):
        group = data[data['ID'] == id]
        column_name = f'Speed(t-{1})'
        group[column_name] = group['Speed'].shift(1)
        column_name = f'direction(t-{1})'
        group[column_name] = np.arctan2(group['Longitude'] - group['Longitude'].shift(1), group['Latitude'] - group['Latitude'].shift(1))*(180/np.pi)
        column_name = f'TimeDiff(t-{1})'
        group[column_name] = (group['rounded_Time'] - group['rounded_Time'].shift(1))/ pd.Timedelta(minutes=1)
        group['Direction_Code'] = group['direction'].apply(angle_to_direction)
        for i in range(1, n_steps+1):
            column_name = f'Longitude(t-{i})'
            group[column_name] = group['Longitude'].shift(i)
        for i in range(1, n_steps+1):
            column_name = f'Latitude(t-{i})'
            group[column_name] = group['Latitude'].shift(i)
        modified_data = pd.concat([modified_data, group])
    modified_data.dropna(inplace=True)

    return modified_data


def prepare_dataframe_coordID(data, n_steps=5):
    data = data.copy()
    data = data.sort_values(by=['ID', 'rounded_Time'])
    vids = data.ID.unique()

    modified_data = pd.DataFrame()

    for id in range(TaxiNewtmp.ID.unique().shape[0]):
        group = data[data['ID'] == id]
        column_name = f'Speed(t-{1})'
        group[column_name] = group['Speed'].shift(1)
        column_name = f'direction(t-{1})'
        group[column_name] = np.arctan2(group['Longitude'] - group['Longitude'].shift(1), group['Latitude'] - group['Latitude'].shift(1))*(180/np.pi)
        column_name = f'TimeDiff(t-{1})'
        group[column_name] = (group['rounded_Time'] - group['rounded_Time'].shift(1))/ pd.Timedelta(minutes=1)
        for i in range(1, n_steps+1):
            column_name = f'LongIDX(t-{i})'
            group[column_name] = group['LongIDX'].shift(i)
            group[column_name] = pd.to_numeric(group[column_name])
        for i in range(1, n_steps+1):
            column_name = f'LatIDX(t-{i})'
            group[column_name] = group['LatIDX'].shift(i)
            group[column_name] = pd.to_numeric(group[column_name])

        modified_data = pd.concat([modified_data, group])
    modified_data.dropna(inplace=True)

    return modified_data

def create_grid_json(grid_size=1):
    block_size = 0.01  # 1 km in degrees
    block_size *= grid_size
    min_lat = cur_min_LAT
    min_lon = cur_min_LONG
    max_lat = min_lat + block_size
    max_lon = min_lon + block_size
    id_counter = 1
    polygons = []
    while min_lat < cur_max_LAT:
        while min_lon < cur_min_LONG:

            polygon_coords = [
                (min_lat, min_lon),
                (min_lat, max_lon),
                (max_lat, max_lon),
                (max_lat, min_lon)
            ]
            polygon = {
                "geo_array": polygon_coords,
                "get_id": id_counter
            }
            polygons.append(polygon)
            min_lon = max_lon
            max_lon = min_lon + block_size
            id_counter += 1
        min_lon = cur_min_LONG
        max_lon = min_lon + block_size
        min_lat = max_lat
        max_lat = min_lat + block_size
    map_data = {
        "out_edge": polygons,
        "minmax": {
            "min_x": cur_min_LAT,
            "min_y": cur_min_LONG,
            "max_x": cur_max_LAT,
            "max_y": cur_max_LONG
        }
    }
    map_json = json.dumps(map_data, indent=4)
    file_path = f"data/shenzen_map_{grid_size*1000}m.json"
    with open(file_path, "w") as json_file:
        json.dump(map_data, json_file, indent=4)

    print(f"Map data has been saved to {file_path}")


def raw_data_preprocessing():
    print("BEGINNING DATA PROCESSING...")

    EVdata = pd.read_csv('data/EVData.csv')
    EVdata['id'] = EVdata['id'] + 10e7

    BusData = pd.read_csv('data/BusData.csv', header=None)
    TaxiData = pd.read_csv('data/TaxiData.csv', header=None)
    TruckData = pd.read_csv('data/TruckData.csv', header=None)

    print("LOADED ALL VEHICLES DATA")

    EVdata.columns = ['VID', 'Latitude', 'Longitude', 'Time', 'Speed']
    TaxiData.columns = ['VID', 'Time', 'Longitude', 'Latitude', 'OccupancyStatus', 'Speed']
    BusData.columns = ['VID', 'Time', 'PlateID', 'Longitude', 'Latitude', 'Speed']
    TruckData.columns = ['VID', 'Time', 'Longitude', 'Latitude', 'Speed']
    TaxiData = TaxiData.drop_duplicates()
    BusData = BusData.drop_duplicates()
    EVdata = EVdata.drop_duplicates()
    TruckData = TruckData.drop_duplicates()
    BusData['PlateID'] = BusData['PlateID'].str[2:]
    BusData = BusData.drop(columns=['PlateID'])
    TaxiData = TaxiData.drop(columns=['OccupancyStatus'])
    desired_order = ['VID', 'Time', 'Longitude', 'Latitude', 'Speed']
    EVdata = EVdata[desired_order]

    EVdata['Type'] = 'E'
    TaxiData['Type'] = 'F'
    BusData['Type'] = 'B'
    TruckData['Type'] = 'T'

    frames = [EVdata, BusData, TaxiData, TruckData]

    for i in range(len(frames)):
        if i < len(frames)-1:
            frames[i]['Time'] = pd.to_datetime(frames[i]['Time'])
            frames[i]['Timestamp'] = pd.to_datetime(frames[i]['Time'].dt.strftime('%Y-%m-%d %H:%M'))
            frames[i] = frames[i].sort_values(by=['Timestamp', 'VID'])
            frames[i]['Date'] = frames[i].groupby(['VID', frames[i]['Timestamp']])['Time'].transform(lambda x: pd.date_range('2023-01-15', periods=len(x), freq='D'))
            frames[i]['Time_New'] = frames[i]['Date'].dt.strftime('%Y-%m-%d') + ' ' + frames[i]['Timestamp'].dt.strftime('%H:%M')
            frames[i]['Time_New'] = pd.to_datetime(frames[i]['Time_New'], format='%Y-%m-%d %H:%M')

        else:
            frames[i]['Time'] = pd.to_datetime(frames[i]['Time'].str.replace(r'^\*\*\*\*\-\*\*', '2023-01', regex=True), format='%Y-%m-%d %H:%M:%S')
            frames[i]['Timestamp'] = pd.to_datetime(frames[i]['Time'].dt.strftime('%Y-%m-%d %H:%M'), format='%Y-%m-%d %H:%M')
            frames[i]['Date'] = pd.to_datetime(frames[i]['Timestamp'].dt.strftime('%Y-%m-%d'))
            frames[i]['Time_New'] = frames[i]['Timestamp']
            frames[i] = frames[i].drop_duplicates(subset=['Time_New', 'VID'], keep='last')

        frames[i] = frames[i][(frames[i]['Time_New'] < "2023-01-22")]
        print(i, frames[i].groupby(['Date']).size().reset_index(name='Number_of_Rows'))
        frames[i].drop(columns=['Timestamp', 'Date'])
        desired_order = ['VID', 'Type', 'Time', 'Time_New', 'Longitude', 'Latitude', 'Speed']
        frames[i] = frames[i][desired_order]

    print("DATA FORMATTED")
    df = pd.concat(frames).reset_index()

    print("DATA COMBINED")
    df = df[~((df['Longitude'] < cur_min_LONG) | (df['Longitude'] > cur_max_LONG))]
    df = df[~((df['Latitude'] < cur_min_LAT) | (df['Latitude'] > cur_max_LAT))]
    import re

    def extract_time(datetime_str):
        match = re.search(r'\d{2}:\d{2}:\d{2}', datetime_str)
        if match:
            return match.group()
        return None
    df['ID'] = df['VID'].rank(method='dense').astype(int) - 1
    df['Long_id'] = ((df['Longitude'] - cur_min_LONG) / 0.01).astype(int)
    df['Lat_id'] = ((cur_max_LAT - df['Latitude']) / 0.01).astype(int)
    df['Position'] = df['Lat_id']*n_cols + df['Long_id']

    df['VID'] = df['VID'].astype(int)
    df['ID'] = df['ID'].astype(int)
    print("DATA CLEANED PART I")
    desired_order = ['ID','VID', 'Type', 'Time', 'Time_New', 'Speed', 'Longitude', 'Latitude', 'Long_id', 'Lat_id', 'Position']
    df = df[desired_order]
    df.columns = ['ID','VID', 'Type', 'Time','Time_New', 'Speed', 'Longitude', 'Latitude', 'Long_id', 'Lat_id', 'Position']

    df = df[~((df['Long_id'] < 9) & (df['Lat_id'] > 25))]
    df = df[~(((df['Long_id'] < 11) & (df['Lat_id'] > 27)) | ((df['Long_id'] > 21) & (df['Lat_id'] >= 38)))]
    df = df[~((df['Long_id'] > 53) & (df['Long_id'] < 62) & (df['Lat_id'] >= 28))]

    print("DATA CLEANED PART II")

    print(df.describe())
    print(df.head(20))
    print(df.info())
    df.to_csv('data/data.csv', header=True, index=False)

    print("SAVED DATAFRAME TO data/data.csv")

def raw_data_preprocessing_old():
    print("BEGINNING DATA PROCESSING...")

    EVdata = pd.read_csv('data/EVData.csv')
    EVdata['id'] = EVdata['id'] + 10e7

    BusData = pd.read_csv('data/BusData.csv', header=None)
    TaxiData = pd.read_csv('data/TaxiData.csv', header=None)
    TruckData = pd.read_csv('data/TruckData.csv', header=None)

    print("LOADED ALL VEHICLES DATA")

    EVdata.columns = ['VID', 'Latitude', 'Longitude', 'Time', 'Speed']
    TaxiData.columns = ['VID', 'Time', 'Longitude', 'Latitude', 'OccupancyStatus', 'Speed']
    BusData.columns = ['VID', 'Time', 'PlateID', 'Longitude', 'Latitude', 'Speed']
    TruckData.columns = ['VID', 'Time', 'Longitude', 'Latitude', 'Speed']

    TaxiData = TaxiData.drop_duplicates()
    BusData = BusData.drop_duplicates()
    EVdata = EVdata.drop_duplicates()
    TruckData = TruckData.drop_duplicates()
    BusData['PlateID'] = BusData['PlateID'].str[2:]
    BusData = BusData.drop(columns=['PlateID'])
    TaxiData = TaxiData.drop(columns=['OccupancyStatus'])
    desired_order = ['VID', 'Time', 'Longitude', 'Latitude', 'Speed']
    EVdata = EVdata[desired_order]

    EVdata['Type'] = 'E'
    TaxiData['Type'] = 'F'
    BusData['Type'] = 'B'
    TruckData['Type'] = 'T'

    print("DATA FORMATTED")

    frames = [EVdata, BusData, TaxiData, TruckData]

    df = pd.concat(frames).reset_index()

    print("DATA COMBINED")
    df = df[~((df['Longitude'] < cur_min_LONG) | (df['Longitude'] > cur_max_LONG))]
    df = df[~((df['Latitude'] < cur_min_LAT) | (df['Latitude'] > cur_max_LAT))]
    import re

    def extract_time(datetime_str):
        match = re.search(r'\d{2}:\d{2}:\d{2}', datetime_str)
        if match:
            return match.group()
        return None
    df['Time'] = df['Time'].apply(extract_time)

    df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S')
    df['rounded_Time'] = df['Time'].dt.strftime('%H:%M')
    df['ID'] = df['VID'].rank(method='dense').astype(int) - 1
    df['Long_id'] = ((df['Longitude'] - cur_min_LONG) / 0.01).astype(int)
    df['Lat_id'] = ((cur_max_LAT - df['Latitude']) / 0.01).astype(int)
    df['Position'] = df['Lat_id']*91 + df['Long_id']

    df['VID'] = df['VID'].astype(int)
    df['ID'] = df['ID'].astype(int)
    df['Time'] = pd.to_datetime('2023-01-01 ' + df['rounded_Time'])

    print("DATA CLEANED PART I")

    df = df.drop_duplicates(subset=['Time', 'VID'], keep='last')
    print(df.head(10))
    print(df.describe())
    df.drop(columns=['Speed', 'rounded_Time'])
    desired_order = ['ID','VID', 'Type', 'Time','Longitude', 'Latitude', 'Long_id', 'Lat_id', 'Position']
    df = df[desired_order]
    df.columns = ['ID','VID', 'Type', 'Time','Longitude', 'Latitude', 'Long_id', 'Lat_id', 'Position']

    df = df[~((df['Long_id'] < 9) & (df['Lat_id'] > 25))]
    df = df[~(((df['Long_id'] < 11) & (df['Lat_id'] > 27)) | ((df['Long_id'] > 21) & (df['Lat_id'] >= 38)))]
    df = df[~((df['Long_id'] > 53) & (df['Long_id'] < 62) & (df['Lat_id'] >= 28))]

    print("DATA CLEANED PART II")

    print(df.describe())
    print(df.head(20))

    df.to_csv('data/data_new.csv', header=True, index=False)

    print("SAVED DATAFRAME TO data/data_new.csv")


def save_df(df, fname):
    df.to_csv(fname, header=True, index=False)

def path_grid_map(grid_size, path, alg, alg_name, cov, t, start_locs, nDiv, seed, tf, km_time):

    map_image = Image.open("figs/shenzen-map_new.png")

    n_rows, n_cols, nPosns, n_rows_valid, n_cols_valid, nPosns_valid = gridSize_info_dict[grid_size] 
    fig, ax = plt.subplots(1, figsize=(8, 8))  # Adjust the figure size for a larger image
    ax.imshow(map_image)
    final_max_LONG, final_max_LAT = 0, 0
    max_LONG_id, max_LAT_id = 0, 0
    final_max_LONG_coord, final_max_LAT_coord = cur_min_LONG, cur_min_LAT

    grid_ids = {}
    fnamex = f'data/map/grids_new_{str(int(grid_size))}m.txt'
    with open(fnamex, "r") as file:
        for line in file:
            lon_id, lat_id, min_LONG_coord, max_LONG_coord, min_LAT_coord, max_LAT_coord, min_LONG, max_LONG, min_LAT, max_LAT  = line.split(',')

            val2 = [min_LONG_coord, max_LONG_coord, min_LAT_coord, max_LAT_coord]
            val1 = [min_LONG, max_LONG, min_LAT, max_LAT]

            val1 = list(map(float, val1))
            val2 = list(map(float, val2))

            grid_ids[int(lon_id), (int(lat_id))] = [val1, val2]

    file.close()
    for key, value in grid_ids.items():
        min_LONG, max_LONG, min_LAT, max_LAT = value[1]
        min_LONG_coord, max_LONG_coord, min_LAT_coord, max_LAT_coord = value[0]
        Long_id, Lat_id = key
        if Long_id > max_LONG_id:
            final_max_LONG = max_LONG
            max_LONG_id = Long_id
            final_max_LONG_coord = max_LONG_coord
        if Lat_id > max_LAT_id:
            final_max_LAT = max_LAT
            max_LAT_id = Lat_id
            final_max_LAT_coord = max_LAT_coord

        grid_rect = patches.Rectangle(
            (min_LONG, min_LAT),  # (x, y)
            max_LONG - min_LONG,  # width
            max_LAT - min_LAT,  # height
            linewidth=0.2,
            edgecolor='k',  # red color
            facecolor='none'
        )
        ax.add_patch(grid_rect)
    visited_cells = set()
    for i, pos in enumerate(cov):
        lat, lon = pos//n_cols, pos%n_cols
        visited_cells.add((lon, lat))

    for cell in visited_cells:
        grid_x, grid_y = cell
        grid_values = grid_ids[(grid_x, grid_y)]
        min_LONG, max_LONG, min_LAT, max_LAT = grid_values[1]

        min_LONG_coord, max_LONG_coord, min_LAT_coord, max_LAT_coord = grid_values[0]

        visited_grid_rect = patches.Rectangle(
            (min_LONG, min_LAT),  # (x, y)
            max_LONG - min_LONG,  # width
            max_LAT - min_LAT,  # height
            linewidth=0.5,
            edgecolor='k',  # red color
            facecolor='y'
        )
        ax.add_patch(visited_grid_rect)
    visited_cells = set()
    start_posns = set()
    for i, pos in enumerate(path):
        if not isinstance(pos, int):
            pos = pos.item()
        lat, lon = pos//n_cols, pos%n_cols
        visited_cells.add((lon, lat))
        if pos in start_locs:
            start_posns.add((lon, lat))
    if alg[:5] == 'FAS3T':
        c = 'k'
    elif alg[:28] == 'TSMTC-C':
        c = 'g'
    elif alg[:19] == 'REASSIGN-F':
        c = 'r'
    elif alg[:15] == 'AGD-C':
        c = 'orange'
    elif alg[:9] == 'SDPR-F':
        c = 'b'
    elif alg[:4] == 'fuse':
        c = 'k'
    else:
        c = 'y'

    started = False
    for cell in visited_cells:
        grid_x, grid_y = cell
        grid_values = grid_ids[(grid_x, grid_y)]
        min_LONG, max_LONG, min_LAT, max_LAT = grid_values[1]

        min_LONG_coord, max_LONG_coord, min_LAT_coord, max_LAT_coord = grid_values[0]

        if (grid_x, grid_y) not in start_posns:
            visited_grid_rect = patches.Rectangle(
                (min_LONG, min_LAT),  # (x, y)
                max_LONG - min_LONG,  # width
                max_LAT - min_LAT,  # height
                linewidth=0.5,
                edgecolor='k',  # red color
                facecolor=c
            )
        else:
            visited_grid_rect = patches.Rectangle(
                (min_LONG, min_LAT),  # (x, y)
                max_LONG - min_LONG,  # width
                max_LAT - min_LAT,  # height
                linewidth=0.5,
                edgecolor='k',  # red color
                facecolor='m'
            )
        ax.add_patch(visited_grid_rect)
    ax.set_aspect('equal')
    x_axis = list(range(0, np.ceil(final_max_LONG).astype(int), 100))
    y_axis = list(range(0, np.ceil(final_max_LAT).astype(int), 100))
    conversion_factor = [90/final_max_LONG, 41/final_max_LAT]
    desired_x_axis = [str(int(x * conversion_factor[0])) for x in x_axis]
    desired_y_axis = [str(int(y * conversion_factor[1])) for y in y_axis]
    plt.xticks(x_axis, desired_x_axis)
    plt.yticks(y_axis, desired_y_axis)
    fname = f"figs_paper/GS_{int(grid_size)}m/TF_"+str(tf)+"/Div_"+str(nDiv)+"/Rep_"+str(seed)+"/"+alg_name+"/path_grid_"+str(tf)+"_"+alg+"_hour"+str(t)+'_'+str(nDiv)+"divs" + '_'+str(seed)+".png"
    plt.savefig(fname, dpi=300)  # Adjust the DPI for a higher resolution image
    plt.close()
    return fname

def read_data(fname, start_date, end_date):

    df = pd.read_csv(fname)
    df['Time'] = pd.to_datetime(df['Time'])
    df = df[((df['Time'] >= start_date) & (df['Time'] <= end_date))]

    print("Data Loaded from {};  Number of Rows: {}".format(fname, len(df)))
    return df

def combine_results(seed=-1):
    if seed == -1:
        seeds = ["42", "14", "25", "8", "35"]
    else:
        seeds = [str(seed)]

    algs = ["FAS3T", "GreedyCov_wPredFull_wHistory", "GreedyCov_wPredFull", "GreedyCov_wPred", "GreedyCov"]
    nDivs = ["8", "4", "2", "1"]

    df_all = pd.DataFrame()
    df_seed = pd.DataFrame()

    for seed in seeds:
        for alg in algs:
            for nDiv in nDivs:
                fname = 'result/metrics_'+str(alg)+'_'+str(nDiv)+'divs' + '_'+str(seed)+'.csv'
                print(fname, os.path.isfile(fname))
                if os.path.isfile(fname):
                    data = pd.read_csv(fname)
                    data['seed'] = seed
                    data['nDiv'] = nDiv
                    df_seed = pd.concat([df_seed, data]) #, ignore_index=True)
        comb_fname = 'result/combined_metrics_'+str(seed)+'.csv'
        save_df(df_seed, comb_fname)
        print("\nRandom (seed=", seed,") results data saved to ", comb_fname )

        df_all = pd.concat([df_all, df_seed]) #, ignore_index=True)
        df_seed = pd.DataFrame()

    all_fname = 'result/all_result_metrics.csv'
    save_df(df_all, all_fname)
    print("\nAll results data saved to ", all_fname )

def generate_neighbors_list(grid_size):

    visited_cells = set()
    visited_locs = set()
    with open(f'data/map/valid_grids_{str(int(grid_size))}m.txt', "r") as file:
        for line in file:
            lon, lat = map(int, line.split(','))
            visited_cells.add((lon, lat))
            pos = n_cols*lat + lon
            visited_locs.add(pos)
    file.close()
    neighbors = collections.defaultdict(list)

    directions = [[-1, -1], [0, -1], [1, -1], [-1, 0], [1, 0], [-1, 1], [0, 1], [1, 1]]

    for cell in visited_cells:
        lon, lat = map(int, cell)
        pos = n_cols*lat + lon

        if pos not in neighbors:
            for dlon,dlat in directions:
                posLon, posLat = lon + dlon, lat + dlat
                posNei = n_cols*posLat + posLon
                if (posLon < 0 or posLon >= n_cols) or (posLat < 0 or posLat >= 41) or (posNei not in visited_locs):
                    continue
                neighbors[pos].append(posNei)

    sorted_neighbors = {key: neighbors[key] for key in sorted(neighbors)}

    with open('data/valid_grids_neighbors.txt', 'w') as file:
        for key, val in sorted_neighbors.items():
            line = str(key) + '|' + ','.join(map(str, val)) + '\n'
            file.write(line)
    file.close()

def generate_neighbors_list_wBorders(n, grid_size, df, alg, tf, km_time):

    n_rows, n_cols, nPosns, n_rows_valid, n_cols_valid, nPosns_valid = gridSize_info_dict[grid_size] 
    print(f"generate_neighbors_list_wBorders --> n_rows:{n_rows}, n_cols:{n_cols};  n_rows_valid:{n_rows_valid}, n_cols_valid:{n_cols_valid}")

    ts = tf//km_time
    tfms = [] # For FUSE DQN
    valid_cells = set()
    valid_locs = set()
    with open(f'data/map/valid_grids_{str(int(grid_size))}m.txt', "r") as file:
        for line in file:
            lon, lat = map(int, line.split(','))
            valid_cells.add((lat, lon))
            pos = n_cols*lat + lon
            valid_locs.add(pos)
    file.close()
    div_coords = {}

    neighbors = collections.defaultdict(list)
    print('NDIV PASSED:', n)
    directions = [[-1, -1], [0, -1], [1, -1], [-1, 0], [1, 0], [-1, 1], [0, 1], [1, 1]]
    area_nrows, area_ncols = size_dict[n]
    area_width, area_height = n_cols_valid//area_ncols, n_rows_valid//area_nrows
    if alg=="FAS3T":
        mat = [invalid_loc_val for i in range(nPosns)]
    elif alg[:4]=="fuse":

        mat = get_default_mat2D_DQN(grid_size, df, tf, km_time)
    elif alg == "AGD-C":
        mat, _ = get_default_mat2D_coverage(grid_size, df, ts)
    elif alg[-1]=="F":
        mat = get_default_mat2D_DQN(grid_size, df, tf, km_time)

    else:
        mat = [(2**31)-1 for i in range(nPosns)]
    valid_div_pos = []
    pos_to_div = {}

    print("\nDivisions:")
    for i in range(area_nrows):
        for j in range(area_ncols):
            div_pos = set()
            div_id = i*area_ncols + j
            start_row, end_row = i*area_height, (i+1)*area_height
            start_col, end_col = j*area_width, (j+1)*area_width

            if j == area_ncols-1: 
                end_col = n_cols_valid
            if i == area_nrows-1:
                end_row = n_rows_valid
            for lat in range(start_row, end_row):
                for lon in range(start_col, end_col):
                    if (lat, lon) in valid_cells:
                        pos = n_cols*lat + lon
                        if alg[:4]=="fuse":

                            pos_to_div[(lat, lon)] = [div_id, lat - start_row + ts, lon - start_col + ts] # [division id, relative position in div]
                        elif alg[-1]=="F":
                            pos_to_div[(lat, lon)] = [div_id, lat - start_row + ts, lon - start_col + ts]
                        elif alg=="AGD-C":
                            pos_to_div[(lat, lon)] = [div_id, lat - start_row + ts, lon - start_col + ts]
                            mat[lat, lon] = 1
                        else:
                            pos_to_div[pos] = div_id
                            mat[pos] = 1
                        div_pos.add(pos)
                        if pos not in neighbors:
                            for dlon,dlat in directions:
                                posLon, posLat = lon + dlon, lat + dlat
                                posNei = n_cols*posLat + posLon
                                if (posLon < start_col or posLon >= end_col) or (posLat < start_row or posLat >= end_row) or (posNei not in valid_locs):
                                    continue
                                neighbors[pos].append(posNei)

            valid_div_pos.append(div_pos)
            div_coords[div_id] = [start_row, end_row, start_col, end_col]
            print("Sub-Region {}: Rows[{} - {}] & Cols[{} - {}] ".format(i*area_ncols+j, start_row, end_row-1, start_col, end_col-1))
    sorted_neighbors = {key: neighbors[key] for key in sorted(neighbors)}
    fname = f'data/map/valid_grids_neighbors_{str(int(grid_size))}m_{n}.txt'
    with open(fname, 'w') as file:
        for key, val in sorted_neighbors.items():
            line = str(key) + '|' + ','.join(map(str, val)) + '\n'
            file.write(line)
    file.close()

    fname2 = f'data/map/valid_grids_{str(int(grid_size))}m.txt'
    with open(fname2, 'w') as file:
        for key, val in sorted_neighbors.items():
            lat, lon = key//n_cols, key%n_cols
            line = str(lon)+ ',' +str(lat) +'\n'
            file.write(line)
    file.close()

    if alg[:4]=="fuse" or alg[-1]=="F" or alg=="AGD-C":
        return sorted_neighbors, np.array(mat), valid_div_pos, pos_to_div, valid_cells, div_coords

    return sorted_neighbors, np.array(mat), valid_div_pos, pos_to_div, valid_locs, div_coords


def get_default_mat2D_DQN(grid_size, df, tf, km_time):

    time_intervals = load_time_intervals(tf)
    ts = tf//km_time
    fhv_locs_visited_cnt_24h_ts, valid_coords = get_default_mat2D_no_padding(grid_size, ts)

    for t in range(24*int(60//tf)): # len(time_intervals)):
        interval = timedelta(minutes=tf-1)
        curTime = time_intervals[t][0]
        nextTime = time_intervals[t][1]
        time_t = curTime + interval
        data_interval = fetch_rows(df, curTime, time_t)
        cov_cells = get_FHV_cov_coord(grid_size, data_interval, valid_coords)

        coord_arr = np.array(list(cov_cells))

        fhv_locs_visited_cnt_24h_ts[coord_arr[:, 0], coord_arr[:, 1]] += 1
    coord_valid = np.array(list(valid_coords))
    fhv_locs_visited_cnt_24h_ts[coord_valid[:, 0], coord_valid[:, 1]] = (((72-fhv_locs_visited_cnt_24h_ts[coord_valid[:, 0], coord_valid[:, 1]])//18)*(72-fhv_locs_visited_cnt_24h_ts[coord_valid[:, 0], coord_valid[:, 1]]))
    return fhv_locs_visited_cnt_24h_ts


def get_default_mat2D_no_padding(grid_size, df,ts=10):
    valid_cells = set()

    n_rows, n_cols, nPosns, n_rows_valid, n_cols_valid, nPosns_valid = gridSize_info_dict[grid_size] 
    with open(f'data/map/valid_grids_{str(int(grid_size))}m.txt', "r") as file:
        for line in file:
            lon, lat = map(int, line.split(','))
            valid_cells.add((lat, lon))
            pos = n_cols*lat + lon

    file.close()

    coordinates_array = np.array(list(valid_cells))

    mat = np.full((n_rows, n_cols), invalid_loc_val, dtype=int)
    mat[coordinates_array[:, 0], coordinates_array[:, 1]] = 0

    return mat, valid_cells


def get_default_mat2D(grid_size, df, ts):
    valid_cells = set()

    n_rows, n_cols, nPosns, n_rows_valid, n_cols_valid, nPosns_valid = gridSize_info_dict[grid_size] 
    with open(f'data/map/valid_grids_{str(int(grid_size))}m.txt', "r") as file:
        for line in file:
            lon, lat = map(int, line.split(','))
            valid_cells.add((lat, lon))
            pos = n_cols*lat + lon

    file.close()

    coordinates_array = np.array(list(valid_cells))

    mat = np.full((n_rows+(2*ts), n_cols+(2*ts)), invalid_loc_val, dtype=int)
    mat[coordinates_array[:, 0]+ts, coordinates_array[:, 1]+ts] = 0

    return mat, valid_cells


def get_mat2D_nDivs_fromFullMap(grid_size, df, tfm_map, n, ts, valid_cells):

    tfms = []
    coordinates_array = np.array(list(valid_cells))
    n_rows, n_cols, nPosns, n_rows_valid, n_cols_valid, nPosns_valid = gridSize_info_dict[grid_size] 

    mat = tfm_map.copy()

    area_nrows, area_ncols = size_dict[n]
    area_width, area_height = n_cols_valid//area_ncols, n_rows_valid//area_nrows
    print("\nDivisions:")
    for i in range(area_nrows):
        for j in range(area_ncols):
            div_pos = set()
            start_row, end_row = i*area_height, (i+1)*area_height
            start_col, end_col = j*area_width, (j+1)*area_width

            if j == area_ncols-1: 
                end_col = n_cols_valid
            if i == area_nrows-1:
                end_row = n_rows_valid

            sub_mat = mat[start_row:end_row, start_col:end_col]

            sub_mat_with_padding = np.pad(sub_mat, pad_width=ts, mode='constant', constant_values=invalid_loc_val)

            tfms.append(sub_mat_with_padding)
    return tfms

def get_default_mat2D_nDivs(grid_size, df, n, ts, valid_cells):

    tfms = []
    coordinates_array = np.array(list(valid_cells))
    n_rows, n_cols, nPosns, n_rows_valid, n_cols_valid, nPosns_valid = gridSize_info_dict[grid_size] 
    mat = np.full((n_rows, n_cols), invalid_loc_val, dtype=int)
    mat[coordinates_array[:, 0], coordinates_array[:, 1]] = 0

    area_nrows, area_ncols = size_dict[n]
    area_width, area_height = n_cols_valid//area_ncols, n_rows_valid//area_nrows
    print("\nDivisions:")
    for i in range(area_nrows):
        for j in range(area_ncols):
            div_pos = set()
            start_row, end_row = i*area_height, (i+1)*area_height
            start_col, end_col = j*area_width, (j+1)*area_width

            if j == area_ncols-1: 
                end_col = n_cols_valid
            if i == area_nrows-1:
                end_row = n_rows_valid

            sub_mat = mat[start_row:end_row, start_col:end_col]

            sub_mat_with_padding = np.pad(sub_mat, pad_width=ts, mode='constant', constant_values=invalid_loc_val)

            tfms.append(sub_mat_with_padding)
    return tfms

def get_default_mat(grid_size, df):
    visited_cells = set()
    visited_locs = set()
    n_rows, n_cols, nPosns, n_rows_valid, n_cols_valid, nPosns_valid = gridSize_info_dict[grid_size] 

    with open(f'data/map/valid_grids_{str(int(grid_size))}m.txt', "r") as file:
        for line in file:
            lon, lat = map(int, line.split(','))
            visited_cells.add((lon, lat))
            pos = n_cols*lat + lon
            visited_locs.add(pos)
    file.close()

    mat = [-((2**31)-1) for i in range(nPosns)]
    for cell in visited_cells:
        lon, lat = map(int, cell)
        pos = n_cols*lat + lon
        mat[pos] = 1
    return np.array(mat)

def get_default_cnt_mat(grid_size, df):
    visited_cells = set()
    visited_locs = set()
    n_rows, n_cols, nPosns, n_rows_valid, n_cols_valid, nPosns_valid = gridSize_info_dict[grid_size] 

    with open(f'data/map/valid_grids_{str(int(grid_size))}m.txt', "r") as file:
        for line in file:
            lon, lat = map(int, line.split(','))
            visited_cells.add((lon, lat))
            pos = n_cols*lat + lon
            visited_locs.add(pos)
    file.close()

    mat = [-1 for i in range(nPosns)]
    for cell in visited_cells:
        lon, lat = map(int, cell)
        pos = n_cols*lat + lon
        mat[pos] = 0
    return np.array(mat)

def get_cov_matrix(grid_size, df, cov):

        mat = get_default_mat(grid_size, df)
        mat[list(cov)] = 0
        return mat


def get_default_mat2D_coverage(grid_size, df, ts=0):
    valid_cells = set()

    n_rows, n_cols, nPosns, n_rows_valid, n_cols_valid, nPosns_valid = gridSize_info_dict[grid_size] 
    with open(f'data/map/valid_grids_{str(int(grid_size))}m.txt', "r") as file:
        for line in file:
            lon, lat = map(int, line.split(','))
            valid_cells.add((lat, lon))
            pos = n_cols*lat + lon

    file.close()

    coordinates_array = np.array(list(valid_cells))

    mat = np.full((n_rows, n_cols), invalid_loc_val, dtype=int)
    mat[coordinates_array[:, 0], coordinates_array[:, 1]] = 1

    return mat, valid_cells

def get_cov_matrix_2D(grid_size, df, cov):

    mat, _ = get_default_mat2D_coverage(grid_size, df)
    print(mat.shape)
    coordinates_array = np.array(list(cov))
    mat[coordinates_array[:, 0], coordinates_array[:, 1]] = 0
    return mat

def load_neighbors_list():

    neighbors = collections.defaultdict(list)
    with open('data/valid_grids_neighbors.txt', "r") as file:
        for line in file:
            pos, nei = line.split('|')
            pos = int(pos)
            nei = nei.split(',')
            nei = list(map(int, nei))
            neighbors[pos] = set(nei)
    file.close()

    print(len(neighbors))

    return neighbors

def generate_time_intervals(tf):

    start_date = datetime(2023, 1, 1, 0, 0)
    end_date = datetime(2023, 1, 2, 0, 0)  # Change this to the end date you need

    interval = timedelta(minutes=(tf-1))
    next_slot = timedelta(minutes=tf)
    current_date = start_date
    fname = 'data/time_intervals_'+str(tf)+'.txt'
    with open(fname, 'w') as file:
        while current_date < end_date:
            next_date = current_date + interval
            date_time_str = f"{current_date.strftime('%Y-%m-%d %H:%M')},{next_date.strftime('%Y-%m-%d %H:%M')}\n"
            file.write(date_time_str)
            current_date = current_date + next_slot

def load_time_intervals(tf):
    list_intervals = []
    fname = 'data/time_intervals_'+str(tf)+'.txt'
    with open(fname, 'r') as file:

        for line in file:
            start, end = line.split(',')
            start, end = pd.to_datetime(start), pd.to_datetime(end)
            list_intervals.append([start, end])

    return list_intervals

def fetch_rows(df, start, end):
    rows_fhv = df[((df['Time'] >= start) & (df['Time'] <= end))]
    return rows_fhv #, rows_dsv

def fetch_rows_at(df, cur_time):
    rows_fhv = df[((df['Time'] == cur_time))]
    return rows_fhv #, rows_dsv

def path_calculator(start, end):
    path = set()
    latStart, lonStart = start//n_cols, start%n_cols
    latEnd, lonEnd = end//n_cols, end%n_cols

    dirLong = int((lonEnd - lonStart)/(abs(lonEnd - lonStart))) if abs(lonEnd - lonStart) > 0  else 0
    dirLat = int((latEnd - latStart)/(abs(latEnd - latStart))) if abs(latEnd - latStart) > 0  else 0
    while (abs(latEnd - latStart)!=0 and abs(lonStart - lonEnd)!=0):
        pos = n_cols*latStart + lonStart
        path.add(pos)

        latStart += dirLat
        lonStart += dirLong

    pos = n_cols*latStart + lonStart
    path.add(pos)
    dirLong = int((lonEnd - lonStart)/(abs(lonEnd - lonStart))) if abs(lonEnd - lonStart) > 0  else 0
    dirLat = int((latEnd - latStart)/(abs(latEnd - latStart))) if abs(latEnd - latStart) > 0  else 0

    while (abs(latEnd - latStart)!=0):  # and abs(lonStart - lonEnd)!=0):
        pos = n_cols*latStart + lonStart
        path.add(pos)

        latStart += dirLat
        lonStart += dirLong

    while (abs(lonStart - lonEnd)!=0): # abs(latEnd - latEnd)!=0):
        pos = n_cols*latStart + lonStart
        path.add(pos)

        latStart += dirLat
        lonStart += dirLong

    pos = n_cols*latStart + lonStart
    path.add(pos)

    return path

def generate_path_dict(validLoc):
    pathDict = {}
    validLocs = set(validLoc.keys())
    locs = [i for i in range(nPosns)]
    for i in range(len(locs)):
        if (i not in validLocs):

            continue
        pathDict[(i, i)] = [i]
        for j in range(i+1, len(locs)):
            if (j not in validLocs):

                continue
            path = path_calculator(i, j)
            pathDict[(i, j)] = sorted(list(path))
            pathDict[(j, i)] = sorted(list(path))

        print("Completed Location: {} ".format(i))

    with open('data/path_dict.txt', 'w') as file:
        for key, val in pathDict.items():
            x, y = key
            line = str(x) + ',' + str(y) + '|' + ','.join(map(str, val)) + '\n'
            file.write(line)
    file.close()

def valid_neighbors(r, c, validLoc, path):
    directions = [[-1, 1], [0, 1], [1, 1], [-1, 0], [1, 0], [-1, -1], [0, -1], [1, -1]]
    cur_pos = r*n_cols + c
    valid_nei = []
    for dr, dc in directions:
        row, col = r+dr, c+dc
        pos = row*n_cols + col
        if (row < 0 or row >= n_rows_valid or 
            col < 0 or col >= n_cols_valid or 
            pos not in validLoc or 
            pos not in validLoc[cur_pos] or
            pos in path):
            continue
        valid_nei.append([row, col])
    return valid_nei

def generate_path_dict_timeframe(locs, validLoc, timeframe):

    path_dict = defaultdict(list)
    validLocs = set(validLoc.keys())

    path = set()

    def dfs(row, col, full_path, parent):

        if parent not in validLoc:
            return

        if len(full_path) == timeframe+1:
            path_dict[parent].append(full_path.copy())

            return

        for r, c in valid_neighbors(row, col, validLoc, path):
            curPos = r*n_cols + c
            path.add(curPos)
            full_path.append(curPos)

            dfs(r, c, full_path, parent)

            path.remove(curPos)
            full_path.pop()
        return
    for i, pos in enumerate(locs):
        if (pos not in validLoc):

            continue

        path.add(pos)

        latIDStart, lonIDStart = pos//n_cols, pos%n_cols

        dfs(latIDStart, lonIDStart, [pos], pos)

        path.remove(pos)

    return path_dict

def parallel_generate_path_dict_timeframe_thread(N, validLoc, timeframe, nthreads=140):
    '''
    Parallel-generate the path dictionary for all positions.
    '''
    print("Running Path Dict Generation for Timeframe=", timeframe)

    nthreads_obj = nthreads
    N_split_local = np.array_split(N, nthreads_obj)

    with concurrent.futures.ThreadPoolExecutor(max_workers=nthreads_obj) as executor:
        futures = [executor.submit(generate_path_dict_timeframe, split, validLoc, timeframe) for split in N_split_local]
        all_path_dicts = [f.result() for f in futures]
    path_dict = {}

    for dict in all_path_dicts:
        path_dict.update(dict)

    print("Generated Path Dict for TimeFrame: ", timeframe)
    fname = 'data/path_dict_' +str(timeframe)+ '.txt'
    with open(fname, 'w') as file:
        for key, all_paths in path_dict.items():
            x= key
            paths = [','.join(map(str, path_i)) for path_i in all_paths]
            line = str(x) + '|' + '|'.join(paths) + '\n'
            file.write(line)
    file.close()

    print("Saved Path Dict for TimeFrame: ", timeframe)
    return

def generate_path_dict_forTimeFrames(validLoc):
    locs = [i for i in range(nPosns)]
    time_frames = [5, 10, 30, 60, 120, 180, 360, 720, 1440]

    for t in time_frames:
        parallel_generate_path_dict_timeframe_thread(locs, validLoc, t)

def load_path_dict():

    paths = collections.defaultdict(list)
    with open('data/path_dict.txt', "r") as file:
        for line in file:
            pos, nei = line.split('|')
            x, y = map(int, pos.split(','))
            nei = nei.split(',')
            nei = list(map(int, nei))
            paths[(x, y)] = set(nei)
    file.close()

    return paths

def initialize_STF_weights(validLoc):
    TFM = [-1e9]*nPosns

    for pos in validLoc.keys():
        TFM[pos] = 0

    condition = np.array(TFM) > -1e9
    indices = np.where(condition)
    indices_array = np.array(indices).flatten()
    print("Valid elements in TFM: ", len(indices_array))

    return np.array(TFM)


def pos_to_coord(grid_size, pos):
    n_rows, n_cols, nPosns, n_rows_valid, n_cols_valid, nPosns_valid = gridSize_info_dict[grid_size] 

    return (pos//n_cols, pos%n_cols)


def coord_to_pos(grid_size, lat, lon):
    n_rows, n_cols, nPosns, n_rows_valid, n_cols_valid, nPosns_valid = gridSize_info_dict[grid_size] 

    return n_cols*lat + lon


def get_FHV_cov(df, valid_locs):
    locs =  set(df.Position.unique())
    return list(locs.intersection(valid_locs.keys()))


def get_FHV_cov_coord(grid_size, df, valid_coords):
    locs = set(df.Position.unique())
    coords = set([pos_to_coord(grid_size, pos) for pos in locs])
    coords = set(list(coords.intersection(valid_coords)))
    return coords               


def get_FHV_cov_cnt(grid_size, df, valid_locs):
    n_rows, n_cols, nPosns, n_rows_valid, n_cols_valid, nPosns_valid = gridSize_info_dict[grid_size] 

    locs_cnts =  df.Position.value_counts()
    filtered_locs_cnts = locs_cnts[locs_cnts.index.isin(valid_locs)]

    locs_cnts_array = np.zeros(nPosns, dtype=int)
    locs_cnts_array[filtered_locs_cnts.index] = filtered_locs_cnts.values
    return locs_cnts_array

def gen_fairness_heatMap(grid_size, df, tf, time_intervals, valid_locs):
    n_rows, n_cols, nPosns, n_rows_valid, n_cols_valid, nPosns_valid = gridSize_info_dict[grid_size] 

    fhv_locs_visited_cnt_ts = get_default_cnt_mat(grid_size, df)
    fhv_locs_visited_cnt_24h_ts = get_default_cnt_mat(grid_size, df)

    fhv_locs_visited_cnt_vehs = get_default_cnt_mat(grid_size, df)
    fhv_locs_visited_cnt_24h_vehs = get_default_cnt_mat(grid_size, df)

    times_vec = []
    std_vec_ts = []
    std_vec_vehs = []
    std_vec_ts_24h = []
    std_vec_vehs_24h = []

    mean_vec_ts = []
    mean_vec_vehs = []
    mean_vec_ts_24h = []
    mean_vec_vehs_24h = []

    for t in range(24*int(60//tf)): # len(time_intervals)):
        interval = timedelta(minutes=tf-1)
        curTime = time_intervals[t][0]
        nextTime = time_intervals[t][1]
        time_t = curTime + interval

        print("\nTime: ", curTime, time_t)

        data_interval = fetch_rows(df, curTime, time_t)
        locs_interval_ts = get_FHV_cov(data_interval, valid_locs)
        locs_interval_cnt_vehs = get_FHV_cov_cnt(grid_size, data_interval, valid_locs)

        fhv_locs_visited_cnt_ts[locs_interval_ts] += 1
        fhv_locs_visited_cnt_24h_ts[locs_interval_ts] += 1

        fhv_locs_visited_cnt_vehs = np.add(fhv_locs_visited_cnt_vehs, locs_interval_cnt_vehs)
        fhv_locs_visited_cnt_24h_vehs = np.add(fhv_locs_visited_cnt_24h_vehs, locs_interval_cnt_vehs)

        if t%(int(60//tf)) == (int(60//tf)-1):
            std_vec_ts.append(np.std(fhv_locs_visited_cnt_ts[fhv_locs_visited_cnt_ts >= 0]))
            std_vec_vehs.append(np.std(fhv_locs_visited_cnt_vehs[fhv_locs_visited_cnt_vehs >= 0]))

            mean_vec_ts.append(np.mean(fhv_locs_visited_cnt_ts[fhv_locs_visited_cnt_ts >= 0]))
            mean_vec_vehs.append(np.mean(fhv_locs_visited_cnt_vehs[fhv_locs_visited_cnt_vehs >= 0]))
            times_vec.append(curTime.hour)

            print_data = pd.DataFrame(
                    {
                        'Time(h)': times_vec,
                        'Mean_Dev_TimeSlots': mean_vec_ts,
                        'Std_Dev_TimeSlots': std_vec_ts,
                        'Mean_Dev_Vehicles': mean_vec_vehs,
                        'Std_Dev_Vehicles': std_vec_vehs,
                    }
                )

            fname = 'result/heatmap_'+str(tf)+'_shenzen.csv'
            save_df(print_data, fname)

            print("\n{} Hourly Heat Map Results (Min:{}  Max:{}) ({}) Generated and Saved\n********-----********-----********-----********-----*******\n".format(curTime, min(fhv_locs_visited_cnt_ts[fhv_locs_visited_cnt_ts >= 0]), max(fhv_locs_visited_cnt_ts[fhv_locs_visited_cnt_ts >= 0]), fname))
            fhv_locs_visited_cnt_ts = get_default_cnt_mat(grid_size, df)
            fhv_locs_visited_cnt_vehs = get_default_cnt_mat(grid_size, df)

    print_data = pd.DataFrame(
                    {
                        'Time(h)': times_vec,
                        'Mean_Dev_TimeSlots': mean_vec_ts,
                        'Std_Dev_TimeSlots': std_vec_ts,
                        'Mean_Dev_Vehicles': mean_vec_vehs,
                        'Std_Dev_Vehicles': std_vec_vehs,
                    }
                )

    std_vec_ts_24h.append(np.std(fhv_locs_visited_cnt_24h_ts[fhv_locs_visited_cnt_24h_ts >= 0]))
    std_vec_vehs_24h.append(np.std(fhv_locs_visited_cnt_24h_vehs[fhv_locs_visited_cnt_24h_vehs >= 0]))

    mean_vec_ts_24h.append(np.mean(fhv_locs_visited_cnt_24h_ts[fhv_locs_visited_cnt_24h_ts >= 0]))
    mean_vec_vehs_24h.append(np.mean(fhv_locs_visited_cnt_24h_vehs[fhv_locs_visited_cnt_24h_vehs >= 0]))
    avg_std_dev_ts, avg_std_dev_vehs, avg_mean_ts, avg_mean_vehs = np.mean(std_vec_ts), np.mean(std_vec_vehs), np.mean(mean_vec_ts), np.mean(mean_vec_vehs)
    data_avg = pd.DataFrame(
                    {
                        'Time(h)': '2023-01-01 50:00:00',
                        'Mean_Dev_TimeSlots':[avg_mean_ts],
                        'Std_Dev_TimeSlots': [avg_std_dev_ts],
                        'Mean_Dev_Vehicles': [avg_mean_vehs],
                        'Std_Dev_Vehicles': [avg_std_dev_vehs]
                    }
                )

    data_24h = pd.DataFrame(
                    {
                        'Time(h)': '2023-01-01 100:00:00',
                        'Mean_Dev_TimeSlots': mean_vec_ts_24h,
                        'Std_Dev_TimeSlots': std_vec_ts_24h,
                        'Mean_Dev_Vehicles': mean_vec_vehs_24h,
                        'Std_Dev_Vehicles': std_vec_vehs_24h
                    }
                )

    print_data_new = pd.concat([print_data, data_avg, data_24h])
    fname = 'result/heatmap_'+str(tf)+'_shenzen.csv'
    save_df(print_data_new, fname)
    image = plt.imread('figs/shenzen-map_new.png')

    heatmap_ts_24 = fhv_locs_visited_cnt_24h_ts.reshape(n_rows, n_cols)
    heatmap_vehs_24 = fhv_locs_visited_cnt_24h_vehs.reshape(n_rows, n_cols)
    fhv_locs_visited_cnt_24h_ts_hourly = fhv_locs_visited_cnt_24h_ts/24
    heatmap_ts_24_hourly = fhv_locs_visited_cnt_24h_ts_hourly.reshape(n_rows, n_cols)

    masked_heatmap_ts = np.ma.masked_where(heatmap_ts_24 < 0, heatmap_ts_24)
    masked_heatmap_ts_hourly = np.ma.masked_where(heatmap_ts_24_hourly < 0, heatmap_ts_24_hourly)
    plt.imshow(image, extent=[0, image.shape[1], 0, image.shape[0]])
    heatmap = plt.imshow(masked_heatmap_ts, cmap='plasma', interpolation='nearest', extent=[0, image.shape[1], 0, image.shape[0]])
    plt.xticks([])
    plt.yticks([])
    plt.grid(visible=True, color='black', linestyle='-', linewidth=0.5, which='both')  # Add gridlines
    cbar = plt.colorbar(heatmap, orientation='horizontal', shrink=0.5, ticks=[np.min(masked_heatmap_ts), np.max(masked_heatmap_ts)], location='top')
    plt.savefig('figs/map/heatmap_ts_'+str(tf)+'.png', transparent=True)
    plt.clf()
    plt.imshow(image, extent=[0, image.shape[1], 0, image.shape[0]])
    heatmap = plt.imshow(masked_heatmap_ts_hourly, cmap='plasma', interpolation='nearest', extent=[0, image.shape[1], 0, image.shape[0]])
    plt.xticks([])
    plt.yticks([])
    plt.grid(visible=True, color='black', linestyle='-', linewidth=0.5, which='both')  # Add gridlines
    cbar = plt.colorbar(heatmap, orientation='horizontal', shrink=0.5, ticks=[0, 1, 2, 3], location='top', label='Average Hourly FHV Sensing Frequency')
    plt.savefig('figs/map/heatmap_ts_hourly_'+str(tf)+'.png', transparent=True)

    plt.clf()
    bins = [0, 1, 2, 3]
    categories = pd.cut(masked_heatmap_ts_hourly.flatten(), bins, labels=['0-1', '1-2', '2-3'])
    result_df = pd.DataFrame({'Value': masked_heatmap_ts_hourly.flatten(), 'Category': categories})
    frequency_table = result_df['Category'].value_counts().sort_index()
    print(frequency_table)

    cv_values = print_data['Std_Dev_TimeSlots'] / print_data['Mean_Dev_TimeSlots']
    plt.bar(print_data['Time(h)'], cv_values) #, color='skyblue')
    plt.xlabel('Time (h)')
    plt.ylabel('Coefficient of Variation (CV = $\\sigma/\\mu$)')
    plt.title('Coefficient of Variation ($\\sigma/\\mu$) of FHV Sensing at Each Hour')
    plt.xticks([0, 6, 12, 18, 24])

    plt.savefig('figs/map/cv_ts_'+str(tf)+'.png')
    print(cv_values)
    print("\nFull Day Heatmap Results Generated (Min:{}  Max:{}) and Saved ({})\n********-----********-----********-----********-----*******\n".format(min(fhv_locs_visited_cnt_24h_ts[fhv_locs_visited_cnt_24h_ts >= 0]), max(fhv_locs_visited_cnt_24h_ts[fhv_locs_visited_cnt_24h_ts >= 0]),fname))

def gen_train_data(df, tf, valid_locs, grid_size=1000):
    tf_train = 2

    tfm_map, valid_coords = get_default_mat2D(tf)

    generate_time_intervals(tf_train)

    time_intervals = load_time_intervals(tf_train)

    print("Generating training data...")

    fname = 'data/training/training_data_2X_tmp.h5'

    with h5py.File(fname, 'w') as file:
        for t in range(24*int(60//tf_train)): # len(time_intervals)):
            interval = timedelta(minutes=tf_train-1)
            curTime = time_intervals[t][0]
            nextTime = time_intervals[t][1]
            time_t = curTime + interval
            hour, mins = time_t.hour, time_t.minute
            print("\nTime: ", curTime, time_t)

            data_interval = fetch_rows(df, curTime, time_t)

            locs_interval_ts = get_FHV_cov_coord(grid_size, data_interval, valid_locs)

            tfm_map = update_tfm_sfm_tEnd(tfm_map, valid_coords, tf)

            tfm_map = update_tfm_sfm_cov(tfm_map, locs_interval_ts, tf)

            ts = tf//2 # Since vehicles can traverse from one grid to next in 2 minutes (30 km/h)

            for x, y in valid_coords:
                nx, ny = x+tf, y+tf
                submatrix = tfm_map[nx - ts : nx + ts + 1, ny - ts : ny + ts + 1] # For ts matrices

                submatrix = submatrix.copy()
                submatrix[submatrix.shape[0] // 2, submatrix.shape[1] // 2] = dsv_loc_val
                file.create_dataset(f'matrix_{x}_{y}_{hour}_{mins}', data=submatrix)
            print("Submatrix generated for {}:{}".format(hour, mins))
    print("Training Data Generated and Saved({})".format(fname))

def generate_directional_channels(matrix, ts, km_time):

    x, y = np.where(matrix == 0)  # Example vehicle location
    vehicle_location = (x.item(), y.item())
    directions = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]
    directional_channels = []

    for direction in directions:
        dx, dy = direction
        center_y, center_x = vehicle_location
        if dx < 0:
            start_x, end_x =  max(center_x + dx - ts +1, 0), center_x
        elif dx > 0:
            start_x, end_x =  center_x, min(center_x + dx + ts -1, matrix.shape[0])
        else:
            start_x, end_x =  max(center_x + dx - ts//km_time, 0), min(center_x + dx + ts//km_time, matrix.shape[0])

        if dy < 0:
            start_y, end_y =  max(center_y + dy - ts +1, 0), center_y
        elif dy > 0:
            start_y, end_y =  center_y, min(center_y + dy + ts -1, matrix.shape[0])
        else:
            start_y, end_y =  max(center_y + dy - ts//km_time, 0), min(center_y + dy + ts//km_time, matrix.shape[0])

        directional_channel = np.zeros((matrix.shape[0], matrix.shape[1]))

        directional_channel[start_y:end_y+1, start_x:end_x+1] = matrix[start_y:end_y+1, start_x:end_x+1]

        directional_channels.append(directional_channel)
    return directional_channels

def load_and_preprocess_training_data( ts):

    file_path = 'data/training/training_data.h5'
    num_channels = 9

    with h5py.File(file_path, 'r') as file:
        num_samples = len(file.keys())
        first_key = list(file.keys())[0]
        first_matrix = file[first_key][()]
        data_shape = first_matrix.shape  # Assuming all matrices have the same shape
        X_data = np.zeros((num_samples,) + data_shape + (num_channels,), dtype=np.float32)
        for i, key in enumerate(file.keys()):
            matrix = file[key][()]
            tmp = matrix.copy()
            matrix[matrix == invalid_loc_val] = 0
            matrix[matrix == dsv_loc_val] = -1

            normalized_matrix = (matrix+1) / (max_valid_value+1)
            normalized_matrix[tmp == invalid_loc_val] = 0
            normalized_matrix[tmp == dsv_loc_val] = -1
            directional_channels = generate_directional_channels(normalized_matrix, ts)
            all_channels = [normalized_matrix] + directional_channels
            preprocessed_matrix = np.stack(all_channels, axis=-1)

            X_data[i] = preprocessed_matrix
            print('Data File {} Processed for training \n'.format(key))
    output_file = 'data/training/training_multi_channel_data_corrected.h5'
    with h5py.File(output_file, 'w') as file:
        file.create_dataset('training_multi_channel_matrix', data=X_data)   

    print('Training data loaded from {} and preprocessed'.format(file_path))
    return X_data


def load_save_training_data( ts):

    file_path = 'data/training/training_data.h5'
    num_channels = 9

    with h5py.File(file_path, 'r') as file:
        num_samples = len(file.keys())
        first_key = list(file.keys())[0]
        first_matrix = file[first_key][()]
        data_shape = first_matrix.shape  # Assuming all matrices have the same shape
        X_data = np.zeros((num_samples,) + data_shape, dtype=np.float32)
        for i, key in enumerate(file.keys()):
            matrix = file[key][()]

            X_data[i] = matrix
            print('{} - Data File {} Processed for training \n'.format(i, key))
    output_file = 'data/training/training_full_data.h5'
    with h5py.File(output_file, 'w') as file:
        file.create_dataset('training_full_data', data=X_data)   

    print('Training data loaded from {} and preprocessed - Shape:{}'.format(file_path, X_data.shape))
    return X_data


def gen_state(tfm_map, y, x, tf, km_time):
    mat = tfm_map.copy()

    ts = tf//km_time
    submatrix = mat[y - ts : y + ts + 1, x - ts : x + ts + 1] # For ts matrices

    submatrix = submatrix.copy()
    submatrix[submatrix.shape[0] // 2, submatrix.shape[1] // 2] = dsv_loc_val
    return submatrix


def gen_state_region(tfm_map, y, x, tf, km_time):
    mat = tfm_map.copy()
    ts = tf//km_time
    submatrix = mat[y - ts : y + ts + 1, x - ts : x + ts + 1] # For ts matrices

    submatrix = submatrix.copy()
    return submatrix


def gen_state_coverage(tfm_map, y, x, tf, km_time):
    mat = tfm_map.copy()
    ts = tf//km_time
    submatrix = mat[y - ts : y + ts + 1, x - ts : x + ts + 1] # For ts matrices

    submatrix = submatrix.copy()
    submatrix[submatrix.shape[0] // 2, submatrix.shape[1] // 2] = 0
    return submatrix


def gen_state_coverage_region(tfm_map, y, x, tf, km_time):
    mat = tfm_map.copy()
    ts = tf//km_time
    submatrix = mat[y - ts : y + ts + 1, x - ts : x + ts + 1] # For ts matrices

    submatrix = submatrix.copy()
    return submatrix

def update_map(tfm_global_map, tfm_map, state, y, x, bounds, tf, km_time):
    mat = tfm_map.copy()
    ts = tf//km_time
    mat[y - ts : y + ts + 1, x - ts : x + ts + 1] = state.copy()
    start_row, end_row, start_col, end_col = bounds
    global_mat = tfm_global_map.copy()
    mat_core = mat[ts:-ts, ts:-ts].copy()
    global_mat[start_row: end_row, start_col: end_col] = mat_core
    return mat, global_mat

def gen_feature(state, ts=10):
    tmp = state.copy()
    matrix = state.copy()        

    matrix[matrix == invalid_loc_val] = -1
    matrix[matrix == dsv_loc_val] = 0

    normalized_matrix = (matrix+1) / (max_valid_value+1)
    normalized_matrix[tmp == invalid_loc_val] = -1
    normalized_matrix[tmp == dsv_loc_val] = 0
    directional_channels = generate_directional_channels(normalized_matrix, ts)
    all_channels = [normalized_matrix] + directional_channels
    preprocessed_matrix = np.stack(all_channels, axis=-1)
    return preprocessed_matrix


def gen_feature_8dir(state, ts=10):
    tmp = state.copy()
    matrix = state.copy()        

    matrix[matrix == invalid_loc_val] = -1
    matrix[matrix == dsv_loc_val] = 0

    normalized_matrix = (matrix+1) / (max_valid_value+1)
    normalized_matrix[tmp == invalid_loc_val] = -1
    normalized_matrix[tmp == dsv_loc_val] = 0
    directional_channels = generate_directional_channels(normalized_matrix, ts)
    all_channels = [normalized_matrix]
    preprocessed_matrix = np.stack(all_channels, axis=-1)
    return preprocessed_matrix

def read_training_data(file_path='data/training/training_multi_channel_data_corrected.h5', mat_name='training_multi_channel_matrix', grid_size=1000):

    n_rows, n_cols, nPosns, n_rows_valid, n_cols_valid, nPosns_valid = gridSize_info_dict[grid_size]

    with h5py.File(file_path, 'r') as file:
        loaded_data = file[mat_name][()]
    print('Loaded Training Data ({}) - Shape:{}'.format(file_path, loaded_data.shape))

    print(f"n_rows:{n_rows}, n_cols:{n_cols};  n_rows_valid:{n_rows_valid}, n_cols_valid:{n_cols_valid}")

    return loaded_data


def update_tfm_sfm_cov(mat, cov_cells, tf, testing=False):
    if testing:
        tf = 0
    tfm = mat.copy()
    coordinates_array = np.array(list(cov_cells))

    tfm[coordinates_array[:, 0]+tf, coordinates_array[:, 1]+tf] = 0

    return tfm

def update_tfm_sfm_tEnd(mat, valid_coords, tf, testing=False):
    if testing:
        tf = 0
    tfm = mat.copy()

    coordinates_array = np.array(list(valid_coords))

    tfm[coordinates_array[:, 0]+tf, coordinates_array[:, 1]+tf] += 1

    return tfm

def update_TSFMetrics(cov, TFM, neighbors):

    TFM[list(cov)] = -BETA

    for pos in cov:
        neis = neighbors[pos]
        TFM[list(neis)] -= BETA

    return TFM #, SFM

def update_TSFMetrics_tEnd(TFM, neighbors):
    for pos in neighbors:
        TFM[pos] += 1
    return TFM

def get_initial_loc_DSVs(grid_size, nDSV=1, seed=42):

    valid_locs = collections.defaultdict(list)

    fname = f'data/map/valid_grids_neighbors_{str(int(grid_size))}m_{nDSV}.txt'

    with open(fname, "r") as file:
        for line in file:
            pos, nei = line.split('|')
            pos = int(pos)
            nei = nei.split(',')
            nei = list(map(int, nei))
            valid_locs[pos] = set(nei)
    file.close()
    random.seed(seed)
    dsv_locs = random.sample(list(valid_locs.keys()), nDSV)
    return dsv_locs, valid_locs

def marginal_gain_path(path, TFM):

    vals = (1+ALPHA)**TFM[list(path)]

    gain = np.sum(vals)/(len(path)*GAMMA)
    interval = timedelta(minutes=(2*len(path))) # 2 minutes per grid / km
    return gain, interval

def get_DSV_val(dsv, vMap, validLoc, paths, TFM, curTime ):
    dLoc, dAvailTime = vMap[dsv]
    if dAvailTime > curTime: # Vehicle Not Available
        return -1
    maxGain = 0
    maxPath = []
    interval = 0
    dest = dLoc
    for loc in validLoc:
        path = paths[(dLoc, loc)]
        if len(path) == 0:
            print(dLoc, loc, path)
        gain, interval = marginal_gain_path(path, TFM)
        if gain > maxGain:
            maxGain = gain

            nextAvail = dAvailTime + interval
            dest = loc
    return {'vid': dsv,'dest':dest, 'gain':gain, 'avail':nextAvail}
