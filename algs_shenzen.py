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

import concurrent.futures
import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from queue import PriorityQueue

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import collections
from collections import deque
from datetime import datetime, timedelta
import random
import json

from collections import deque
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, LeakyReLU, BatchNormalization, Flatten, Dense
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import ModelCheckpoint
checkpoint_path = 'models/best_fuse_model.h5'
checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, mode='min', verbose=1)

import concurrent.futures
import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from queue import PriorityQueue

from utils_shenzen import get_FHV_cov, generate_neighbors_list_wBorders, fetch_rows, get_cov_matrix, generate_path_dict_timeframe, gen_feature, get_default_mat2D, get_FHV_cov_coord, update_tfm_sfm_tEnd, update_tfm_sfm_cov, gen_state, pos_to_coord, coord_to_pos, update_map, get_default_mat2D_nDivs, gen_feature_8dir, get_mat2D_nDivs_fromFullMap, gen_state_coverage, gen_state_region, get_cov_matrix_2D

'''
Global Parameters
'''
nDSV = 500
dsv_loc_val = -1000 # value to signify current loc of dsv
invalid_loc_val = -((2**31)-1)
max_valid_value = 720
gridSize_info_dict = {3000:(13, 30, 390, 13, 28, 364), 1000:(41, 91, 3731, 39, 84, 3276), 500:(83, 182, 15106, 78, 168, 13104)}
size_dict = {1: [1, 1], 2: [1, 2], 4:[2, 2], 8: [2, 4], 16: [4, 4], 32:[4, 8]}

ALPHA = 0.2
BETA = 2
GAMMA = 1

'''
BASELINES
'''

class TSMTC:
    '''
    TSMTC (Trajectory Scheduling for Maximum Task Cover) \cite{FanJLQGLFW21}}: TSMTC recommends the best neighbor location with the maximum coverage possibilities, prioritizing areas with the highest number of unsensed neighbors
    '''
    def __init__(self, df, grid_size, n=1, tf=5, beta=60, seed=42, km_time=2) -> None:
        random.seed(seed)
        self.name = "TSMTC-C"
        self.df = df
        self.n = n
        self.timeframe = tf
        self.grid_size = grid_size
        self.km_time = int(km_time*grid_size/1000) # km_time to complete 1 km (move to next grid)
        self.path_int = tf//self.km_time # path length per interval

        n_rows, n_cols, nPosns, n_rows_valid, n_cols_valid, nPosns_valid = gridSize_info_dict[self.grid_size]
        self.map_area = np.array([[0 for i in range(n_cols_valid)] for j in range(n_rows_valid)])
        self.area_nrows, self.area_ncols = size_dict[self.n]
        self.pos_neis, self.mat, self.valid_div_pos, self.pos_to_div, self.valid_locs, self.div_coords = generate_neighbors_list_wBorders(self.n, grid_size, self.df, self.name, self.timeframe, self.km_time)
        # print(self.pos_neis)
        # exit()
        self.dsvL = [random.sample(list(self.valid_div_pos[i]), 1)[0] for i in range(n)]
        self.dsv_paths = {i:deque([self.dsvL[i]]) for i in range(self.n)}

        print("\n{} Model Object Created".format(self.name))
        print("\nValid Locations Per Region", [len(self.valid_div_pos[i]) for i in range(self.n)])
        print("DSV Starting Locations: ",self.dsvL)
        print("DSV Starting Grid Positions: ", [(pos//n_cols, pos%n_cols) for pos in self.dsvL])
        print('Path Length: ', self.path_int)

    def margvals_returnvals_nei_wSumNei(self, L):

        neis = self.pos_neis[L]
        val = self.mat[L]
        neis_val = []
        if val > 0: # i.e the location 'L' has not been covered yet
            '''
            i.e the location 'L' has not been covered yet
            ''' 
            neis_val = [self.mat[nei] for nei in neis] # How many neighbors of 'L' have not been covered yet? # if self.mat[nei]
            val += sum(neis_val)
        else: 
            '''
            i.e the location 'L' has been covered already
            We divide by 8 as 8 is the maximum possible neighbors, thus the sum will be < 1 
            (and this ensures the best covered 'L' is selected only when none other uncovered exists 'L' as possible next locations)
            '''
            neis_val = [self.mat[nei]/8 for nei in neis] 
            val += sum(neis_val)
        return val

    def margvals_returnvals_wMaxNei(self, L):

        neis = self.pos_neis[L]
        val = 1 - self.mat[L]
        neis_val = [1 - self.mat[nei] for nei in neis]
        return val + max(neis_val) # self.mat[neis].max(0).sum()
        return ele_vals_local_vals

    def margvals_returnvals(self, L):
        return 1 - self.mat[L]
        return ele_vals_local_vals

    def run(self, vid):

        queries = 0

        cur_pos = self.dsvL[vid]

        neis = set([nei for nei in self.pos_neis[cur_pos]])
        neis = list(neis - set(self.dsv_paths[vid]))
        if len(neis) == 0:
            neis = list(set([nei for nei in self.pos_neis[cur_pos]]))
        ele_vals_gain = [ self.margvals_returnvals_nei_wSumNei(nei) for nei in neis]

        argmax_gain = np.argmax( ele_vals_gain );
        cur_pos = neis[argmax_gain]

        self.dsvL[vid] = cur_pos
        self.dsv_paths[vid].append(cur_pos)
        if len(self.dsv_paths[vid]) > 60:
            self.dsv_paths[vid].popleft()

        return cur_pos

    def parallel_run(self, curTime):

        N = [ele for ele in range(self.n)]
        nthreads_obj = self.n
        N_split_local = np.array_split(N, nthreads_obj)
        dsv_paths = {i:[self.dsvL[i]] for i in range(self.n)}

        time_t_prev = curTime
        interval = timedelta(minutes=self.timeframe-1)
        time_t = time_t_prev + interval
        time_t_pastHour = time_t - timedelta(minutes=60-self.timeframe)
        print("Time_t_Pasthour: ", time_t_pastHour, "  Time_t: ", time_t)
        data = fetch_rows(self.df, time_t_pastHour, time_t)
        cov = get_FHV_cov(data, self.pos_neis)

        for t in range(self.path_int):

            cov.extend(self.dsvL)
            cov = list(set(cov))

            self.mat = get_cov_matrix(self.grid_size, self.df, cov)
            return_value = [self.run(vid) for vid in N]

            time_t_prev = time_t

            for vid, next_loc in enumerate(return_value):
                dsv_paths[vid].append(next_loc)

        print("DSV Locations At End ({}):{}".format(time_t, self.dsvL))
        return dsv_paths

class REASSIGN:
    '''
    REASSIGN \cite{lesmana2019balancing}: REASSIGN recommends the neighbor location with the longest waiting period, focusing on selecting the next location based on improving fairness.
    '''
    def __init__(self, df, grid_size, n=1, tf=5, beta=60, seed=42, km_time=2) -> None:
        random.seed(seed)
        self.name = "REASSIGN-F"
        self.df = df
        self.n = n
        self.timeframe = tf
        self.grid_size = grid_size
        self.km_time = int(km_time*grid_size/1000) # km_time to complete 1 km (move to next grid)
        self.path_length = tf//self.km_time # path length per interval

        n_rows, n_cols, nPosns, n_rows_valid, n_cols_valid, nPosns_valid = gridSize_info_dict[self.grid_size]
        self.map_area = np.array([[0 for i in range(n_cols_valid)] for j in range(n_rows_valid)])
        self.area_nrows, self.area_ncols = size_dict[self.n]
        self.pos_neis, self.tfm_map, self.valid_div_pos, self.pos_to_div, self.valid_coords, self.div_coords = generate_neighbors_list_wBorders(self.n, grid_size, self.df, self.name, self.timeframe, self.km_time)
        self.tfm_maps = []
        self.dsvL = [random.sample(list(self.valid_div_pos[i]), 1)[0] for i in range(n)]
        self.dsv_paths = {i:deque([self.dsvL[i]]) for i in range(self.n)}

        print("\n{} Model Object Created".format(self.name))
        print("\nValid Locations Per Region", [len(self.valid_div_pos[i]) for i in range(self.n)])
        print('\nArea Border of Division (start_row, end_row, start_col, end_col):', self.div_coords)
        print("\nStudy Area Gird Shape: ", self.tfm_map.shape)
        print("DSV Starting Locations: ",self.dsvL)
        print("DSV Starting Grid Positions: ", [(pos//n_cols, pos%n_cols) for pos in self.dsvL])
        print('Path Length: ', self.path_length)

    def take_action(self, state, dir_y, dir_x, globalY, globalX):
        next_state = state.copy()

        curY, curX = np.where(next_state == dsv_loc_val)

        dirX, dirY = dir_x, dir_y

        nextX, nextY = curX + dirX, curY + dirY

        next_globalX, next_globalY = globalX + dirX, globalY + dirY # to update the global study area

        next_state[curY, curX] = 0 # previous dsv location

        next_state[next_state >= 0] += 1

        next_state[curY, curX] = 0 # previous dsv location

        next_state[nextY, nextX] = dsv_loc_val # new dsv location

        return next_state, nextY, nextX, next_globalY, next_globalX

    def margvals_returnvals(self, state, dir_y, dir_x, pos):

        cur_y, cur_x  = np.where(state == dsv_loc_val)

        y, x = cur_y+dir_y, cur_x+dir_x
        val = state[y, x]

        if val < -1000:
            return 0.0 - 5.0

        temporal_fairness_reward = val/60 

        reward = temporal_fairness_reward

        return reward

    def run(self, vid):
        cur_pos = self.dsvL[vid]
        dsv_path = [cur_pos]
        global_dsvY, global_dsvX = pos_to_coord(self.grid_size, cur_pos)
        div_id, localY, localX = self.pos_to_div[(global_dsvY, global_dsvX)]

        state = gen_state(self.tfm_maps[div_id], localY, localX, self.timeframe, self.km_time)

        for t in range(self.path_length):

            cur_y, cur_x = pos_to_coord(self.grid_size, self.dsvL[vid])

            neis = set([nei for nei in self.pos_neis[self.dsvL[vid]]])
            neis_coord = [list(pos_to_coord(self.grid_size, nei)) for nei in neis]

            ele_vals_gain = [ self.margvals_returnvals(state, nei[0]-cur_y, nei[1]-cur_x, cur_pos) for nei in neis_coord]

            argmax_gain = np.argmax( ele_vals_gain );
            next_pos = list(neis)[argmax_gain]
            next_y, next_x = pos_to_coord(self.grid_size, next_pos)

            dir_y, dir_x = next_y - cur_y, next_x - cur_x

            next_state, local_dsvY, local_dsvX, global_dsvY, global_dsvX = self.take_action(state, dir_y, dir_x, global_dsvY, global_dsvX)

            state = next_state.copy() # Updating State for next iteration
            next_pos = coord_to_pos(self.grid_size, global_dsvY, global_dsvX)
            dsv_path.append(next_pos)
            self.dsvL[vid] = next_pos

        self.tfm_maps[div_id], self.tfm_map = update_map(self.tfm_map, self.tfm_maps[div_id], state, localY, localX, self.div_coords[div_id], self.timeframe, self.km_time )
        self.dsvL[vid] = next_pos
        return dsv_path

    def parallel_run(self, curTime):

        N = [ele for ele in range(self.n)]
        nthreads_obj = self.n
        dsv_paths = {i:[self.dsvL[i]] for i in range(self.n)}
        time_t_prev = curTime
        times_print = [str(curTime)]
        paths_print = []

        time_t_prev = curTime
        next_time_t = time_t_prev + timedelta(minutes=self.timeframe)

        for t in range(self.path_length): # range(self.path_int+1)
            interval = timedelta(minutes=self.km_time-1)
            time_t = time_t_prev + interval

            data_interval = fetch_rows(self.df, time_t_prev, time_t)

            locs_interval_ts = get_FHV_cov_coord(self.grid_size, data_interval, self.valid_coords)

            self.tfm_map = update_tfm_sfm_tEnd(self.tfm_map, self.valid_coords, self.path_length, testing=True)

            self.tfm_map = update_tfm_sfm_cov(self.tfm_map, locs_interval_ts, self.path_length, testing=True)

            times_print.append(str(time_t))
            time_t_prev = time_t_prev + timedelta(minutes=self.km_time)
        self.tfm_maps = get_mat2D_nDivs_fromFullMap(self.grid_size, self.df, self.tfm_map, self.n, self.path_length, self.valid_coords)

        return_value = [self.run(vid) for vid in N]

        for vid, best_path in enumerate(return_value):
            dsv_paths[vid] = best_path

        print("DSV Locations At End ({}):{}".format(next_time_t, self.dsvL))  
        return dsv_paths

class SDPR:
    '''
    SDPR (Shortest Diversified Path Routing): Based on the K-Shortest Diversified Path Routing (KSDPR) algorithm of \cite{lai2022optimized}, with $k=1$ to select the shortest path to the reachable zone with the longest average waiting period.
    '''
    def __init__(self, df, grid_size, n=1, tf=5, beta=60, seed=42, km_time=2) -> None:
        random.seed(seed)
        self.name = "SDPR-F"
        self.df = df
        self.n = n
        self.timeframe = tf
        self.grid_size = grid_size
        self.km_time = int(km_time*grid_size/1000) # km_time to complete 1 km (move to next grid)
        self.path_length = tf//self.km_time # path length per interval

        n_rows, n_cols, nPosns, n_rows_valid, n_cols_valid, nPosns_valid = gridSize_info_dict[self.grid_size]
        self.map_area = np.array([[0 for i in range(n_cols_valid)] for j in range(n_rows_valid)])
        self.area_nrows, self.area_ncols = size_dict[self.n]
        self.pos_neis, self.tfm_map, self.valid_div_pos, self.pos_to_div, self.valid_coords, self.div_coords = generate_neighbors_list_wBorders(self.n, grid_size, self.df, self.name, self.timeframe, self.km_time)

        self.tfm_maps = []
        self.dsvL = [random.sample(list(self.valid_div_pos[i]), 1)[0] for i in range(n)]
        self.dsv_paths = {i:deque([self.dsvL[i]]) for i in range(self.n)}

        print("\n{} Model Object Created".format(self.name))
        print("\nValid Locations Per Region", [len(self.valid_div_pos[i]) for i in range(self.n)])
        print('\nArea Border of Division (start_row, end_row, start_col, end_col):', self.div_coords)
        print("\nStudy Area Gird Shape: ", self.tfm_map.shape)
        print("DSV Starting Locations: ",self.dsvL)
        print("DSV Starting Grid Positions: ", [(pos//n_cols, pos%n_cols) for pos in self.dsvL])
        print('Path Length: ', self.path_length)

    def take_action(self, state, start_pos, next_pos, next_shift_y, next_shift_x, dir_y, dir_x):
        next_state = state.copy()

        curY, curX = np.where(next_state == dsv_loc_val)
        nextY, nextX = curY+next_shift_y, curX+next_shift_x
        cur_globalY, cur_globalX = pos_to_coord(self.grid_size, start_pos)
        next_globalY, next_globalX = pos_to_coord(self.grid_size, next_pos) # to update the global study area
        dirX, dirY = dir_x, dir_y

        next_state[curY, curX] = 0 # previous dsv location

        while next_globalY != cur_globalY or next_globalX != cur_globalX:

            next_state[next_state >= 0] += 1
            curY, curX = curY+dirY, curX+dirX
            next_state[curY, curX] = 0
            cur_globalY, cur_globalX = cur_globalY + dirX, cur_globalX + dirY

        next_state[nextY, nextX] = dsv_loc_val # new dsv location

        return next_state, nextY, nextX, next_globalY, next_globalX

    def margvals_returnvals(self, state, start_pos, dir_y, dir_x, localY, localX, t=0):

        orig_y, orig_x  = np.where(state == dsv_loc_val)

        y, x = orig_y+(dir_y*self.path_length), orig_x+(dir_x*self.path_length)

        cur_y, cur_x = y, x
        val = state[cur_y, cur_x]
        dsv_y, dsv_x = pos_to_coord(self.grid_size, start_pos)
        if not isinstance(dsv_y, int):
            dsv_y, dsv_x = dsv_y.item(), dsv_x.item()

        ny, nx = dsv_y+(dir_y*self.path_length), dsv_x+(dir_x*self.path_length)

        while val < -1000 or (ny, nx) not in self.pos_to_div: 
            cur_y, cur_x = cur_y - dir_y, cur_x - dir_x
            val = state[cur_y, cur_x]
            ny, nx = ny - dir_y, nx - dir_x

        if cur_y == orig_y and cur_x == orig_x:
            return [0.0 - 5.0, np.array([0]), np.array([0])]

        shift_y, shift_x = cur_y-orig_y, cur_x-orig_x

        r_center_y, r_center_x = dsv_y+shift_y, dsv_x+shift_x
        if not isinstance(r_center_y, int):
            r_center_y, r_center_x = r_center_y.item(), r_center_x.item()

        div_id, localR_Y, localR_X = self.pos_to_div[(r_center_y, r_center_x)]

        region = gen_state_region(self.tfm_maps[div_id], localR_Y, localR_X, self.km_time*max(abs(shift_y), abs(shift_x)).item(), self.km_time)
        region_fairness_reward = np.mean(region[region >= 0])

        reward =  region_fairness_reward
        return [reward, shift_y, shift_x]

    def run(self, vid):

        cur_pos = self.dsvL[vid]
        if not isinstance(cur_pos, int):
            cur_pos = cur_pos.item()
        dsv_path = [cur_pos]
        global_dsvY, global_dsvX = pos_to_coord(self.grid_size, cur_pos)

        div_id, localY, localX = self.pos_to_div[(global_dsvY, global_dsvX)]

        state = gen_state(self.tfm_maps[div_id], localY, localX, self.timeframe, self.km_time)
        directions = [[-1, -1],[0, -1],[1, -1], [-1, 0], [1, 0], [-1, 1], [0, 1], [1, 1]]

        cur_y, cur_x = pos_to_coord(self.grid_size, self.dsvL[vid])

        ele_vals_result = [ self.margvals_returnvals(state, self.dsvL[vid], dy, dx, localY, localX) for dx, dy in directions]

        ele_vals_gain = [ele_vals_result[i][0] for i in range(len(ele_vals_result))]
        argmax_gain = np.argmax( ele_vals_gain );
        next_pos_shift_y, next_pos_shift_x = ele_vals_result[argmax_gain][1], ele_vals_result[argmax_gain][2]
        next_y, next_x = cur_y+next_pos_shift_y, cur_x+next_pos_shift_x
        next_pos = coord_to_pos(self.grid_size, next_y, next_x)

        dir_x, dir_y = directions[argmax_gain]                  
        while cur_y != next_y or cur_x != next_x:
            cur_y, cur_x = cur_y+dir_y, cur_x+dir_x
            if not isinstance(cur_y, int):
                cur_y, cur_x = cur_y.item(), cur_x.item()
            dsv_path.append(coord_to_pos(self.grid_size, cur_y, cur_x))

        dir_y, dir_x = next_y - cur_y, next_x - cur_x

        next_state, local_dsvY, local_dsvX, global_dsvY, global_dsvX = self.take_action(state, cur_pos, next_pos.item(), next_pos_shift_y, next_pos_shift_x, directions[argmax_gain][0], directions[argmax_gain][1])

        state = next_state.copy() # Updating State for next iteration

        self.tfm_maps[div_id], self.tfm_map = update_map(self.tfm_map, self.tfm_maps[div_id], state, localY, localX, self.div_coords[div_id], self.timeframe, self.km_time )
        self.dsvL[vid] = next_pos.item() if not isinstance(next_pos, int) else next_pos
        print(dsv_path)
        return dsv_path

    def parallel_run(self, curTime):

        N = [ele for ele in range(self.n)]
        nthreads_obj = self.n
        dsv_paths = {i:[self.dsvL[i]] for i in range(self.n)}
        time_t_prev = curTime
        times_print = [str(curTime)]
        paths_print = []

        time_t_prev = curTime
        next_time_t = time_t_prev + timedelta(minutes=self.timeframe)

        for t in range(self.path_length): # range(self.path_int+1)
            interval = timedelta(minutes=self.km_time-1)
            time_t = time_t_prev + interval

            data_interval = fetch_rows(self.df, time_t_prev, time_t)

            locs_interval_ts = get_FHV_cov_coord(self.grid_size, data_interval, self.valid_coords)

            self.tfm_map = update_tfm_sfm_tEnd(self.tfm_map, self.valid_coords, self.path_length, testing=True)

            self.tfm_map = update_tfm_sfm_cov(self.tfm_map, locs_interval_ts, self.path_length, testing=True)

            times_print.append(str(time_t))
            time_t_prev = time_t_prev + timedelta(minutes=self.km_time)
        self.tfm_maps = get_mat2D_nDivs_fromFullMap(self.grid_size, self.df, self.tfm_map, self.n, self.path_length, self.valid_coords)

        return_value = [self.run(vid) for vid in N]

        for vid, best_path in enumerate(return_value):
            dsv_paths[vid] = best_path
            print(best_path)

        print("DSV Locations At End ({}):{}".format(next_time_t, self.dsvL))  
        return dsv_paths

class AGD:
    '''
    AGD (Aggregated Greedy Dispatch)  \cite{stollar2018fleet}: AGD recommends the shortest path to the least aggregate coverage zone among the 8-directional reachable zones from the current position of DSV.
    '''
    
    def __init__(self, df, grid_size,n=1, tf=5, beta=60, seed=42, km_time=2) -> None:
        random.seed(seed)
        self.name = "AGD-C"
        self.df = df
        self.n = n
        self.timeframe = tf
        self.grid_size = grid_size
        self.km_time = int(km_time*grid_size/1000) # km_time to complete 1 km (move to next grid)
        self.path_length = tf//self.km_time # path length per interval

        n_rows, n_cols, nPosns, n_rows_valid, n_cols_valid, nPosns_valid = gridSize_info_dict[self.grid_size]
        self.map_area = np.array([[0 for i in range(n_cols_valid)] for j in range(n_rows_valid)])
        self.area_nrows, self.area_ncols = size_dict[self.n]
        self.pos_neis, self.tfm_map, self.valid_div_pos, self.pos_to_div, self.valid_coords, self.div_coords = generate_neighbors_list_wBorders(self.n, grid_size, self.df, self.name, self.timeframe, self.km_time)
        self.tfm_maps = []
        self.dsvL = [random.sample(list(self.valid_div_pos[i]), 1)[0] for i in range(n)]
        self.dsv_paths = {i:deque([self.dsvL[i]]) for i in range(self.n)}

        print("\n{} Model Object Created".format(self.name))
        print("\nValid Locations Per Region", [len(self.valid_div_pos[i]) for i in range(self.n)])
        print('\nArea Border of Division (start_row, end_row, start_col, end_col):', self.div_coords)
        print("\nStudy Area Gird Shape: ", self.tfm_map.shape)
        print("DSV Starting Locations: ",self.dsvL)
        print("DSV Starting Grid Positions: ", [(pos//n_cols, pos%n_cols) for pos in self.dsvL])
        print('Path Length: ', self.path_length)

    def margvals_returnvals(self, state, start_pos, dir_y, dir_x, div_id, t=0):
        orig_y, orig_x  = state.shape[0]//2, state.shape[1]//2

        y, x = orig_y+(dir_y*self.path_length), orig_x+(dir_x*self.path_length)

        cur_y, cur_x = y, x

        val = state[cur_y, cur_x]

        dsv_y, dsv_x = pos_to_coord(self.grid_size, start_pos)
        if not isinstance(dsv_y, int):
            dsv_y, dsv_x = dsv_y.item(), dsv_x.item()

        ny, nx = dsv_y+(dir_y*self.path_length), dsv_x+(dir_x*self.path_length)

        while val < 0 or (ny, nx) not in self.pos_to_div: 
            cur_y, cur_x = cur_y - dir_y, cur_x - dir_x
            val = state[cur_y, cur_x]
            ny, nx = ny - dir_y, nx - dir_x

        if cur_y == orig_y and cur_x == orig_x:
            return [0.0 - 5.0, np.array([0]), np.array([0])]

        shift_y, shift_x = cur_y-orig_y, cur_x-orig_x

        dsv_y, dsv_x = pos_to_coord(self.grid_size, start_pos)
        r_center_y, r_center_x = dsv_y+shift_y, dsv_x+shift_x
        if not isinstance(r_center_y, int):
            r_center_y, r_center_x = r_center_y.item(), r_center_x.item()
        div_id, localR_Y, localR_X = self.pos_to_div[(r_center_y, r_center_x)]

        if not isinstance(shift_x, int):
            shift_y, shift_x = shift_y.item(), shift_x.item()
        region = gen_state_coverage(self.tfm_maps[div_id], localR_Y, localR_X, self.km_time*max(abs(shift_y), abs(shift_x)), self.km_time)
        region_coverage_reward = np.sum(region > 0)
        reward =  region_coverage_reward
        return [reward, shift_y, shift_x]
    def run(self, vid):
        ''' 
        INPUTS:

        OUTPUTS:
        '''    

        cur_pos = self.dsvL[vid]
        if not isinstance(cur_pos, int):
            cur_pos = cur_pos.item()
        dsv_path = [cur_pos]
        global_dsvY, global_dsvX = pos_to_coord(self.grid_size, cur_pos)

        div_id, localY, localX = self.pos_to_div[(global_dsvY, global_dsvX)]

        state = gen_state_coverage(self.tfm_maps[div_id], localY, localX, self.timeframe, self.km_time)

        directions = [[-1, -1],[0, -1],[1, -1], [-1, 0], [1, 0], [-1, 1], [0, 1], [1, 1]]
        cur_y, cur_x = pos_to_coord(self.grid_size, self.dsvL[vid])

        ele_vals_result = [ self.margvals_returnvals(state, self.dsvL[vid], dy, dx, div_id) for dx, dy in directions]

        ele_vals_gain = [ele_vals_result[i][0] for i in range(len(ele_vals_result))]

        argmax_gain = np.argmax( ele_vals_gain );
        next_pos_shift_y, next_pos_shift_x = ele_vals_result[argmax_gain][1], ele_vals_result[argmax_gain][2]
        next_y, next_x = cur_y+next_pos_shift_y, cur_x+next_pos_shift_x
        next_pos = coord_to_pos(self.grid_size, next_y, next_x)

        dir_x, dir_y = directions[argmax_gain]                  
        while cur_y != next_y or cur_x != next_x:
            cur_y, cur_x = cur_y+dir_y, cur_x+dir_x
            if not isinstance(cur_y, int):
                cur_y, cur_x = cur_y.item(), cur_x.item()
            dsv_path.append(coord_to_pos(self.grid_size, cur_y, cur_x))
        self.dsvL[vid] = next_pos.item() if not isinstance(next_pos, int) else next_pos
        return dsv_path

    def parallel_run(self, curTime):
        ''' 
        INPUTS:

        OUTPUTS:

        '''
        N = [ele for ele in range(self.n)]
        nthreads_obj = self.n
        dsv_paths = {i:[self.dsvL[i]] for i in range(self.n)}
        time_t_prev = curTime
        times_print = [str(curTime)]
        paths_print = []

        time_t_prev = curTime
        interval = timedelta(minutes=self.timeframe-1)
        time_t = time_t_prev + interval
        time_t_pastHour = time_t - timedelta(minutes=60-self.timeframe)
        print("Time_t_Pasthour: ", time_t_pastHour, "  Time_t: ", time_t)
        data = fetch_rows(self.df, time_t_pastHour, time_t)
        cov = get_FHV_cov_coord(self.grid_size, data, self.valid_coords)
        for i in range(self.n):
            if not isinstance(self.dsvL[i], int):

                self.dsvL[i] = self.dsvL[i].item()
            cov.add(pos_to_coord(self.grid_size, self.dsvL[i]))

        self.tfm_map = get_cov_matrix_2D(self.grid_size, self.df, cov)

        self.tfm_maps = get_mat2D_nDivs_fromFullMap(self.grid_size, self.df, self.tfm_map, self.n, self.path_length, self.valid_coords)

        return_value = [self.run(vid) for vid in N]

        time_t_prev = time_t

        for vid, best_path in enumerate(return_value):
            dsv_paths[vid] = best_path
            print(best_path)

        print("DSV Locations At End ({}):{}".format(time_t, self.dsvL))
        return dsv_paths
'''
OUR FUSE MODEL ANS VARIANTS
'''

class FUSE_DQN_Train:
    def __init__(self, state_data, val_data, train_Samples, tf=20, nDiv=1, action_size=8, episodes=100, batch_size=128, alpha=0.01, gamma=0.9, epsilon=0.8, epsilon_decay=0.999, epsilon_min=0.01, update_freq=5):
        self.name = "fuse_dqn"
        self.all_data = state_data
        self.state_size = train_Samples
        self.val_data = val_data
        self.action_size = action_size
        self.memory = deque(maxlen=1000000)
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration-exploitation trade-off
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.update_freq = update_freq
        self.batch_size = batch_size
        self.episodes = episodes
        self.path_length = tf//2 # Since we cover 1km (one grid) every 2 min (30 km/h)
        self.tf = tf

        self.map_action_to_direction = {0:(-1, -1), 
                                        1:(0, -1), 
                                        2:(1, -1), 
                                        3:(-1, 0), 
                                        4:(1, 0), 
                                        5:(-1, 1), 
                                        6:(0, 1), 
                                        7:(1, 1)}

        self.map_direction_to_action = {(-1, -1):0, 
                                        (0, -1):1, 
                                        (1, -1):2, 
                                        (-1, 0):3, 
                                        (1, 0):4, 
                                        (-1, 1):5, 
                                        (0, 1):6, 
                                        (1, 1):7}
        sample_data = self.all_data[:, :, 0]
        sample_feature = self.gen_features_mat(sample_data, 0)
        self.input_shape = sample_feature.shape

        self.model = self._build_model()
        self.target_model = self._build_model()
        self.best_reward_model = self._build_model()

        self.checkpoint_path_best = f'models_new/best_{self.name}_r_model.h5'
        self.checkpoint_path_t = f'models_new/{self.name}_t_model.h5'
        self.checkpoint_path_q = f'models_new/{self.name}_q_model.h5'

        self.best_reward_val = 0.0

    def _build_model(self):
        print('\nBuilding {} Q-Network with Input Shape: {}\n'.format(self.name, self.input_shape))
        model = models.Sequential([
                layers.Dense(64, activation='relu', input_shape=self.input_shape),
                layers.Dense(64, activation='relu', kernel_regularizer='l2'),
                layers.Dense(self.action_size, activation='relu'),
                layers.Dense(1, activation='linear')
            ])

        model.compile(loss='huber_loss', optimizer=optimizers.Adam(learning_rate=self.alpha), metrics=['mae'])
        return model

    def remember(self, state, action, reward, next_state, t):
        self.memory.append((state.copy(), action, reward, next_state.copy(), t))

    def get_observations(self, state, action, t):
        next_state = state.copy()

        curY, curX = np.where(next_state == dsv_loc_val)

        dirX, dirY = self.map_action_to_direction[action]

        x, y = curX + dirX, curY + dirY
        if not ((0 <= y < next_state.shape[0]) and (0 <= x < next_state.shape[1])).any():
            return np.array([-1, -1, -1, -1, -1])

        next_state[curY, curX] = 0 # previous dsv location

        val = next_state[y, x][0]

        cov = val//60 if val >= 0 else -1

        tfm = val/60 if val >= 0 else -1

        n_neighbors = [(dx, dy) for dx in range(-1, 2) for dy in range(-1, 2) if dx != 0 or dy != 0]

        n_nei = sum(1 for dx, dy in n_neighbors if 0 <= y + dy < next_state.shape[0] and 0 <= x + dx < next_state.shape[1] and next_state[y + dy, x + dx] >= 60)  if val >= 0 else -1 

        nei_tfm = [next_state[y + dy, x + dx] for dx, dy in n_neighbors if 0 <= y + dy < next_state.shape[0] and 0 <= x + dx < next_state.shape[1] and next_state[y + dy, x + dx] >= 0]

        sfm = np.mean(nei_tfm)/60  if val >= 0 else -1 

        path_rem = self.path_length - t - 1

        center_x, center_y = x.item(), y.item()
        start_y, end_y =  max(center_y - path_rem, 0), min(center_y + path_rem, next_state.shape[0])
        start_x, end_x =  max(center_x - path_rem, 0), min(center_x + path_rem, next_state.shape[1])
        region = next_state[start_y:end_y+1, start_x:end_x+1]

        rfm = np.mean(region[region >= 0])/60 if val >= 0 else -1

        return np.array([cov, tfm, n_nei, sfm, rfm])
     
    def gen_features_mat(self, state, t):
        state_copy = state.copy()
        y, x = np.where(state_copy == dsv_loc_val)
        features_mat = [self.get_observations(state_copy, action, t) for action in self.map_action_to_direction]
        features_mat = np.array(features_mat)
        min_values = np.min(features_mat, axis=0)
        max_values = np.max(features_mat, axis=0)
        normalized_data = (features_mat - min_values) / (max_values - min_values + 1e-8)

        return normalized_data #features_mat #normalized_data

    def valid_actions(self, state):
        state_copy = state.copy()
        y, x = np.where(state_copy == dsv_loc_val)
        n_neighbors = [(dx, dy) for dx in range(-1, 2) for dy in range(-1, 2) if dx != 0 or dy != 0]

        valid_dirs = [(dx, dy) for dx, dy in n_neighbors if 0 <= y + dy < state.shape[0] and 0 <= x + dx < state.shape[1] and state[y + dy, x + dx] != invalid_loc_val]
        valid_acts = [self.map_direction_to_action[(dx, dy)] for dx, dy in valid_dirs]
        return sorted(valid_acts)

    def choose_action(self, state_features, state, t):

        valid_acts = self.valid_actions(state)
        if np.random.rand() <= self.epsilon:
            return random.choice(valid_acts)  # Explore
        else:
            q_values = self.model.predict(np.array([state_features.copy()]))
            state_copy = state.copy()

            return valid_acts[np.argmax(q_values[0][valid_acts])]  # Exploit

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        states, targets = [], []

        for state, action, reward, next_state, t in minibatch:
            target = reward

            state_features = self.gen_features_mat(state, t)
            if next_state is not None:

                next_state_features = self.gen_features_mat(next_state, t+1)

                next_state_preds = self.target_model.predict(np.array([next_state_features]))

                target = reward + self.gamma * np.amax(next_state_preds[0])

            target_f = self.model.predict(np.array([state_features]))

            target_f[0][action] = target

            states.append(state_features)
            targets.append(target_f[0])

        states = np.array(states)
        targets = np.array(targets)
        test_loss = self.model.evaluate(states, targets, verbose=2)

        self.model.fit(states, targets, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    def step(self, state, action, t, return_state=True):
        next_state = state.copy()

        curY, curX = np.where(next_state == dsv_loc_val)

        dirX, dirY = self.map_action_to_direction[action]

        nextX, nextY = curX + dirX, curY + dirY

        next_state[curY, curX] = 0 # previous dsv location

        reward = self.calculate_reward(next_state.copy(), nextY, nextX, t)

        next_state[next_state >= 0] += 1

        next_state[curY, curX] = 0 # previous dsv location

        next_state[nextY, nextX] = dsv_loc_val # new dsv location

        if not return_state:
            return reward 
        return reward, next_state

    def calculate_reward(self, state, y, x, t):
        val = state[y, x]
        if val < -1000:
            return 0.0 - 5.0

        temporal_fairness_reward = val/60 if val >= 0 else -1

        n_neighbors = [(dx, dy) for dx in range(-1, 2) for dy in range(-1, 2) if dx != 0 or dy != 0]
        nei_tfm = [state[y + dy, x + dx]/60 for dx, dy in n_neighbors if 0 <= y + dy < state.shape[0] and 0 <= x + dx < state.shape[1] and state[y + dy, x + dx] >= 0]
        local_fairness_reward = np.mean(nei_tfm) if val >= 0 else -1
        center_y, center_x = y.item(), x.item()
        path_rem = self.path_length - t -1
        start_y, end_y =  max(center_y - path_rem, 0), min(center_y + path_rem, state.shape[0])
        start_x, end_x =  max(center_x - path_rem, 0), min(center_x + path_rem, state.shape[1])
        region = state[start_y:end_y+1, start_x:end_x+1]

        region_fairness_reward = np.mean(region[region >= 0])/60 if val >= 0 else -1

        reward = temporal_fairness_reward + local_fairness_reward + region_fairness_reward
        return reward

    def train(self):
        print(f'\nStarting {self.name} Training...\n')
        best_reward = 0.0

        for episode in range(self.episodes):
            total_reward = 0.0
            train_Samples = 10000
            sampled_indices = np.random.choice(self.all_data.shape[2], train_Samples, replace=False)

            self.train_data = self.all_data[:, :, sampled_indices]
            print('\EPISODE: {}  Randomly Sampled Training State Data Loaded ({} samples) of Shape: {}'.format(episode, self.train_data.shape[2], self.train_data[:, :, 0].shape))

            states = self.train_data.transpose((2, 0, 1))
            print('\nstates.shape: ', states.shape)

            for t in range(self.path_length):

                state_features_mat = np.array([self.gen_features_mat(state, t) for state in states])

                print('state_features_mat.shape: ', state_features_mat.shape)

                actions = np.array([self.choose_action(state_features, states[i], t) for i, state_features in enumerate(state_features_mat)])

                rewards, next_states = zip(*[self.step(states[i], action, t) for i, action in enumerate(actions)])
                for i in range(self.state_size):
                    self.remember(states[i], actions[i], rewards[i], next_states[i], t)

                total_reward += np.sum(rewards)

                states = [next_states[i].copy() for i in range(self.train_data.shape[2])]

                print('\nTIMESTEP: ', t, '  reward: ', np.sum(rewards))

                self.replay(batch_size=self.batch_size)

            print('\nCompleted {} samples; Reward = {}'.format(self.state_size, total_reward))
            print(f"\nEPISODE {episode + 1}, Total Training Reward: {total_reward}, Best Reward: {best_reward}")

            total_reward_val = self.validation()

            if episode == 0:
                self.best_reward_val = total_reward_val

            if self.best_reward_val <= total_reward_val:
                self.best_reward_val = total_reward_val
                self.best_reward_model.set_weights(self.model.get_weights())
                self.best_reward_model.save(self.checkpoint_path_best, save_format='h5')
                print(f"\nBest Model Saved to {self.checkpoint_path_best}")
                
    def choose_action_val(self, state_features, state):
        valid_acts = self.valid_actions(state)

        q_values = self.model.predict(np.array([state_features.copy()]))
        return valid_acts[np.argmax(q_values[0][valid_acts])], np.max(q_values[0][valid_acts])

    def validation(self):
        states = self.val_data.transpose((2, 0, 1))
        print('Validation states.shape: ', states.shape)

        total_reward = 0.0
        for i, state in enumerate(states):

            for t in range(self.path_length):

                state_features = self.gen_features_mat(state, t)

                action, q_val = self.choose_action_val(state_features, state)
                reward, next_state = self.step(state, action, t)

                loss = tf.keras.losses.Huber()(reward, q_val)
                state = next_state.copy() # Updating State for next iteration
                total_reward += reward
        print('\nVALIDATION Completed {} samples; Total Validation Reward = {};  Best Validation Reward= {}'.format(states.shape[0], total_reward, self.best_reward_val))

        return total_reward

class FUSE_DQN_Test:

    def __init__(self, df, grid_size, n, tf, beta=60, seed=42, km_time=2) -> None:
        random.seed(seed)
        self.name = "fuse"
        self.df = df
        self.n = n
        self.timeframe = tf
        self.grid_size = grid_size
        self.km_time = int(km_time*grid_size/1000) # km_time to complete 1 km (move to next grid)
        self.path_length = tf//self.km_time # path length per interval

        n_rows, n_cols, nPosns, n_rows_valid, n_cols_valid, nPosns_valid = gridSize_info_dict[self.grid_size]
        self.map_area = np.array([[0 for i in range(n_cols_valid)] for j in range(n_rows_valid)])
        self.area_nrows, self.area_ncols = size_dict[self.n]
        self.pos_neis, self.tfm_map, self.valid_div_pos, self.pos_to_div, self.valid_coords, self.div_coords = generate_neighbors_list_wBorders(self.n, grid_size, self.df, self.name, self.timeframe, self.km_time)
        self.tfm_maps = []
        self.dsvL = [random.sample(list(self.valid_div_pos[i]), 1)[0] for i in range(n)]
        self.dsv_paths = {i:deque([self.dsvL[i]]) for i in range(self.n)}

        self.map_action_to_direction = {0:(-1, -1), 
                                        1:(0, -1), 
                                        2:(1, -1), 
                                        3:(-1, 0), 
                                        4:(1, 0), 
                                        5:(-1, 1), 
                                        6:(0, 1), 
                                        7:(1, 1)}

        self.map_direction_to_action = {(-1, -1):0, 
                                        (0, -1):1, 
                                        (1, -1):2, 
                                        (-1, 0):3, 
                                        (1, 0):4, 
                                        (-1, 1):5, 
                                        (0, 1):6, 
                                        (1, 1):7}

        self.checkpoint_path = f'models_new/best_{self.name}_r_model.h5' #'models_new/best_model_fcn_new_1.h5'
        self.model = models.load_model(self.checkpoint_path) #, custom_objects=custom_objects)    

        print("Q-Network Model Loaded:")
        print(self.model.summary())

        print("\n{} Model Object Created".format(self.name))
        print("\nValid Locations Per Region", [len(self.valid_div_pos[i]) for i in range(self.n)])
        print('\nArea Border of Division (start_row, end_row, start_col, end_col):', self.div_coords)
        print("\nStudy Area Gird Shape: ", self.tfm_map.shape)
        print("DSV Starting Locations: ",self.dsvL)
        print("DSV Starting Grid Positions: ", [(pos//n_cols, pos%n_cols) for pos in self.dsvL])
        print('Path Length: ', self.path_length)
                
    def valid_actions(self, state):
        state_copy = state.copy()
        y, x = np.where(state_copy == dsv_loc_val)
        n_neighbors = [(dx, dy) for dx in range(-1, 2) for dy in range(-1, 2) if dx != 0 or dy != 0]

        valid_dirs = [(dx, dy) for dx, dy in n_neighbors if 0 <= y + dy < state.shape[0] and 0 <= x + dx < state.shape[1] and state[y + dy, x + dx] != invalid_loc_val]
        valid_acts = [self.map_direction_to_action[(dx, dy)] for dx, dy in valid_dirs]
        return sorted(valid_acts)

    def choose_action(self, state_features, state):
        valid_acts = self.valid_actions(state)

        q_values = self.model.predict(np.array([state_features.copy()]))
        return valid_acts[np.argmax(q_values[0][valid_acts])]

    def take_action(self, state, action, globalY, globalX):
        next_state = state.copy()

        curY, curX = np.where(next_state == dsv_loc_val)

        dirX, dirY = self.map_action_to_direction[action]

        nextX, nextY = curX + dirX, curY + dirY

        next_globalX, next_globalY = globalX + dirX, globalY + dirY # to update the global study area

        next_state[curY, curX] = 0 # previous dsv location

        next_state[next_state >= 0] += 1

        next_state[curY, curX] = 0 # previous dsv location

        next_state[nextY, nextX] = dsv_loc_val # new dsv location

        return next_state, nextY, nextX, next_globalY, next_globalX

    def run(self, vid):
        ''' 
        INPUTS:

        OUTPUTS:
        '''    

        cur_pos = self.dsvL[vid]
        dsv_path = [cur_pos]
        global_dsvY, global_dsvX = pos_to_coord(self.grid_size, cur_pos)
        div_id, localY, localX = self.pos_to_div[(global_dsvY, global_dsvX)]

        state = gen_state(self.tfm_maps[div_id], localY, localX, self.timeframe, self.km_time)

        for t in range(self.path_length):

            state_features = self.gen_features_mat(state, t)

            action = self.choose_action(state_features, state)
            next_state, local_dsvY, local_dsvX, global_dsvY, global_dsvX = self.take_action(state, action, global_dsvY, global_dsvX)

            state = next_state.copy() # Updating State for next iteration
            next_pos = coord_to_pos(self.grid_size, global_dsvY, global_dsvX)
            dsv_path.append(next_pos)

        self.tfm_maps[div_id], self.tfm_map = update_map(self.tfm_map, self.tfm_maps[div_id], state, localY, localX, self.div_coords[div_id], self.timeframe, self.km_time )
        self.dsvL[vid] = next_pos
        return dsv_path

    def parallel_run(self, curTime):
        ''' 
        INPUTS:

        OUTPUTS:

        '''
        N = [ele for ele in range(self.n)]
        nthreads_obj = self.n
        dsv_paths = {i:[self.dsvL[i]] for i in range(self.n)}
        time_t_prev = curTime
        times_print = [str(curTime)]
        paths_print = []

        time_t_prev = curTime
        next_time_t = time_t_prev + timedelta(minutes=self.timeframe)

        for t in range(self.path_length): # range(self.path_int+1)
            interval = timedelta(minutes=self.km_time-1)
            time_t = time_t_prev + interval

            data_interval = fetch_rows(self.df, time_t_prev, time_t)

            locs_interval_ts = get_FHV_cov_coord(self.grid_size, data_interval, self.valid_coords)

            self.tfm_map = update_tfm_sfm_tEnd(self.tfm_map, self.valid_coords, self.path_length, testing=True)

            self.tfm_map = update_tfm_sfm_cov(self.tfm_map, locs_interval_ts, self.path_length, testing=True)

            times_print.append(str(time_t))
            time_t_prev = time_t_prev + timedelta(minutes=self.km_time)
        self.tfm_maps = get_mat2D_nDivs_fromFullMap(self.grid_size, self.df, self.tfm_map, self.n, self.path_length, self.valid_coords)

        return_value = [self.run(vid) for vid in N]

        for vid, best_path in enumerate(return_value):
            dsv_paths[vid] = best_path

        print("DSV Locations At End ({}):{}".format(next_time_t, self.dsvL))  
        return dsv_paths

    def get_observations(self, state, action, t):
        next_state = state.copy()

        curY, curX = np.where(next_state == dsv_loc_val)

        dirX, dirY = self.map_action_to_direction[action]

        x, y = curX + dirX, curY + dirY

        if not (0 <= y < next_state.shape[0] and 0 <= x < next_state.shape[1]):
            return np.array([-1, -1, -1, -1, -1])

        next_state[curY, curX] = 0 # previous dsv location

        val = next_state[y, x][0]

        cov = val//60 if val >= 0 else -1

        tfm = val/60 if val >= 0 else -1

        n_neighbors = [(dx, dy) for dx in range(-1, 2) for dy in range(-1, 2) if dx != 0 or dy != 0]

        n_nei = sum(1 for dx, dy in n_neighbors if 0 <= y + dy < next_state.shape[0] and 0 <= x + dx < next_state.shape[1] and next_state[y + dy, x + dx] >= 60)  if val >= 0 else -1 

        nei_tfm = [next_state[y + dy, x + dx] for dx, dy in n_neighbors if 0 <= y + dy < next_state.shape[0] and 0 <= x + dx < next_state.shape[1] and next_state[y + dy, x + dx] >= 0]

        sfm = np.mean(nei_tfm)/60  if val >= 0 else -1 

        path_rem = self.path_length - t - 1

        center_x, center_y = x.item(), y.item()
        start_y, end_y =  max(center_y - path_rem, 0), min(center_y + path_rem, next_state.shape[0])
        start_x, end_x =  max(center_x - path_rem, 0), min(center_x + path_rem, next_state.shape[1])
        region = next_state[start_y:end_y+1, start_x:end_x+1]

        rfm = np.mean(region[region >= 0])/60 if val >= 0 else -1

        return np.array([cov, tfm, n_nei, sfm, rfm])
     
    def gen_features_mat(self, state, t):
        state_copy = state.copy()
        y, x = np.where(state_copy == dsv_loc_val)
        features_mat = [self.get_observations(state_copy, action, t) for action in self.map_action_to_direction]
        features_mat = np.array(features_mat)
        min_values = np.min(features_mat, axis=0)
        max_values = np.max(features_mat, axis=0)
        normalized_data = (features_mat - min_values) / (max_values - min_values + 1e-8)

        return normalized_data #features_mat #normalized_data
class FUSE_DQN_Train_wo12:
    def __init__(self, state_data, val_data, train_Samples, tf=20, nDiv=1, action_size=8, episodes=100, batch_size=128, alpha=0.01, gamma=0.9, epsilon=0.8, epsilon_decay=0.999, epsilon_min=0.01, update_freq=5):
        self.name = "fuse_dqn_wo12"
        self.all_data = state_data
        self.state_size = train_Samples
        self.val_data = val_data
        self.action_size = action_size
        self.memory = deque(maxlen=1000000)
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration-exploitation trade-off
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.update_freq = update_freq
        self.batch_size = batch_size
        self.episodes = episodes
        self.path_length = tf//2 # Since we cover 1km (one grid) every 2 min (30 km/h)
        self.tf = tf

        self.map_action_to_direction = {0:(-1, -1), 
                                        1:(0, -1), 
                                        2:(1, -1), 
                                        3:(-1, 0), 
                                        4:(1, 0), 
                                        5:(-1, 1), 
                                        6:(0, 1), 
                                        7:(1, 1)}

        self.map_direction_to_action = {(-1, -1):0, 
                                        (0, -1):1, 
                                        (1, -1):2, 
                                        (-1, 0):3, 
                                        (1, 0):4, 
                                        (-1, 1):5, 
                                        (0, 1):6, 
                                        (1, 1):7}
        sample_data = self.all_data[:, :, 0]
        sample_feature = self.gen_features_mat(sample_data, 0)
        self.input_shape = sample_feature.shape

        self.model = self._build_model()
        self.target_model = self._build_model()
        self.best_reward_model = self._build_model()

        self.checkpoint_path_best = f'models_new/best_{self.name}_r_model.h5'
        self.checkpoint_path_t = f'models_new/{self.name}_t_model.h5'
        self.checkpoint_path_q = f'models_new/{self.name}_q_model.h5'

        self.best_reward_val = 0.0

    def _build_model(self):
        print('\nBuilding {} Q-Network with Input Shape: {}\n'.format(self.name, self.input_shape))
        model = models.Sequential([
                layers.Dense(64, activation='relu', input_shape=self.input_shape),
                layers.Dense(64, activation='relu', kernel_regularizer='l2'),
                layers.Dense(self.action_size, activation='relu'),
                layers.Dense(1, activation='linear')
            ])

        model.compile(loss='huber_loss', optimizer=optimizers.Adam(learning_rate=self.alpha), metrics=['mae'])
        return model

    def remember(self, state, action, reward, next_state, t):
        self.memory.append((state.copy(), action, reward, next_state.copy(), t))

    def get_observations(self, state, action, t):
        next_state = state.copy()

        curY, curX = np.where(next_state == dsv_loc_val)

        dirX, dirY = self.map_action_to_direction[action]

        x, y = curX + dirX, curY + dirY
        if not ((0 <= y < next_state.shape[0]) and (0 <= x < next_state.shape[1])).any():
            return np.array([-1, -1, -1, -1, -1])

        next_state[curY, curX] = 0 # previous dsv location

        val = next_state[y, x][0]

        cov = val//60 if val >= 0 else -1

        tfm = val/60 if val >= 0 else -1

        n_neighbors = [(dx, dy) for dx in range(-1, 2) for dy in range(-1, 2) if dx != 0 or dy != 0]

        n_nei = sum(1 for dx, dy in n_neighbors if 0 <= y + dy < next_state.shape[0] and 0 <= x + dx < next_state.shape[1] and next_state[y + dy, x + dx] >= 60)  if val >= 0 else -1 

        nei_tfm = [next_state[y + dy, x + dx] for dx, dy in n_neighbors if 0 <= y + dy < next_state.shape[0] and 0 <= x + dx < next_state.shape[1] and next_state[y + dy, x + dx] >= 0]

        sfm = np.mean(nei_tfm)/60  if val >= 0 else -1 

        path_rem = self.path_length - t - 1

        center_x, center_y = x.item(), y.item()
        start_y, end_y =  max(center_y - path_rem, 0), min(center_y + path_rem, next_state.shape[0])
        start_x, end_x =  max(center_x - path_rem, 0), min(center_x + path_rem, next_state.shape[1])
        region = next_state[start_y:end_y+1, start_x:end_x+1]

        rfm = np.mean(region[region >= 0])/60 if val >= 0 else -1

        return np.array([n_nei, sfm, rfm])
     
    def gen_features_mat(self, state, t):
        state_copy = state.copy()
        y, x = np.where(state_copy == dsv_loc_val)
        features_mat = [self.get_observations(state_copy, action, t) for action in self.map_action_to_direction]
        features_mat = np.array(features_mat)
        min_values = np.min(features_mat, axis=0)
        max_values = np.max(features_mat, axis=0)
        normalized_data = (features_mat - min_values) / (max_values - min_values + 1e-8)

        return normalized_data #features_mat #normalized_data

    def valid_actions(self, state):
        state_copy = state.copy()
        y, x = np.where(state_copy == dsv_loc_val)
        n_neighbors = [(dx, dy) for dx in range(-1, 2) for dy in range(-1, 2) if dx != 0 or dy != 0]

        valid_dirs = [(dx, dy) for dx, dy in n_neighbors if 0 <= y + dy < state.shape[0] and 0 <= x + dx < state.shape[1] and state[y + dy, x + dx] != invalid_loc_val]
        valid_acts = [self.map_direction_to_action[(dx, dy)] for dx, dy in valid_dirs]
        return sorted(valid_acts)

    def choose_action(self, state_features, state, t):

        valid_acts = self.valid_actions(state)
        if np.random.rand() <= self.epsilon:
            return random.choice(valid_acts)  # Explore
        else:
            q_values = self.model.predict(np.array([state_features.copy()]))
            state_copy = state.copy()

            return valid_acts[np.argmax(q_values[0][valid_acts])]  # Exploit

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        states, targets = [], []

        for state, action, reward, next_state, t in minibatch:
            target = reward

            state_features = self.gen_features_mat(state, t)
            if next_state is not None:

                next_state_features = self.gen_features_mat(next_state, t+1)

                next_state_preds = self.target_model.predict(np.array([next_state_features]))

                target = reward + self.gamma * np.amax(next_state_preds[0])

            target_f = self.model.predict(np.array([state_features]))

            target_f[0][action] = target

            states.append(state_features)
            targets.append(target_f[0])

        states = np.array(states)
        targets = np.array(targets)
        test_loss = self.model.evaluate(states, targets, verbose=2)

        self.model.fit(states, targets, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    def step(self, state, action, t, return_state=True):
        next_state = state.copy()

        curY, curX = np.where(next_state == dsv_loc_val)

        dirX, dirY = self.map_action_to_direction[action]

        nextX, nextY = curX + dirX, curY + dirY

        next_state[curY, curX] = 0 # previous dsv location

        reward = self.calculate_reward(next_state.copy(), nextY, nextX, t)

        next_state[next_state >= 0] += 1

        next_state[curY, curX] = 0 # previous dsv location

        next_state[nextY, nextX] = dsv_loc_val # new dsv location

        if not return_state:
            return reward 
        return reward, next_state

    def calculate_reward(self, state, y, x, t):
        val = state[y, x]
        if val < -1000:
            return 0.0 - 5.0

        temporal_fairness_reward = val/60 if val >= 0 else -1

        n_neighbors = [(dx, dy) for dx in range(-1, 2) for dy in range(-1, 2) if dx != 0 or dy != 0]
        nei_tfm = [state[y + dy, x + dx]/60 for dx, dy in n_neighbors if 0 <= y + dy < state.shape[0] and 0 <= x + dx < state.shape[1] and state[y + dy, x + dx] >= 0]
        local_fairness_reward = np.mean(nei_tfm) if val >= 0 else -1
        center_y, center_x = y.item(), x.item()
        path_rem = self.path_length - t -1
        start_y, end_y =  max(center_y - path_rem, 0), min(center_y + path_rem, state.shape[0])
        start_x, end_x =  max(center_x - path_rem, 0), min(center_x + path_rem, state.shape[1])
        region = state[start_y:end_y+1, start_x:end_x+1]

        region_fairness_reward = np.mean(region[region >= 0])/60 if val >= 0 else -1

        reward = temporal_fairness_reward + local_fairness_reward + region_fairness_reward
        return reward

    def train(self):
        print(f'\nStarting {self.name} Training...\n')
        best_reward = 0.0

        for episode in range(self.episodes):
            total_reward = 0.0
            train_Samples = 10000
            sampled_indices = np.random.choice(self.all_data.shape[2], train_Samples, replace=False)

            self.train_data = self.all_data[:, :, sampled_indices]
            print('\EPISODE: {}  Randomly Sampled Training State Data Loaded ({} samples) of Shape: {}'.format(episode, self.train_data.shape[2], self.train_data[:, :, 0].shape))

            states = self.train_data.transpose((2, 0, 1))
            print('\nstates.shape: ', states.shape)

            for t in range(self.path_length):

                state_features_mat = np.array([self.gen_features_mat(state, t) for state in states])

                print('state_features_mat.shape: ', state_features_mat.shape)

                actions = np.array([self.choose_action(state_features, states[i], t) for i, state_features in enumerate(state_features_mat)])

                rewards, next_states = zip(*[self.step(states[i], action, t) for i, action in enumerate(actions)])
                for i in range(self.state_size):
                    self.remember(states[i], actions[i], rewards[i], next_states[i], t)

                total_reward += np.sum(rewards)

                states = [next_states[i].copy() for i in range(self.train_data.shape[2])]

                print('\nTIMESTEP: ', t, '  reward: ', np.sum(rewards))

                self.replay(batch_size=self.batch_size)

            print('\nCompleted {} samples; Reward = {}'.format(self.state_size, total_reward))
            print(f"\nEPISODE {episode + 1}, Total Training Reward: {total_reward}, Best Reward: {best_reward}")

            total_reward_val = self.validation()

            if episode == 0:
                self.best_reward_val = total_reward_val

            if self.best_reward_val <= total_reward_val:
                self.best_reward_val = total_reward_val
                self.best_reward_model.set_weights(self.model.get_weights())
                self.best_reward_model.save(self.checkpoint_path_best, save_format='h5')
                print(f"\nBest Model Saved to {self.checkpoint_path_best}")
                
    def choose_action_val(self, state_features, state):
        valid_acts = self.valid_actions(state)

        q_values = self.model.predict(np.array([state_features.copy()]))
        return valid_acts[np.argmax(q_values[0][valid_acts])], np.max(q_values[0][valid_acts])

    def validation(self):
        states = self.val_data.transpose((2, 0, 1))
        print('Validation states.shape: ', states.shape)

        total_reward = 0.0
        for i, state in enumerate(states):

            for t in range(self.path_length):
                state_features = self.gen_features_mat(state, t)

                action, q_val = self.choose_action_val(state_features, state)
                reward, next_state = self.step(state, action, t)

                loss = tf.keras.losses.Huber()(reward, q_val)
                state = next_state.copy() # Updating State for next iteration
                total_reward += reward
        print('\nVALIDATION Completed {} samples; Total Validation Reward = {};  Best Validation Reward= {}'.format(states.shape[0], total_reward, self.best_reward_val))

        return total_reward

class FUSE_DQN_Test_wo12:

    def __init__(self, df, grid_size, n, tf, beta=60, seed=42, km_time=2) -> None:
        random.seed(seed)
        self.name = "fuse_wo12"
        self.df = df
        self.n = n
        self.timeframe = tf
        self.grid_size = grid_size
        self.km_time = int(km_time*grid_size/1000) # km_time to complete 1 km (move to next grid)
        self.path_length = tf//self.km_time # path length per interval

        n_rows, n_cols, nPosns, n_rows_valid, n_cols_valid, nPosns_valid = gridSize_info_dict[self.grid_size]
        self.map_area = np.array([[0 for i in range(n_cols_valid)] for j in range(n_rows_valid)])
        self.area_nrows, self.area_ncols = size_dict[self.n]
        self.pos_neis, self.tfm_map, self.valid_div_pos, self.pos_to_div, self.valid_coords, self.div_coords = generate_neighbors_list_wBorders(self.n, grid_size, self.df, self.name, self.timeframe, self.km_time)
        self.tfm_maps = []
        self.dsvL = [random.sample(list(self.valid_div_pos[i]), 1)[0] for i in range(n)]
        self.dsv_paths = {i:deque([self.dsvL[i]]) for i in range(self.n)}

        self.map_action_to_direction = {0:(-1, -1), 
                                        1:(0, -1), 
                                        2:(1, -1), 
                                        3:(-1, 0), 
                                        4:(1, 0), 
                                        5:(-1, 1), 
                                        6:(0, 1), 
                                        7:(1, 1)}

        self.map_direction_to_action = {(-1, -1):0, 
                                        (0, -1):1, 
                                        (1, -1):2, 
                                        (-1, 0):3, 
                                        (1, 0):4, 
                                        (-1, 1):5, 
                                        (0, 1):6, 
                                        (1, 1):7}

        self.checkpoint_path = f'models_new/best_{self.name}_r_model.h5' #'models_new/best_model_fcn_new_1.h5'
        self.model = models.load_model(self.checkpoint_path) #, custom_objects=custom_objects)    

        print("Q-Network Model Loaded:")
        print(self.model.summary())

        print("\n{} Model Object Created".format(self.name))
        print("\nValid Locations Per Region", [len(self.valid_div_pos[i]) for i in range(self.n)])
        print('\nArea Border of Division (start_row, end_row, start_col, end_col):', self.div_coords)
        print("\nStudy Area Gird Shape: ", self.tfm_map.shape)
        print("DSV Starting Locations: ",self.dsvL)
        print("DSV Starting Grid Positions: ", [(pos//n_cols, pos%n_cols) for pos in self.dsvL])
        print('Path Length: ', self.path_length)
                
    def valid_actions(self, state):
        state_copy = state.copy()
        y, x = np.where(state_copy == dsv_loc_val)
        n_neighbors = [(dx, dy) for dx in range(-1, 2) for dy in range(-1, 2) if dx != 0 or dy != 0]

        valid_dirs = [(dx, dy) for dx, dy in n_neighbors if 0 <= y + dy < state.shape[0] and 0 <= x + dx < state.shape[1] and state[y + dy, x + dx] != invalid_loc_val]
        valid_acts = [self.map_direction_to_action[(dx, dy)] for dx, dy in valid_dirs]
        return sorted(valid_acts)

    def choose_action(self, state_features, state):
        valid_acts = self.valid_actions(state)

        q_values = self.model.predict(np.array([state_features.copy()]))
        return valid_acts[np.argmax(q_values[0][valid_acts])]

    def take_action(self, state, action, globalY, globalX):
        next_state = state.copy()

        curY, curX = np.where(next_state == dsv_loc_val)

        dirX, dirY = self.map_action_to_direction[action]

        nextX, nextY = curX + dirX, curY + dirY

        next_globalX, next_globalY = globalX + dirX, globalY + dirY # to update the global study area

        next_state[curY, curX] = 0 # previous dsv location

        next_state[next_state >= 0] += 1

        next_state[curY, curX] = 0 # previous dsv location

        next_state[nextY, nextX] = dsv_loc_val # new dsv location

        return next_state, nextY, nextX, next_globalY, next_globalX

    def run(self, vid):
        ''' 
        INPUTS:

        OUTPUTS:
        '''    

        cur_pos = self.dsvL[vid]
        dsv_path = [cur_pos]
        global_dsvY, global_dsvX = pos_to_coord(self.grid_size, cur_pos)
        div_id, localY, localX = self.pos_to_div[(global_dsvY, global_dsvX)]
        state = gen_state(self.tfm_maps[div_id], localY, localX, self.timeframe, self.km_time)

        for t in range(self.path_length):

            state_features = self.gen_features_mat(state, t)

            action = self.choose_action(state_features, state)
            next_state, local_dsvY, local_dsvX, global_dsvY, global_dsvX = self.take_action(state, action, global_dsvY, global_dsvX)
            state = next_state.copy() # Updating State for next iteration
            next_pos = coord_to_pos(self.grid_size, global_dsvY, global_dsvX)
            dsv_path.append(next_pos)

        self.tfm_maps[div_id], self.tfm_map = update_map(self.tfm_map, self.tfm_maps[div_id], state, localY, localX, self.div_coords[div_id], self.timeframe, self.km_time )
        self.dsvL[vid] = next_pos
        return dsv_path

    def parallel_run(self, curTime):
        ''' 
        INPUTS:

        OUTPUTS:

        '''
        N = [ele for ele in range(self.n)]
        nthreads_obj = self.n
        dsv_paths = {i:[self.dsvL[i]] for i in range(self.n)}
        time_t_prev = curTime
        times_print = [str(curTime)]
        paths_print = []

        time_t_prev = curTime
        next_time_t = time_t_prev + timedelta(minutes=self.timeframe)

        for t in range(self.path_length): # range(self.path_int+1)
            interval = timedelta(minutes=self.km_time-1)
            time_t = time_t_prev + interval

            data_interval = fetch_rows(self.df, time_t_prev, time_t)

            locs_interval_ts = get_FHV_cov_coord(self.grid_size, data_interval, self.valid_coords)

            self.tfm_map = update_tfm_sfm_tEnd(self.tfm_map, self.valid_coords, self.path_length, testing=True)

            self.tfm_map = update_tfm_sfm_cov(self.tfm_map, locs_interval_ts, self.path_length, testing=True)

            times_print.append(str(time_t))
            time_t_prev = time_t_prev + timedelta(minutes=self.km_time)
        self.tfm_maps = get_mat2D_nDivs_fromFullMap(self.grid_size, self.df, self.tfm_map, self.n, self.path_length, self.valid_coords)

        return_value = [self.run(vid) for vid in N]

        for vid, best_path in enumerate(return_value):
            dsv_paths[vid] = best_path

        print("DSV Locations At End ({}):{}".format(next_time_t, self.dsvL))  
        return dsv_paths

    def get_observations(self, state, action, t):
        next_state = state.copy()

        curY, curX = np.where(next_state == dsv_loc_val)

        dirX, dirY = self.map_action_to_direction[action]

        x, y = curX + dirX, curY + dirY

        if not (0 <= y < next_state.shape[0] and 0 <= x < next_state.shape[1]):
            return np.array([-1, -1, -1, -1, -1])

        next_state[curY, curX] = 0 # previous dsv location

        val = next_state[y, x][0]

        cov = val//60 if val >= 0 else -1

        tfm = val/60 if val >= 0 else -1

        n_neighbors = [(dx, dy) for dx in range(-1, 2) for dy in range(-1, 2) if dx != 0 or dy != 0]

        n_nei = sum(1 for dx, dy in n_neighbors if 0 <= y + dy < next_state.shape[0] and 0 <= x + dx < next_state.shape[1] and next_state[y + dy, x + dx] >= 60)  if val >= 0 else -1 

        nei_tfm = [next_state[y + dy, x + dx] for dx, dy in n_neighbors if 0 <= y + dy < next_state.shape[0] and 0 <= x + dx < next_state.shape[1] and next_state[y + dy, x + dx] >= 0]

        sfm = np.mean(nei_tfm)/60  if val >= 0 else -1 

        path_rem = self.path_length - t - 1

        center_x, center_y = x.item(), y.item()
        start_y, end_y =  max(center_y - path_rem, 0), min(center_y + path_rem, next_state.shape[0])
        start_x, end_x =  max(center_x - path_rem, 0), min(center_x + path_rem, next_state.shape[1])
        region = next_state[start_y:end_y+1, start_x:end_x+1]

        rfm = np.mean(region[region >= 0])/60 if val >= 0 else -1

        return np.array([n_nei, sfm, rfm])
     
    def gen_features_mat(self, state, t):
        state_copy = state.copy()
        y, x = np.where(state_copy == dsv_loc_val)
        features_mat = [self.get_observations(state_copy, action, t) for action in self.map_action_to_direction]
        features_mat = np.array(features_mat)
        min_values = np.min(features_mat, axis=0)
        max_values = np.max(features_mat, axis=0)
        normalized_data = (features_mat - min_values) / (max_values - min_values + 1e-8)

        return normalized_data #features_mat #normalized_data
class FUSE_DQN_Train_wo34:
    def __init__(self, state_data, val_data, train_Samples, tf=20, nDiv=1, action_size=8, episodes=100, batch_size=128, alpha=0.01, gamma=0.9, epsilon=0.8, epsilon_decay=0.999, epsilon_min=0.01, update_freq=5):
        self.name = "fuse_dqn_wo34"
        self.all_data = state_data
        self.state_size = train_Samples
        self.val_data = val_data
        self.action_size = action_size
        self.memory = deque(maxlen=1000000)
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration-exploitation trade-off
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.update_freq = update_freq
        self.batch_size = batch_size
        self.episodes = episodes
        self.path_length = tf//2 # Since we cover 1km (one grid) every 2 min (30 km/h)
        self.tf = tf

        self.map_action_to_direction = {0:(-1, -1), 
                                        1:(0, -1), 
                                        2:(1, -1), 
                                        3:(-1, 0), 
                                        4:(1, 0), 
                                        5:(-1, 1), 
                                        6:(0, 1), 
                                        7:(1, 1)}

        self.map_direction_to_action = {(-1, -1):0, 
                                        (0, -1):1, 
                                        (1, -1):2, 
                                        (-1, 0):3, 
                                        (1, 0):4, 
                                        (-1, 1):5, 
                                        (0, 1):6, 
                                        (1, 1):7}
        sample_data = self.all_data[:, :, 0]
        sample_feature = self.gen_features_mat(sample_data, 0)
        self.input_shape = sample_feature.shape

        self.model = self._build_model()
        self.target_model = self._build_model()
        self.best_reward_model = self._build_model()

        self.checkpoint_path_best = f'models_new/best_{self.name}_r_model.h5'
        self.checkpoint_path_t = f'models_new/{self.name}_t_model.h5'
        self.checkpoint_path_q = f'models_new/{self.name}_q_model.h5'

        self.best_reward_val = 0.0

    def _build_model(self):
        print('\nBuilding {} Q-Network with Input Shape: {}\n'.format(self.name, self.input_shape))
        model = models.Sequential([
                layers.Dense(64, activation='relu', input_shape=self.input_shape),
                layers.Dense(64, activation='relu', kernel_regularizer='l2'),
                layers.Dense(self.action_size, activation='relu'),
                layers.Dense(1, activation='linear')
            ])

        model.compile(loss='huber_loss', optimizer=optimizers.Adam(learning_rate=self.alpha), metrics=['mae'])
        return model

    def remember(self, state, action, reward, next_state, t):
        self.memory.append((state.copy(), action, reward, next_state.copy(), t))

    def get_observations(self, state, action, t):
        next_state = state.copy()

        curY, curX = np.where(next_state == dsv_loc_val)

        dirX, dirY = self.map_action_to_direction[action]

        x, y = curX + dirX, curY + dirY
        if not ((0 <= y < next_state.shape[0]) and (0 <= x < next_state.shape[1])).any():
            return np.array([-1, -1, -1, -1, -1])

        next_state[curY, curX] = 0 # previous dsv location

        val = next_state[y, x][0]

        cov = val//60 if val >= 0 else -1

        tfm = val/60 if val >= 0 else -1

        n_neighbors = [(dx, dy) for dx in range(-1, 2) for dy in range(-1, 2) if dx != 0 or dy != 0]

        n_nei = sum(1 for dx, dy in n_neighbors if 0 <= y + dy < next_state.shape[0] and 0 <= x + dx < next_state.shape[1] and next_state[y + dy, x + dx] >= 60)  if val >= 0 else -1 

        nei_tfm = [next_state[y + dy, x + dx] for dx, dy in n_neighbors if 0 <= y + dy < next_state.shape[0] and 0 <= x + dx < next_state.shape[1] and next_state[y + dy, x + dx] >= 0]

        sfm = np.mean(nei_tfm)/60  if val >= 0 else -1 

        path_rem = self.path_length - t - 1

        center_x, center_y = x.item(), y.item()
        start_y, end_y =  max(center_y - path_rem, 0), min(center_y + path_rem, next_state.shape[0])
        start_x, end_x =  max(center_x - path_rem, 0), min(center_x + path_rem, next_state.shape[1])
        region = next_state[start_y:end_y+1, start_x:end_x+1]

        rfm = np.mean(region[region >= 0])/60 if val >= 0 else -1

        return np.array([cov, tfm, rfm])
     
    def gen_features_mat(self, state, t):
        state_copy = state.copy()
        y, x = np.where(state_copy == dsv_loc_val)
        features_mat = [self.get_observations(state_copy, action, t) for action in self.map_action_to_direction]
        features_mat = np.array(features_mat)
        min_values = np.min(features_mat, axis=0)
        max_values = np.max(features_mat, axis=0)
        normalized_data = (features_mat - min_values) / (max_values - min_values + 1e-8)

        return normalized_data #features_mat #normalized_data

    def valid_actions(self, state):
        state_copy = state.copy()
        y, x = np.where(state_copy == dsv_loc_val)
        n_neighbors = [(dx, dy) for dx in range(-1, 2) for dy in range(-1, 2) if dx != 0 or dy != 0]

        valid_dirs = [(dx, dy) for dx, dy in n_neighbors if 0 <= y + dy < state.shape[0] and 0 <= x + dx < state.shape[1] and state[y + dy, x + dx] != invalid_loc_val]
        valid_acts = [self.map_direction_to_action[(dx, dy)] for dx, dy in valid_dirs]
        return sorted(valid_acts)

    def choose_action(self, state_features, state, t):

        valid_acts = self.valid_actions(state)
        if np.random.rand() <= self.epsilon:
            return random.choice(valid_acts)  # Explore
        else:
            q_values = self.model.predict(np.array([state_features.copy()]))
            state_copy = state.copy()

            return valid_acts[np.argmax(q_values[0][valid_acts])]  # Exploit

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        states, targets = [], []

        for state, action, reward, next_state, t in minibatch:
            target = reward

            state_features = self.gen_features_mat(state, t)
            if next_state is not None:

                next_state_features = self.gen_features_mat(next_state, t+1)

                next_state_preds = self.target_model.predict(np.array([next_state_features]))

                target = reward + self.gamma * np.amax(next_state_preds[0])

            target_f = self.model.predict(np.array([state_features]))

            target_f[0][action] = target

            states.append(state_features)
            targets.append(target_f[0])

        states = np.array(states)
        targets = np.array(targets)
        test_loss = self.model.evaluate(states, targets, verbose=2)

        self.model.fit(states, targets, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    def step(self, state, action, t, return_state=True):
        next_state = state.copy()

        curY, curX = np.where(next_state == dsv_loc_val)

        dirX, dirY = self.map_action_to_direction[action]

        nextX, nextY = curX + dirX, curY + dirY

        next_state[curY, curX] = 0 # previous dsv location

        reward = self.calculate_reward(next_state.copy(), nextY, nextX, t)

        next_state[next_state >= 0] += 1

        next_state[curY, curX] = 0 # previous dsv location

        next_state[nextY, nextX] = dsv_loc_val # new dsv location

        if not return_state:
            return reward 
        return reward, next_state

    def calculate_reward(self, state, y, x, t):
        val = state[y, x]
        if val < -1000:
            return 0.0 - 5.0

        temporal_fairness_reward = val/60 if val >= 0 else -1

        n_neighbors = [(dx, dy) for dx in range(-1, 2) for dy in range(-1, 2) if dx != 0 or dy != 0]
        nei_tfm = [state[y + dy, x + dx]/60 for dx, dy in n_neighbors if 0 <= y + dy < state.shape[0] and 0 <= x + dx < state.shape[1] and state[y + dy, x + dx] >= 0]
        local_fairness_reward = np.mean(nei_tfm) if val >= 0 else -1
        center_y, center_x = y.item(), x.item()
        path_rem = self.path_length - t -1
        start_y, end_y =  max(center_y - path_rem, 0), min(center_y + path_rem, state.shape[0])
        start_x, end_x =  max(center_x - path_rem, 0), min(center_x + path_rem, state.shape[1])
        region = state[start_y:end_y+1, start_x:end_x+1]

        region_fairness_reward = np.mean(region[region >= 0])/60 if val >= 0 else -1

        reward = temporal_fairness_reward + local_fairness_reward + region_fairness_reward
        return reward

    def train(self):
        print(f'\nStarting {self.name} Training...\n')
        best_reward = 0.0

        for episode in range(self.episodes):
            total_reward = 0.0
            train_Samples = 10000
            sampled_indices = np.random.choice(self.all_data.shape[2], train_Samples, replace=False)

            self.train_data = self.all_data[:, :, sampled_indices]
            print('\EPISODE: {}  Randomly Sampled Training State Data Loaded ({} samples) of Shape: {}'.format(episode, self.train_data.shape[2], self.train_data[:, :, 0].shape))

            states = self.train_data.transpose((2, 0, 1))
            print('\nstates.shape: ', states.shape)

            for t in range(self.path_length):

                state_features_mat = np.array([self.gen_features_mat(state, t) for state in states])

                print('state_features_mat.shape: ', state_features_mat.shape)

                actions = np.array([self.choose_action(state_features, states[i], t) for i, state_features in enumerate(state_features_mat)])

                rewards, next_states = zip(*[self.step(states[i], action, t) for i, action in enumerate(actions)])
                for i in range(self.state_size):
                    self.remember(states[i], actions[i], rewards[i], next_states[i], t)

                total_reward += np.sum(rewards)

                states = [next_states[i].copy() for i in range(self.train_data.shape[2])]

                print('\nTIMESTEP: ', t, '  reward: ', np.sum(rewards))

                self.replay(batch_size=self.batch_size)

            print('\nCompleted {} samples; Reward = {}'.format(self.state_size, total_reward))
            print(f"\nEPISODE {episode + 1}, Total Training Reward: {total_reward}, Best Reward: {best_reward}")

            total_reward_val = self.validation()

            if episode == 0:
                self.best_reward_val = total_reward_val

            if self.best_reward_val <= total_reward_val:
                self.best_reward_val = total_reward_val
                self.best_reward_model.set_weights(self.model.get_weights())
                self.best_reward_model.save(self.checkpoint_path_best, save_format='h5')
                print(f"\nBest Model Saved to {self.checkpoint_path_best}")
                
    def choose_action_val(self, state_features, state):
        valid_acts = self.valid_actions(state)

        q_values = self.model.predict(np.array([state_features.copy()]))
        return valid_acts[np.argmax(q_values[0][valid_acts])], np.max(q_values[0][valid_acts])

    def validation(self):
        states = self.val_data.transpose((2, 0, 1))
        print('Validation states.shape: ', states.shape)

        total_reward = 0.0
        for i, state in enumerate(states):

            for t in range(self.path_length):
                state_features = self.gen_features_mat(state, t)

                action, q_val = self.choose_action_val(state_features, state)
                reward, next_state = self.step(state, action, t)

                loss = tf.keras.losses.Huber()(reward, q_val)
                state = next_state.copy() # Updating State for next iteration
                total_reward += reward
        print('\nVALIDATION Completed {} samples; Total Validation Reward = {};  Best Validation Reward= {}'.format(states.shape[0], total_reward, self.best_reward_val))

        return total_reward

class FUSE_DQN_Test_wo34:

    def __init__(self, df, grid_size, n, tf, beta=60, seed=42, km_time=2) -> None:
        random.seed(seed)
        self.name = "fuse_wo34"
        self.df = df
        self.n = n
        self.timeframe = tf
        self.grid_size = grid_size
        self.km_time = int(km_time*grid_size/1000) # km_time to complete 1 km (move to next grid)
        self.path_length = tf//self.km_time # path length per interval

        n_rows, n_cols, nPosns, n_rows_valid, n_cols_valid, nPosns_valid = gridSize_info_dict[self.grid_size]
        self.map_area = np.array([[0 for i in range(n_cols_valid)] for j in range(n_rows_valid)])
        self.area_nrows, self.area_ncols = size_dict[self.n]
        self.pos_neis, self.tfm_map, self.valid_div_pos, self.pos_to_div, self.valid_coords, self.div_coords = generate_neighbors_list_wBorders(self.n, grid_size, self.df, self.name, self.timeframe, self.km_time)
        self.tfm_maps = []
        self.dsvL = [random.sample(list(self.valid_div_pos[i]), 1)[0] for i in range(n)]
        self.dsv_paths = {i:deque([self.dsvL[i]]) for i in range(self.n)}

        self.map_action_to_direction = {0:(-1, -1), 
                                        1:(0, -1), 
                                        2:(1, -1), 
                                        3:(-1, 0), 
                                        4:(1, 0), 
                                        5:(-1, 1), 
                                        6:(0, 1), 
                                        7:(1, 1)}

        self.map_direction_to_action = {(-1, -1):0, 
                                        (0, -1):1, 
                                        (1, -1):2, 
                                        (-1, 0):3, 
                                        (1, 0):4, 
                                        (-1, 1):5, 
                                        (0, 1):6, 
                                        (1, 1):7}

        self.checkpoint_path = f'models_new/best_{self.name}_r_model.h5' #'models_new/best_model_fcn_new_1.h5'
        self.model = models.load_model(self.checkpoint_path) #, custom_objects=custom_objects)    

        print("Q-Network Model Loaded:")
        print(self.model.summary())

        print("\n{} Model Object Created".format(self.name))
        print("\nValid Locations Per Region", [len(self.valid_div_pos[i]) for i in range(self.n)])
        print('\nArea Border of Division (start_row, end_row, start_col, end_col):', self.div_coords)
        print("\nStudy Area Gird Shape: ", self.tfm_map.shape)
        print("DSV Starting Locations: ",self.dsvL)
        print("DSV Starting Grid Positions: ", [(pos//n_cols, pos%n_cols) for pos in self.dsvL])
        print('Path Length: ', self.path_length)
                
    def valid_actions(self, state):
        state_copy = state.copy()
        y, x = np.where(state_copy == dsv_loc_val)
        n_neighbors = [(dx, dy) for dx in range(-1, 2) for dy in range(-1, 2) if dx != 0 or dy != 0]

        valid_dirs = [(dx, dy) for dx, dy in n_neighbors if 0 <= y + dy < state.shape[0] and 0 <= x + dx < state.shape[1] and state[y + dy, x + dx] != invalid_loc_val]
        valid_acts = [self.map_direction_to_action[(dx, dy)] for dx, dy in valid_dirs]
        return sorted(valid_acts)

    def choose_action(self, state_features, state):
        valid_acts = self.valid_actions(state)

        q_values = self.model.predict(np.array([state_features.copy()]))
        return valid_acts[np.argmax(q_values[0][valid_acts])]

    def take_action(self, state, action, globalY, globalX):
        next_state = state.copy()

        curY, curX = np.where(next_state == dsv_loc_val)

        dirX, dirY = self.map_action_to_direction[action]

        nextX, nextY = curX + dirX, curY + dirY

        next_globalX, next_globalY = globalX + dirX, globalY + dirY # to update the global study area

        next_state[curY, curX] = 0 # previous dsv location

        next_state[next_state >= 0] += 1

        next_state[curY, curX] = 0 # previous dsv location

        next_state[nextY, nextX] = dsv_loc_val # new dsv location

        return next_state, nextY, nextX, next_globalY, next_globalX

    def run(self, vid):
        ''' 
        INPUTS:

        OUTPUTS:
        '''    

        cur_pos = self.dsvL[vid]
        dsv_path = [cur_pos]
        global_dsvY, global_dsvX = pos_to_coord(self.grid_size, cur_pos)
        div_id, localY, localX = self.pos_to_div[(global_dsvY, global_dsvX)]
        state = gen_state(self.tfm_maps[div_id], localY, localX, self.timeframe, self.km_time)

        for t in range(self.path_length):

            state_features = self.gen_features_mat(state, t)
            action = self.choose_action(state_features, state)
            next_state, local_dsvY, local_dsvX, global_dsvY, global_dsvX = self.take_action(state, action, global_dsvY, global_dsvX)
            state = next_state.copy() # Updating State for next iteration
            next_pos = coord_to_pos(self.grid_size, global_dsvY, global_dsvX)
            dsv_path.append(next_pos)

        self.tfm_maps[div_id], self.tfm_map = update_map(self.tfm_map, self.tfm_maps[div_id], state, localY, localX, self.div_coords[div_id], self.timeframe, self.km_time )
        self.dsvL[vid] = next_pos
        return dsv_path

    def parallel_run(self, curTime):
        ''' 
        INPUTS:

        OUTPUTS:

        '''
        N = [ele for ele in range(self.n)]
        nthreads_obj = self.n
        dsv_paths = {i:[self.dsvL[i]] for i in range(self.n)}
        time_t_prev = curTime
        times_print = [str(curTime)]
        paths_print = []

        time_t_prev = curTime
        next_time_t = time_t_prev + timedelta(minutes=self.timeframe)

        for t in range(self.path_length): # range(self.path_int+1)
            interval = timedelta(minutes=self.km_time-1)
            time_t = time_t_prev + interval

            data_interval = fetch_rows(self.df, time_t_prev, time_t)

            locs_interval_ts = get_FHV_cov_coord(self.grid_size, data_interval, self.valid_coords)

            self.tfm_map = update_tfm_sfm_tEnd(self.tfm_map, self.valid_coords, self.path_length, testing=True)

            self.tfm_map = update_tfm_sfm_cov(self.tfm_map, locs_interval_ts, self.path_length, testing=True)

            times_print.append(str(time_t))
            time_t_prev = time_t_prev + timedelta(minutes=self.km_time)
        self.tfm_maps = get_mat2D_nDivs_fromFullMap(self.grid_size, self.df, self.tfm_map, self.n, self.path_length, self.valid_coords)

        return_value = [self.run(vid) for vid in N]

        for vid, best_path in enumerate(return_value):
            dsv_paths[vid] = best_path

        print("DSV Locations At End ({}):{}".format(next_time_t, self.dsvL))  
        return dsv_paths

    def get_observations(self, state, action, t):
        next_state = state.copy()

        curY, curX = np.where(next_state == dsv_loc_val)

        dirX, dirY = self.map_action_to_direction[action]

        x, y = curX + dirX, curY + dirY

        if not (0 <= y < next_state.shape[0] and 0 <= x < next_state.shape[1]):
            return np.array([-1, -1, -1, -1, -1])

        next_state[curY, curX] = 0 # previous dsv location

        val = next_state[y, x][0]

        cov = val//60 if val >= 0 else -1

        tfm = val/60 if val >= 0 else -1

        n_neighbors = [(dx, dy) for dx in range(-1, 2) for dy in range(-1, 2) if dx != 0 or dy != 0]

        n_nei = sum(1 for dx, dy in n_neighbors if 0 <= y + dy < next_state.shape[0] and 0 <= x + dx < next_state.shape[1] and next_state[y + dy, x + dx] >= 60)  if val >= 0 else -1 

        nei_tfm = [next_state[y + dy, x + dx] for dx, dy in n_neighbors if 0 <= y + dy < next_state.shape[0] and 0 <= x + dx < next_state.shape[1] and next_state[y + dy, x + dx] >= 0]

        sfm = np.mean(nei_tfm)/60  if val >= 0 else -1 

        path_rem = self.path_length - t - 1

        center_x, center_y = x.item(), y.item()
        start_y, end_y =  max(center_y - path_rem, 0), min(center_y + path_rem, next_state.shape[0])
        start_x, end_x =  max(center_x - path_rem, 0), min(center_x + path_rem, next_state.shape[1])
        region = next_state[start_y:end_y+1, start_x:end_x+1]

        rfm = np.mean(region[region >= 0])/60 if val >= 0 else -1

        return np.array([cov, tfm, rfm])
     
    def gen_features_mat(self, state, t):
        state_copy = state.copy()
        y, x = np.where(state_copy == dsv_loc_val)
        features_mat = [self.get_observations(state_copy, action, t) for action in self.map_action_to_direction]
        features_mat = np.array(features_mat)
        min_values = np.min(features_mat, axis=0)
        max_values = np.max(features_mat, axis=0)
        normalized_data = (features_mat - min_values) / (max_values - min_values + 1e-8)

        return normalized_data #features_mat #normalized_data
class FUSE_DQN_Train_wo5:
    def __init__(self, state_data, val_data, train_Samples, tf=20, nDiv=1, action_size=8, episodes=100, batch_size=128, alpha=0.01, gamma=0.9, epsilon=0.8, epsilon_decay=0.999, epsilon_min=0.01, update_freq=5):
        self.name = "fuse_dqn_wo5"
        self.all_data = state_data
        self.state_size = train_Samples
        self.val_data = val_data
        self.action_size = action_size
        self.memory = deque(maxlen=1000000)
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration-exploitation trade-off
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.update_freq = update_freq
        self.batch_size = batch_size
        self.episodes = episodes
        self.path_length = tf//2 # Since we cover 1km (one grid) every 2 min (30 km/h)
        self.tf = tf

        self.map_action_to_direction = {0:(-1, -1), 
                                        1:(0, -1), 
                                        2:(1, -1), 
                                        3:(-1, 0), 
                                        4:(1, 0), 
                                        5:(-1, 1), 
                                        6:(0, 1), 
                                        7:(1, 1)}

        self.map_direction_to_action = {(-1, -1):0, 
                                        (0, -1):1, 
                                        (1, -1):2, 
                                        (-1, 0):3, 
                                        (1, 0):4, 
                                        (-1, 1):5, 
                                        (0, 1):6, 
                                        (1, 1):7}
        sample_data = self.all_data[:, :, 0]
        sample_feature = self.gen_features_mat(sample_data, 0)
        self.input_shape = sample_feature.shape

        self.model = self._build_model()
        self.target_model = self._build_model()
        self.best_reward_model = self._build_model()

        self.checkpoint_path_best = f'models_new/best_{self.name}_r_model.h5'
        self.checkpoint_path_t = f'models_new/{self.name}_t_model.h5'
        self.checkpoint_path_q = f'models_new/{self.name}_q_model.h5'

        self.best_reward_val = 0.0

    def _build_model(self):
        print('\nBuilding {} Q-Network with Input Shape: {}\n'.format(self.name, self.input_shape))
        model = models.Sequential([
                layers.Dense(64, activation='relu', input_shape=self.input_shape),
                layers.Dense(64, activation='relu', kernel_regularizer='l2'),
                layers.Dense(self.action_size, activation='relu'),
                layers.Dense(1, activation='linear')
            ])

        model.compile(loss='huber_loss', optimizer=optimizers.Adam(learning_rate=self.alpha), metrics=['mae'])
        return model

    def remember(self, state, action, reward, next_state, t):
        self.memory.append((state.copy(), action, reward, next_state.copy(), t))

    def get_observations(self, state, action, t):
        next_state = state.copy()

        curY, curX = np.where(next_state == dsv_loc_val)

        dirX, dirY = self.map_action_to_direction[action]

        x, y = curX + dirX, curY + dirY
        if not ((0 <= y < next_state.shape[0]) and (0 <= x < next_state.shape[1])).any():
            return np.array([-1, -1, -1, -1, -1])

        next_state[curY, curX] = 0 # previous dsv location

        val = next_state[y, x][0]

        cov = val//60 if val >= 0 else -1

        tfm = val/60 if val >= 0 else -1

        n_neighbors = [(dx, dy) for dx in range(-1, 2) for dy in range(-1, 2) if dx != 0 or dy != 0]

        n_nei = sum(1 for dx, dy in n_neighbors if 0 <= y + dy < next_state.shape[0] and 0 <= x + dx < next_state.shape[1] and next_state[y + dy, x + dx] >= 60)  if val >= 0 else -1 

        nei_tfm = [next_state[y + dy, x + dx] for dx, dy in n_neighbors if 0 <= y + dy < next_state.shape[0] and 0 <= x + dx < next_state.shape[1] and next_state[y + dy, x + dx] >= 0]

        sfm = np.mean(nei_tfm)/60  if val >= 0 else -1 

        path_rem = self.path_length - t - 1

        center_x, center_y = x.item(), y.item()
        start_y, end_y =  max(center_y - path_rem, 0), min(center_y + path_rem, next_state.shape[0])
        start_x, end_x =  max(center_x - path_rem, 0), min(center_x + path_rem, next_state.shape[1])
        region = next_state[start_y:end_y+1, start_x:end_x+1]

        rfm = np.mean(region[region >= 0])/60 if val >= 0 else -1

        return np.array([cov, tfm, n_nei, sfm])
     
    def gen_features_mat(self, state, t):
        state_copy = state.copy()
        y, x = np.where(state_copy == dsv_loc_val)
        features_mat = [self.get_observations(state_copy, action, t) for action in self.map_action_to_direction]
        features_mat = np.array(features_mat)
        min_values = np.min(features_mat, axis=0)
        max_values = np.max(features_mat, axis=0)
        normalized_data = (features_mat - min_values) / (max_values - min_values + 1e-8)

        return normalized_data #features_mat #normalized_data

    def valid_actions(self, state):
        state_copy = state.copy()
        y, x = np.where(state_copy == dsv_loc_val)
        n_neighbors = [(dx, dy) for dx in range(-1, 2) for dy in range(-1, 2) if dx != 0 or dy != 0]

        valid_dirs = [(dx, dy) for dx, dy in n_neighbors if 0 <= y + dy < state.shape[0] and 0 <= x + dx < state.shape[1] and state[y + dy, x + dx] != invalid_loc_val]
        valid_acts = [self.map_direction_to_action[(dx, dy)] for dx, dy in valid_dirs]
        return sorted(valid_acts)

    def choose_action(self, state_features, state, t):

        valid_acts = self.valid_actions(state)
        if np.random.rand() <= self.epsilon:
            return random.choice(valid_acts)  # Explore
        else:
            q_values = self.model.predict(np.array([state_features.copy()]))
            state_copy = state.copy()

            return valid_acts[np.argmax(q_values[0][valid_acts])]  # Exploit

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        states, targets = [], []

        for state, action, reward, next_state, t in minibatch:
            target = reward

            state_features = self.gen_features_mat(state, t)
            if next_state is not None:

                next_state_features = self.gen_features_mat(next_state, t+1)

                next_state_preds = self.target_model.predict(np.array([next_state_features]))

                target = reward + self.gamma * np.amax(next_state_preds[0])

            target_f = self.model.predict(np.array([state_features]))

            target_f[0][action] = target

            states.append(state_features)
            targets.append(target_f[0])

        states = np.array(states)
        targets = np.array(targets)
        test_loss = self.model.evaluate(states, targets, verbose=2)

        self.model.fit(states, targets, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def step(self, state, action, t, return_state=True):
        next_state = state.copy()

        curY, curX = np.where(next_state == dsv_loc_val)

        dirX, dirY = self.map_action_to_direction[action]

        nextX, nextY = curX + dirX, curY + dirY

        next_state[curY, curX] = 0 # previous dsv location

        reward = self.calculate_reward(next_state.copy(), nextY, nextX, t)

        next_state[next_state >= 0] += 1

        next_state[curY, curX] = 0 # previous dsv location

        next_state[nextY, nextX] = dsv_loc_val # new dsv location

        if not return_state:
            return reward 
        return reward, next_state

    def calculate_reward(self, state, y, x, t):
        val = state[y, x]
        if val < -1000:
            return 0.0 - 5.0

        temporal_fairness_reward = val/60 if val >= 0 else -1

        n_neighbors = [(dx, dy) for dx in range(-1, 2) for dy in range(-1, 2) if dx != 0 or dy != 0]
        nei_tfm = [state[y + dy, x + dx]/60 for dx, dy in n_neighbors if 0 <= y + dy < state.shape[0] and 0 <= x + dx < state.shape[1] and state[y + dy, x + dx] >= 0]
        local_fairness_reward = np.mean(nei_tfm) if val >= 0 else -1
        center_y, center_x = y.item(), x.item()
        path_rem = self.path_length - t -1
        start_y, end_y =  max(center_y - path_rem, 0), min(center_y + path_rem, state.shape[0])
        start_x, end_x =  max(center_x - path_rem, 0), min(center_x + path_rem, state.shape[1])
        region = state[start_y:end_y+1, start_x:end_x+1]

        region_fairness_reward = np.mean(region[region >= 0])/60 if val >= 0 else -1

        reward = temporal_fairness_reward + local_fairness_reward + region_fairness_reward
        return reward

    def train(self):
        print(f'\nStarting {self.name} Training...\n')
        best_reward = 0.0

        for episode in range(self.episodes):
            total_reward = 0.0
            train_Samples = 10000
            sampled_indices = np.random.choice(self.all_data.shape[2], train_Samples, replace=False)

            self.train_data = self.all_data[:, :, sampled_indices]
            print('\EPISODE: {}  Randomly Sampled Training State Data Loaded ({} samples) of Shape: {}'.format(episode, self.train_data.shape[2], self.train_data[:, :, 0].shape))

            states = self.train_data.transpose((2, 0, 1))
            print('\nstates.shape: ', states.shape)

            for t in range(self.path_length):

                state_features_mat = np.array([self.gen_features_mat(state, t) for state in states])

                print('state_features_mat.shape: ', state_features_mat.shape)

                actions = np.array([self.choose_action(state_features, states[i], t) for i, state_features in enumerate(state_features_mat)])

                rewards, next_states = zip(*[self.step(states[i], action, t) for i, action in enumerate(actions)])
                for i in range(self.state_size):
                    self.remember(states[i], actions[i], rewards[i], next_states[i], t)

                total_reward += np.sum(rewards)

                states = [next_states[i].copy() for i in range(self.train_data.shape[2])]

                print('\nTIMESTEP: ', t, '  reward: ', np.sum(rewards))

                self.replay(batch_size=self.batch_size)

            print('\nCompleted {} samples; Reward = {}'.format(self.state_size, total_reward))
            print(f"\nEPISODE {episode + 1}, Total Training Reward: {total_reward}, Best Reward: {best_reward}")

            total_reward_val = self.validation()

            if episode == 0:
                self.best_reward_val = total_reward_val

            if self.best_reward_val <= total_reward_val:
                self.best_reward_val = total_reward_val
                self.best_reward_model.set_weights(self.model.get_weights())
                self.best_reward_model.save(self.checkpoint_path_best, save_format='h5')
                print(f"\nBest Model Saved to {self.checkpoint_path_best}")
                
    def choose_action_val(self, state_features, state):
        valid_acts = self.valid_actions(state)

        q_values = self.model.predict(np.array([state_features.copy()]))
        return valid_acts[np.argmax(q_values[0][valid_acts])], np.max(q_values[0][valid_acts])

    def validation(self):
        states = self.val_data.transpose((2, 0, 1))
        print('Validation states.shape: ', states.shape)

        total_reward = 0.0
        for i, state in enumerate(states):

            for t in range(self.path_length):
                state_features = self.gen_features_mat(state, t)

                action, q_val = self.choose_action_val(state_features, state)
                reward, next_state = self.step(state, action, t)

                loss = tf.keras.losses.Huber()(reward, q_val)
                state = next_state.copy() # Updating State for next iteration
                total_reward += reward
        print('\nVALIDATION Completed {} samples; Total Validation Reward = {};  Best Validation Reward= {}'.format(states.shape[0], total_reward, self.best_reward_val))

        return total_reward

class FUSE_DQN_Test_wo5:

    def __init__(self, df, grid_size, n, tf, beta=60, seed=42, km_time=2) -> None:
        random.seed(seed)
        self.name = "fuse_wo5"
        self.df = df
        self.n = n
        self.timeframe = tf
        self.grid_size = grid_size
        self.km_time = int(km_time*grid_size/1000) # km_time to complete 1 km (move to next grid)
        self.path_length = tf//self.km_time # path length per interval

        n_rows, n_cols, nPosns, n_rows_valid, n_cols_valid, nPosns_valid = gridSize_info_dict[self.grid_size]
        self.map_area = np.array([[0 for i in range(n_cols_valid)] for j in range(n_rows_valid)])
        self.area_nrows, self.area_ncols = size_dict[self.n]
        self.pos_neis, self.tfm_map, self.valid_div_pos, self.pos_to_div, self.valid_coords, self.div_coords = generate_neighbors_list_wBorders(self.n, grid_size, self.df, self.name, self.timeframe, self.km_time)
        self.tfm_maps = []
        self.dsvL = [random.sample(list(self.valid_div_pos[i]), 1)[0] for i in range(n)]
        self.dsv_paths = {i:deque([self.dsvL[i]]) for i in range(self.n)}

        self.map_action_to_direction = {0:(-1, -1), 
                                        1:(0, -1), 
                                        2:(1, -1), 
                                        3:(-1, 0), 
                                        4:(1, 0), 
                                        5:(-1, 1), 
                                        6:(0, 1), 
                                        7:(1, 1)}

        self.map_direction_to_action = {(-1, -1):0, 
                                        (0, -1):1, 
                                        (1, -1):2, 
                                        (-1, 0):3, 
                                        (1, 0):4, 
                                        (-1, 1):5, 
                                        (0, 1):6, 
                                        (1, 1):7}

        self.checkpoint_path = f'models_new/best_{self.name}_r_model.h5' #'models_new/best_model_fcn_new_1.h5'
        self.model = models.load_model(self.checkpoint_path) #, custom_objects=custom_objects)    

        print("Q-Network Model Loaded:")
        print(self.model.summary())

        print("\n{} Model Object Created".format(self.name))
        print("\nValid Locations Per Region", [len(self.valid_div_pos[i]) for i in range(self.n)])
        print('\nArea Border of Division (start_row, end_row, start_col, end_col):', self.div_coords)
        print("\nStudy Area Gird Shape: ", self.tfm_map.shape)
        print("DSV Starting Locations: ",self.dsvL)
        print("DSV Starting Grid Positions: ", [(pos//n_cols, pos%n_cols) for pos in self.dsvL])
        print('Path Length: ', self.path_length)
                
    def valid_actions(self, state):
        state_copy = state.copy()
        y, x = np.where(state_copy == dsv_loc_val)
        n_neighbors = [(dx, dy) for dx in range(-1, 2) for dy in range(-1, 2) if dx != 0 or dy != 0]

        valid_dirs = [(dx, dy) for dx, dy in n_neighbors if 0 <= y + dy < state.shape[0] and 0 <= x + dx < state.shape[1] and state[y + dy, x + dx] != invalid_loc_val]
        valid_acts = [self.map_direction_to_action[(dx, dy)] for dx, dy in valid_dirs]
        return sorted(valid_acts)

    def choose_action(self, state_features, state):
        valid_acts = self.valid_actions(state)

        q_values = self.model.predict(np.array([state_features.copy()]))
        return valid_acts[np.argmax(q_values[0][valid_acts])]

    def take_action(self, state, action, globalY, globalX):
        next_state = state.copy()

        curY, curX = np.where(next_state == dsv_loc_val)

        dirX, dirY = self.map_action_to_direction[action]

        nextX, nextY = curX + dirX, curY + dirY

        next_globalX, next_globalY = globalX + dirX, globalY + dirY # to update the global study area

        next_state[curY, curX] = 0 # previous dsv location

        next_state[next_state >= 0] += 1

        next_state[curY, curX] = 0 # previous dsv location

        next_state[nextY, nextX] = dsv_loc_val # new dsv location

        return next_state, nextY, nextX, next_globalY, next_globalX

    def run(self, vid):
        ''' 
        INPUTS:

        OUTPUTS:
        '''    

        cur_pos = self.dsvL[vid]
        dsv_path = [cur_pos]
        global_dsvY, global_dsvX = pos_to_coord(self.grid_size, cur_pos)
        div_id, localY, localX = self.pos_to_div[(global_dsvY, global_dsvX)]

        state = gen_state(self.tfm_maps[div_id], localY, localX, self.timeframe, self.km_time)

        for t in range(self.path_length):

            state_features = self.gen_features_mat(state, t)

            action = self.choose_action(state_features, state)
            next_state, local_dsvY, local_dsvX, global_dsvY, global_dsvX = self.take_action(state, action, global_dsvY, global_dsvX)

            state = next_state.copy() # Updating State for next iteration
            next_pos = coord_to_pos(self.grid_size, global_dsvY, global_dsvX)
            dsv_path.append(next_pos)

        self.tfm_maps[div_id], self.tfm_map = update_map(self.tfm_map, self.tfm_maps[div_id], state, localY, localX, self.div_coords[div_id], self.timeframe, self.km_time )
        self.dsvL[vid] = next_pos
        return dsv_path

    def parallel_run(self, curTime):
        ''' 
        INPUTS:

        OUTPUTS:

        '''
        N = [ele for ele in range(self.n)]
        nthreads_obj = self.n
        dsv_paths = {i:[self.dsvL[i]] for i in range(self.n)}
        time_t_prev = curTime
        times_print = [str(curTime)]
        paths_print = []

        time_t_prev = curTime
        next_time_t = time_t_prev + timedelta(minutes=self.timeframe)

        for t in range(self.path_length): # range(self.path_int+1)
            interval = timedelta(minutes=self.km_time-1)
            time_t = time_t_prev + interval

            data_interval = fetch_rows(self.df, time_t_prev, time_t)

            locs_interval_ts = get_FHV_cov_coord(self.grid_size, data_interval, self.valid_coords)

            self.tfm_map = update_tfm_sfm_tEnd(self.tfm_map, self.valid_coords, self.path_length, testing=True)

            self.tfm_map = update_tfm_sfm_cov(self.tfm_map, locs_interval_ts, self.path_length, testing=True)

            times_print.append(str(time_t))
            time_t_prev = time_t_prev + timedelta(minutes=self.km_time)
        self.tfm_maps = get_mat2D_nDivs_fromFullMap(self.grid_size, self.df, self.tfm_map, self.n, self.path_length, self.valid_coords)

        return_value = [self.run(vid) for vid in N]

        for vid, best_path in enumerate(return_value):
            dsv_paths[vid] = best_path

        print("DSV Locations At End ({}):{}".format(next_time_t, self.dsvL))  
        return dsv_paths

    def get_observations(self, state, action, t):
        next_state = state.copy()

        curY, curX = np.where(next_state == dsv_loc_val)

        dirX, dirY = self.map_action_to_direction[action]

        x, y = curX + dirX, curY + dirY

        if not (0 <= y < next_state.shape[0] and 0 <= x < next_state.shape[1]):
            return np.array([-1, -1, -1, -1, -1])

        next_state[curY, curX] = 0 # previous dsv location

        val = next_state[y, x][0]

        cov = val//60 if val >= 0 else -1

        tfm = val/60 if val >= 0 else -1

        n_neighbors = [(dx, dy) for dx in range(-1, 2) for dy in range(-1, 2) if dx != 0 or dy != 0]

        n_nei = sum(1 for dx, dy in n_neighbors if 0 <= y + dy < next_state.shape[0] and 0 <= x + dx < next_state.shape[1] and next_state[y + dy, x + dx] >= 60)  if val >= 0 else -1 

        nei_tfm = [next_state[y + dy, x + dx] for dx, dy in n_neighbors if 0 <= y + dy < next_state.shape[0] and 0 <= x + dx < next_state.shape[1] and next_state[y + dy, x + dx] >= 0]

        sfm = np.mean(nei_tfm)/60  if val >= 0 else -1 

        path_rem = self.path_length - t - 1

        center_x, center_y = x.item(), y.item()
        start_y, end_y =  max(center_y - path_rem, 0), min(center_y + path_rem, next_state.shape[0])
        start_x, end_x =  max(center_x - path_rem, 0), min(center_x + path_rem, next_state.shape[1])
        region = next_state[start_y:end_y+1, start_x:end_x+1]

        rfm = np.mean(region[region >= 0])/60 if val >= 0 else -1

        return np.array([cov, tfm, n_nei, sfm])
     
    def gen_features_mat(self, state, t):
        state_copy = state.copy()
        y, x = np.where(state_copy == dsv_loc_val)
        features_mat = [self.get_observations(state_copy, action, t) for action in self.map_action_to_direction]
        features_mat = np.array(features_mat)
        min_values = np.min(features_mat, axis=0)
        max_values = np.max(features_mat, axis=0)
        normalized_data = (features_mat - min_values) / (max_values - min_values + 1e-8)

        return normalized_data #features_mat #normalized_data
