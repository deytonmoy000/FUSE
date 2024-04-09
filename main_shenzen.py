
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

import collections
from datetime import datetime, timedelta
import time
import random
import json
import h5py

import tensorflow as tf

import concurrent.futures
import os
import sys

os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1"
from queue import PriorityQueue
from utils_shenzen import create_grid_json_updated, csv_to_grid_updated, grid_to_map_updated, gen_granular_data, update_grids_info, raw_data_preprocessing, raw_data_preprocessing_old, gen_fairness_heatMap, gen_train_data, load_and_preprocess_training_data, read_training_data
from utils_shenzen import generate_neighbors_list, generate_time_intervals, generate_path_dict, generate_neighbors_list_wBorders, get_default_cnt_mat, save_df, path_grid_map
from utils_shenzen import read_data, load_neighbors_list, generate_path_dict_forTimeFrames, load_time_intervals, load_path_dict, initialize_STF_weights, get_initial_loc_DSVs, fetch_rows, generate_path_dict_timeframe
from utils_shenzen import get_FHV_cov, update_TSFMetrics, update_TSFMetrics_tEnd
from algs_shenzen import FUSE_DQN_Train, FUSE_DQN_Test, FUSE_DQN_Train_wo12, FUSE_DQN_Train_wo34, FUSE_DQN_Train_wo5, FUSE_DQN_Test_wo12, FUSE_DQN_Test_wo34, FUSE_DQN_Test_wo5, TSMTC, REASSIGN, AGD, SDPR
if __name__ == "__main__":

    nDiv = int(sys.argv[1])
    tf = int(sys.argv[2])
    alg = sys.argv[3]
    seed = int(sys.argv[4])
    grid_size = int(sys.argv[5])

    grid_size_fname = str(grid_size)

    print("\n********-----********-----********-----********-----*******\n")
    print("\nNumber of Divisions: {}    Time Interval: {}    Seed (Randomization): {}".format(nDiv, tf, seed))

    generate_time_intervals(tf)

    time_intervals = load_time_intervals(tf)

    start_time, end_time = time_intervals[0][0], time_intervals[-1][-1]
    print("Start Time: {} - End Time: {}".format(start_time, end_time))
    print("\n********-----********-----********-----********-----*******\n")
    
    data_fname = f'data/data_new_{grid_size}m.csv'
    df = read_data(data_fname, start_time, end_time)

    n_rows, n_cols, nPosns, n_rows_valid, n_cols_valid, nPosns_valid = update_grids_info(df, grid_size)
    print("\n********-----********-----********-----********-----*******\n")

    print("\n********-----********-----********-----********-----*******\n")

    if alg == "TSMTC":
        model = TSMTC(df, grid_size, nDiv, tf, seed=seed)

    elif alg == "REASSIGN":
        model = REASSIGN(df, grid_size, nDiv, tf, seed=seed)

    elif alg == "AGD":
        model = AGD(df, grid_size, nDiv, tf, seed=seed)

    elif alg == "SDPR":
        model = SDPR(df, grid_size, nDiv, tf, seed=seed)

    elif alg == 'fuse_test':
        model = FUSE_DQN_Test(df, grid_size, nDiv, tf, seed=seed)
        print("MAX TFM MAP: ", np.max(model.tfm_map), "   MINTFM MAP: ", np.min(model.tfm_map))

    elif alg == 'fuse_test_wo12':
        model = FUSE_DQN_Test_wo12(df, grid_size, nDiv, tf, seed=seed)
        print("MAX TFM MAP: ", np.max(model.tfm_map), "   MINTFM MAP: ", np.min(model.tfm_map))

    elif alg == 'fuse_test_wo34':
        model = FUSE_DQN_Test_wo34(df, grid_size, nDiv, tf, seed=seed)
        print("MAX TFM MAP: ", np.max(model.tfm_map), "   MINTFM MAP: ", np.min(model.tfm_map))

    elif alg == 'fuse_test_wo5':
        model = FUSE_DQN_Test_wo5(df, grid_size, nDiv, tf, seed=seed)
        print("MAX TFM MAP: ", np.max(model.tfm_map), "   MINTFM MAP: ", np.min(model.tfm_map))

    elif alg == "fuse_train":
        train_data_fname = 'data/training/training_data_combined.h5'
        X_data = read_training_data(train_data_fname, mat_name='training_data')
        print('Training State Data Loaded {} samples of Shape: {}'.format(X_data.shape[2], X_data[:, :, 0].shape))

        n_Samples = 10000
        val_Samples = 5000

        sampled_indices_val = np.random.choice(X_data.shape[2], val_Samples, replace=False)
        sampled_val_data = X_data[:, :, sampled_indices_val]

        model = FUSE_DQN_Train(X_data, sampled_val_data, n_Samples, tf)

        print("\n********-----********-----********-----********-----*******\n")
        model.train()

        print("\n********-----********-----********-----********-----*******\n")
        exit()

    elif alg == "fuse_train_wo12":
        train_data_fname = 'data/training/training_data_combined.h5'
        X_data = read_training_data(train_data_fname, mat_name='training_data')
        print('Training State Data Loaded {} samples of Shape: {}'.format(X_data.shape[2], X_data[:, :, 0].shape))

        n_Samples = 10000
        val_Samples = 5000

        sampled_indices_val = np.random.choice(X_data.shape[2], val_Samples, replace=False)
        sampled_val_data = X_data[:, :, sampled_indices_val]

        model = FUSE_DQN_Train_wo12(X_data, sampled_val_data, n_Samples, tf)

        print("\n********-----********-----********-----********-----*******\n")
        model.train()

        print("\n********-----********-----********-----********-----*******\n")
        exit()

    elif alg == "fuse_train_wo34":
        train_data_fname = 'data/training/training_data_combined.h5'
        X_data = read_training_data(train_data_fname, mat_name='training_data')
        print('Training State Data Loaded {} samples of Shape: {}'.format(X_data.shape[2], X_data[:, :, 0].shape))

        n_Samples = 10000
        val_Samples = 5000

        sampled_indices_val = np.random.choice(X_data.shape[2], val_Samples, replace=False)
        sampled_val_data = X_data[:, :, sampled_indices_val]

        model = FUSE_DQN_Train_wo34(X_data, sampled_val_data, n_Samples, tf)

        print("\n********-----********-----********-----********-----*******\n")
        model.train()

        print("\n********-----********-----********-----********-----*******\n")
        exit()

    elif alg == "fuse_train_wo5":
        train_data_fname = 'data/training/training_data_combined.h5'
        X_data = read_training_data(train_data_fname, mat_name='training_data')
        print('Training State Data Loaded {} samples of Shape: {}'.format(X_data.shape[2], X_data[:, :, 0].shape))

        n_Samples = 10000
        val_Samples = 5000

        sampled_indices_val = np.random.choice(X_data.shape[2], val_Samples, replace=False)
        sampled_val_data = X_data[:, :, sampled_indices_val]

        model = FUSE_DQN_Train_wo5(X_data, sampled_val_data, n_Samples, tf)

        print("\n********-----********-----********-----********-----*******\n")
        model.train()

        print("\n********-----********-----********-----********-----*******\n")
        exit()
    print("\n********-----********-----********-----********-----*******\n")
    start_locs = set(model.dsvL)
    locs_visited_cnt = get_default_cnt_mat(grid_size, df)
    veh_locs_visited_cnt = get_default_cnt_mat(grid_size, df)
    dsv_locs_visited_cnt = get_default_cnt_mat(grid_size, df)
    fhv_locs_visited_cnt = get_default_cnt_mat(grid_size, df)
    locs_visited_cnt_24h = get_default_cnt_mat(grid_size, df)
    veh_locs_visited_cnt_24h = get_default_cnt_mat(grid_size, df)
    dsv_locs_visited_cnt_24h = get_default_cnt_mat(grid_size, df)
    fhv_locs_visited_cnt_24h = get_default_cnt_mat(grid_size, df)
    dsv_locs = {i:[] for i in range(nDiv)}

    times_vec = []
    cov_vec = []
    std_vec = []
    mean_vec = []
    cov_percent = []
    models_vec = []
    new_locs_vec = []
    csv_locs_vec = []
    dsv_locs_vec = []
    eff_vec = []
    std_veh_loc_vec = []
    mean_runtime_vec = []
    mean_runtime_2h_vec = []
    cov_vec_24h = []
    std_vec_24h = []
    mean_vec_24h = []
    cov_percent_24h = []
    eff_vec_24h = []
    std_veh_loc_vec_24h = []
    new_locs_vec_24h = []
    csv_locs_vec_24h = []
    dsv_locs_vec_24h = []

    cov_h = []
    runtimes = []
    full_start_time = time.time()
    for t in range(24*int(60//tf)): # len(time_intervals)):

        interval = timedelta(minutes=tf-1)
        curTime = time_intervals[t][0]
        nextTime = time_intervals[t][1]
        time_t = curTime + interval

        data_interval = fetch_rows(df, curTime, time_t)
        locs_interval = get_FHV_cov(data_interval, model.pos_neis)
        cov_h.extend(locs_interval)
        locs_visited_cnt[locs_interval] += 1
        locs_visited_cnt_24h[locs_interval] += 1
        fhv_locs_visited_cnt[locs_interval] += 1
        fhv_locs_visited_cnt_24h[locs_interval] += 1

        print("\nTime Slot StartTime: ", curTime, "  EndTime: ", time_t)

        start_time = time.time()
        dsv_paths = model.parallel_run(curTime)
        end_time = time.time()
        runtime = end_time - start_time
        runtimes.append(runtime)
        print(dsv_paths)

        path_cov = []
        for i in range(len(dsv_paths)):
            dsv_locs[i].extend(dsv_paths[i][:-1])
            for j in range(len(dsv_paths[i][:-1])):
                locs_visited_cnt[dsv_paths[i][j]] += 1
                locs_visited_cnt_24h[dsv_paths[i][j]] += 1
                veh_locs_visited_cnt_24h[dsv_paths[i][j]] += 1
                veh_locs_visited_cnt[dsv_paths[i][j]] += 1
                dsv_locs_visited_cnt[dsv_paths[i][j]] += 1
                dsv_locs_visited_cnt_24h[dsv_paths[i][j]] += 1

                path_cov.extend(dsv_locs[i])
        if t%(int(60//tf)) == (int(60//tf)-1):
            cov_vec.append(sum(locs_visited_cnt > 0))
            std_vec.append(np.std(locs_visited_cnt[locs_visited_cnt >= 0]))
            mean_vec.append(np.mean(locs_visited_cnt[locs_visited_cnt >= 0]))
            mean_runtime_vec.append(np.mean(runtimes))
            runtimes = []
            cov_percent.append(sum(locs_visited_cnt > 0)/2017)

            veh_locs = set(np.where(veh_locs_visited_cnt > 0)[0])
            fhv_locs = set(np.where(fhv_locs_visited_cnt > 0)[0])

            combined_locs = set(np.where(locs_visited_cnt > 0)[0])
            locs_added = veh_locs - fhv_locs
            veh_locs_cnts = locs_visited_cnt[list(veh_locs)]
            print("\n{} Hourly Total Cov: {}  --FHV Cov: {}  --DSV Cov: {}  --New Locations: {}".format(nextTime, len(combined_locs), len(fhv_locs), len(veh_locs), len(locs_added)))
            print(len(combined_locs), len(veh_locs), len(fhv_locs), len(list(set(combined_locs) - set(veh_locs))), len(veh_locs_cnts),"\n")
            new_locs_vec.append(len(locs_added))
            csv_locs_vec.append(len(fhv_locs))
            dsv_locs_vec.append(len(veh_locs))
            eff_vec.append(len(locs_added)/((60//model.km_time)*nDiv))
            std_veh_loc_vec.append(np.std(veh_locs_cnts))

            path_cov = []
            for i in range(len(dsv_locs)):
                path_cov.extend(dsv_locs[i])
            path_image_fname = path_grid_map(grid_size, path_cov, model.name, alg, list(set(cov_h)), t//(int(60//tf)), start_locs, nDiv, seed, tf, int(grid_size/1000*2))

            cov_h = []
            locs_visited_cnt = get_default_cnt_mat(grid_size, df)
            veh_locs_visited_cnt = get_default_cnt_mat(grid_size, df)
            fhv_locs_visited_cnt = get_default_cnt_mat(grid_size, df)

            times_vec.append(nextTime)
            models_vec.append(model.name)

            print_data = pd.DataFrame(
                    {
                        'seed': [seed for _ in range(t//(int(60//tf))+1)],
                        'Time(h)': times_vec,
                        'model': models_vec,
                        'Coverage': cov_vec,
                        'Mean': mean_vec,
                        'Std_Dev': std_vec,
                        'CSV_Coverage': csv_locs_vec,
                        'DSV_Coverage': dsv_locs_vec,
                        'New_locs': new_locs_vec,
                        'Cov_Efficiency': eff_vec,
                        'CovDev_Vehicle_Path': std_veh_loc_vec,
                        'Mean_Runtime': mean_runtime_vec
                    }
                )
            fname = f'results_paper_Shenzen/GS_{grid_size_fname}m/TF_'+str(tf)+'/Div_'+str(nDiv)+'/Rep_'+str(seed)+'/metrics_'+str(model.name)+'_'+str(tf)+'_'+str(nDiv)+'divs' + '_'+str(seed)+'.csv'
            save_df(print_data, fname)

            print("\n{} {} Hourly Path Map ({}) and Results ({}) Generated and Saved\n********-----********-----********-----********-----*******\n".format(nextTime, model.name, path_image_fname, fname))

    full_end_time = time.time()
    full_runtime = full_end_time - full_start_time

    print_data = pd.DataFrame(
                    {
                        'seed': [seed for _ in range(t//(int(60//tf))+1)],
                        'Time(h)': times_vec,
                        'model': models_vec,
                        'Coverage': cov_vec,
                        'Mean': mean_vec,
                        'Std_Dev': std_vec,
                        'CSV_Coverage': csv_locs_vec,
                        'DSV_Coverage': dsv_locs_vec,
                        'New_locs': new_locs_vec,
                        'Cov_Efficiency': eff_vec,
                        'CovDev_Vehicle_Path': std_veh_loc_vec,
                        'Mean_Runtime': mean_runtime_vec
                    }
                )

    for i in range(len(dsv_paths)):
        dsv_locs[i].append(model.dsvL[i])
        locs_visited_cnt_24h[model.dsvL[i]] += 1
        veh_locs_visited_cnt_24h[model.dsvL[i]] += 1

    cov_vec_24h.append(sum(locs_visited_cnt_24h > 0))
    std_vec_24h.append(np.std(locs_visited_cnt_24h[locs_visited_cnt_24h >= 0]))
    mean_vec_24h.append(np.mean(locs_visited_cnt_24h[locs_visited_cnt_24h >= 0]))
    cov_percent_24h.append(sum(locs_visited_cnt_24h > 0)/2017)
    veh_locs_24h = set(np.where(veh_locs_visited_cnt_24h > 0)[0])
    fhv_locs_24h = set(np.where(fhv_locs_visited_cnt_24h > 0)[0])
    dsv_locs_24h = set(np.where(dsv_locs_visited_cnt_24h > 0)[0])

    combined_locs_24h = set(np.where(locs_visited_cnt_24h > 0)[0])
    locs_added_24h = veh_locs_24h - fhv_locs_24h

    new_locs_vec_24h.append(len(locs_added_24h))
    csv_locs_vec_24h.append(len(fhv_locs_24h))
    dsv_locs_vec_24h.append(len(dsv_locs_24h))

    eff_vec_24h.append(len(locs_added_24h)/((60//model.km_time)*24*nDiv))
    std_veh_loc_vec_24h.append(np.std(fhv_locs_visited_cnt_24h[veh_locs_visited_cnt_24h > 0]))
    mean_runtime_2h_vec.append(full_runtime)

    avg_runtime, avg_cov, avg_cov_percent, avg_mean, avg_std_dev, avg_new_locs, avg_eff, avg_veh_covdev = np.mean(mean_runtime_vec), np.mean(cov_vec), np.mean(cov_percent), np.mean(mean_vec), np.mean(std_vec), np.mean(new_locs_vec), np.mean(eff_vec), np.mean(std_veh_loc_vec)
    avg_csv_locs, avg_dsv_locs = np.mean(csv_locs_vec), np.mean(dsv_locs_vec)

    data_avg = pd.DataFrame(
                    {
                        'seed': [seed], 
                        'Time(h)': '2023-01-01 50:00:00',
                        'model': model.name,
                        'Coverage': [avg_cov],
                        'Mean': [avg_mean],
                        'Std_Dev': [avg_std_dev],
                        'CSV_Coverage': [avg_csv_locs],
                        'DSV_Coverage': [avg_dsv_locs],
                        'New_locs': [avg_new_locs],
                        'Cov_Efficiency': [avg_eff],
                        'CovDev_Vehicle_Path': [avg_veh_covdev],
                        'Mean_Runtime': [avg_runtime]
                    }
                )

    data_24h = pd.DataFrame(
                    {
                        'seed': [seed], 
                        'Time(h)': '2023-01-01 100:00:00',
                        'model': model.name,
                        'Coverage': cov_vec_24h,
                        'Mean': mean_vec_24h,
                        'Std_Dev': std_vec_24h,
                        'CSV_Coverage': csv_locs_vec_24h,
                        'DSV_Coverage': dsv_locs_vec_24h,
                        'New_locs': new_locs_vec_24h,
                        'Cov_Efficiency': eff_vec_24h,
                        'CovDev_Vehicle_Path': std_veh_loc_vec_24h,
                        'Mean_Runtime': mean_runtime_2h_vec
                    }
                )

    print_data = pd.concat([print_data, data_avg, data_24h])
    fname = f'results_paper_Shenzen/GS_{grid_size_fname}m/TF_'+str(tf)+'/Div_'+str(nDiv)+'/Rep_'+str(seed)+'/metrics_'+str(model.name)+'_'+str(tf)+'_'+str(nDiv)+'divs' + '_'+str(seed)+'.csv'
    save_df(print_data, fname)
    print("\n{} Full Day Results Generated and Saved ({})\n********-----********-----********-----********-----*******\n".format(model.name, fname))
    print("\nRouting Complete\n********-----********-----********-----********-----*******\n")

