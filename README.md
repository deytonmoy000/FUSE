# FairSense: Fairness-Aware Urban Sensing with Submodular Spatio-Temporal Reward Maximization

## Shenzen Files

### Experiment Data Files

- [ ] Please download and store the preprocessed Shenzen GPS readings [data_new.csv](https://drive.google.com/file/d/1-tFdqdS1qdVb6PeduiHRG5lwGGmbQ4Ni/view?usp=share_link) (combined data from the source data files provided below) in the $/data$ directory.


### Code Files
- [ ] [utils_shenzen.py](https://github.com/deytonmoy000/FUSE/blob/master/utils_shenzen.py) Contains all the utility functions.

- [ ] [main_shenzen.py](https://github.com/deytonmoy000/FUSE/blob/master/main_shenzen.py) Runs the experiment.

- [ ] [algs_shenzen.py](https://github.com/deytonmoy000/FUSE/blob/master/algs_shenzen.py) Contains All Algorithms.

### Experiment Run Files
-  Generate the data for the provided *grid_size*:
    - Run *python gen_data_shenzen.py  **gridSize**( in metres [500, 1000])*
-  Run experiments for all algorithms:
    -  Run *bash  run_shenzen.sh*  
-  Run algorithm experiments individually:
    -  Run *python main_shenzen.by **nRegions**  **TimeSlotDuration**  **algName**  **seed**  **gridSize**(metres)*. 
        -  nRegions : **[1, 2, 4, 8]**
        -  TimeSlotDuration: **[15, 30, 60]**
        -  gridSize : **[500, 1000]**
        -  algName : 
            - **TSMTC** :TSMTC,
            - **REASSIGN** :REASSIGN,
            - **SDPR** :SDPR,
            - **AGD** :AGD,
            - **fuse_test** : for testing FUSE,
            - **fuse_train** :for training FUSE

## Beijing Files

### Experiment Data Files

- [ ] Please download and store the preprocessed Shenzen GPS readings [data_beijing_updated.csv](https://drive.google.com/file/d/1XSBVnWgA4vwrrJtMvW750Kx13XICnpiR/view?usp=share_link) (combined data from the source data files provided below) in the $/data$ directory.


### Code Files
- [ ] [utils_beijing.py](https://github.com/deytonmoy000/FUSE/blob/master/utils_beijing.py) Contains all the utility functions.

- [ ] [main_beijing.py](https://github.com/deytonmoy000/FUSE/blob/master/main_beijing.py) Runs the experiment.

- [ ] [algs_beijing.py](https://github.com/deytonmoy000/FUSE/blob/master/algs_beijing.py) Contains All Algorithms.

### Experiment Run Files
-  Generate the data for the provided *grid_size*:
    - Run *python gen_data_beijing.py  **gridSize**( in metres [500, 1000])*
-  Run experiments for all algorithms:
    -  Run *bash  run_beijing.sh*  
-  Run algorithm experiments individually:
    -  Run *python main_beijing.by **nRegions**  **TimeSlotDuration**  **algName**  **seed**  **gridSize**(metres)*. 
        -  nRegions : **[1, 2, 4, 8]**
        -  TimeSlotDuration: **[15, 30, 60]**
        -  gridSize : **[500, 1000]**
        -  algName : 
            - **TSMTC** :TSMTC,
            - **REASSIGN** :REASSIGN,
            - **SDPR** :SDPR,
            - **AGD** :AGD,
            - **fuse_test** : for testing FUSE,
            - **fuse_train** :for training FUSE



## Training FairSense
- [ ] Please download and store the synthetic states training data file [training_data_combined.h5](https://drive.google.com/file/d/1V8l9otNp77B73klXp82WZc7nMy5SlktO/view?usp=share_link) in the $/data/training$ directory.
-  To train *FADQN* using **(1Km)<sup>2</sup>** Grid Size, Time Slot duration of **20** minutes and with the Study Area divided into **4** Regions:
    - Run *python main_shenzen.by  **4  20  fuse_train  42  1000***   


## Data Source Files

- [ ] [TaxiData.csv](http://www-users.cs.umn.edu/~tianhe/BIGDATA/UrbanCPS/TaxiData/TaxiData).
- [ ] [BusData.csv](http://www-users.cs.umn.edu/~tianhe/BIGDATA/UrbanCPS/BusData/BusData).
- [ ] [TruckData.csv](http://www-users.cs.umn.edu/~tianhe/BIGDATA/UrbanCPS/TruckData/TruckData).
- [ ] [EVData.rar](http://guangwang.me/files/ETData.rar) Please extract EVData.csv.
- [ ] [T-drive Taxi Trajectories.zip](https://onedrive.live.com/?authkey=%21ADgmvTgfqs4hn4Q&cid=CF159105855090C5&id=CF159105855090C5%2141466&parId=CF159105855090C5%211438&o=OneUp) Please extract and merge the files.


