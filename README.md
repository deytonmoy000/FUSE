# FUSE: Fair Urban Sensing with Submodular Spatio-Temporal Reward Maximization



## Experiment Data Files

- [ ] Please download and store the preprocessed GPS readings [data_new.csv](https://drive.google.com/file/d/1-tFdqdS1qdVb6PeduiHRG5lwGGmbQ4Ni/view?usp=share_link) (combined data from the source data files provided below) in the $/data$ directory.
- [ ] Please download and store the synthetic states training data file [training_data_combined.h5](https://drive.google.com/file/d/1V8l9otNp77B73klXp82WZc7nMy5SlktO/view?usp=share_link) in the $/data/training$ directory.

## Code Files
- [ ] [utils.py]() Contains all the utility functions.

- [ ] [main.py]() Runs the experiment.

- [ ] [algs.py]() Contains All Algorithms.

## Experiment Run Files
-  Generate the data for the provided *grid_size*:
    - Run *python gen_data.py  **gridSize**( in metres [500, 1000])*
-  Run experiments for all algorithms:
    -  Run *bash  run.sh*  
-  Run algorithm experiments individually:
    -  Run *python main.by **nRegions**  **TimeSlotDuration**  **algName**  **seed**  **gridSize**(metres)*. 
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
  
## Training FUSE
-  To train *FAD<sup>2</sup>QN* using **(1Km)<sup>2</sup>** Grid Size, Time Slot duration of **20** minutes and with the Study Area divided into **4** Regions:
    - Run *python main.by  **4  20  fuse_train  42  1000***   


## Data Source Files

- [ ] [TaxiData.csv](http://www-users.cs.umn.edu/~tianhe/BIGDATA/UrbanCPS/TaxiData/TaxiData).
- [ ] [BusData.csv](http://www-users.cs.umn.edu/~tianhe/BIGDATA/UrbanCPS/BusData/BusData).
- [ ] [TruckData.csv](http://www-users.cs.umn.edu/~tianhe/BIGDATA/UrbanCPS/TruckData/TruckData).
- [ ] [EVData.rar](http://guangwang.me/files/ETData.rar) Please extract EVData.csv.


