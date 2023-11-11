# CS712 - Submission by Team 6

This project contains all source files, as well as results/logs, for our submission. All sample commands below should work on Linux (including WSL2 on Windows) or Mac.

## How to run it

1. create a subfolder `data` and put train/val/test image folders under `data` folder.

```
$ tree -L 1 data
data
├── test
├── train
└── validation
```

2. install all depedencies:
```
$ pip install -r requirements.txt
```

3. preprocess images with the following command:
* preprocessed results are stored as `data/timm_preprocessed_*.npy` files.
```
python preprocess_timm_v2.py
```

4. transform data with the following command:

* transformed data results are stored as `data/distance_timm_preprocessed_*.npy` files.

```
python transform_data.py
```

5. train the model with the following command.
* model checkpoints for all epochs are stored under subfolders `data/checkpoints/`.
```
$ python train_v9.py
```

6. to reproduce results sumbitted to the public leaderboard,
* run the following command which will load the checkpoints from step 4) to generate results.
* validation results are stored in `data/validation.txt` file which is compressed into `data/result.zip`.
```
python validate-timm.py
```

7. to reproduce results sumbitted to the private leaderboard,
* run the following command which will load the checkpoints from step 3) to generate results.
* test results are stored in `data/test.txt` file which is compressed into `data/test-result.zip`. 
```
python validate-timm.py -n test
```
