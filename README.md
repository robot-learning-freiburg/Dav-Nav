# Catch Me If You Hear Me: Audio-Visual Navigation in Complex Unmapped Environments with Moving Sounds

This Repository provides the source code for the paper "Catch Me If You Hear Me: Audio-Visual Navigation in Complex Unmapped Environments with Moving Sounds", see the [project website](http://dav-nav.cs.uni-freiburg.de/).  

https://user-images.githubusercontent.com/42060869/185898432-4e5b6d80-374d-41eb-aa7d-99d958202c40.mp4

Please cite the paper as follows:

      @article{younes2021catch,
            title={Catch Me If You Hear Me: Audio-Visual Navigation in Complex Unmapped Environments with Moving Sounds},
            author={Younes, Abdelrahman and Honerkamp, Daniel  and Welschehold, Tim  and Valada, Abhinav},
            journal={arXiv preprint arXiv:2111.14843},
            year={2021},
      }
      
## Installation 
1. Create conda env:
```
   conda create -n habitat python=3.6 cmake=3.14.0
   conda activate habitat
 ```
2. Install habitat-lab v0.1.6:
    Download the source code:
    ```
    wget https://github.com/facebookresearch/habitat-lab/archive/refs/tags/v0.1.6.tar.gz
    cd habitat-lab-0.1.6
    pip install -r requirements.txt
    python setup.py develop --all # install habitat and habitat_baselines
   ```
3. Install habitat-sim v0.1.6:
   ```
    conda install habitat-sim=0.1.6 -c conda-forge -c aihabitat
    conda install habitat-sim=0.1.6 headless -c conda-forge -c aihabitat # for headless machines
    ```
4. Install this repo into pip by running the following command:
    ```pip install -e .```
5. Install pytorch with torchaudio and torch vision
   ```conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge```   
6. Follow the instructions on the [dataset](https://github.com/facebookresearch/sound-spaces/blob/main/soundspaces/README.md) page to download the rendered audio data and datasets,
-> move into data/scene_datasets, then run ```python scripts/cache_observations.py``` to create the data/scene_observations folder

Below we show the commands for training and evaluating AudioGoal on Replica, 
but it applies to the other two tasks, other sensors and Matterport dataset as well. 
1. Training
```
python ss_baselines/dav_nav/run.py --exp-config ss_baselines/dav_nav/config/audionav/replica/static_source/without_complex_scenarios/train_multiple.yaml --model-dir data/models/replica/audiogoal_depth
```
2. Validation (evaluate each checkpoint and generate a validation curve)
```
python ss_baselines/dav_nav/run.py --run-type eval --exp-config ss_baselines/dav_nav/config/audionav/replica/static_source/without_complex_scenarios/val_multiple.yaml --model-dir data/models/replica/audiogoal_depth
```
3. Test the best validation checkpoint based on validation curve
```
python ss_baselines/dav_nav/run.py --run-type eval --exp-config ss_baselines/dav_nav/config/audionav/replica/static_source/without_complex_scenarios/test_multiple.yaml --model-dir data/models/replica/audiogoal_depth EVAL_CKPT_PATH_DIR data/models/replica/audiogoal_depth/data/ckpt.XXX.pth
```

__

Please follow the following steps to update the above commands according to the desired experiment.
1. Select which model you want to run 
2. Navigate to ss_baselines/"selected_model"/config/audionav/
3. Select which dataset you'd like to train on (Replica or Matterport3d) 
4. Select which audiogoal task you want to train your agent on (with static_source or dynamic_source) 
5. Choose whether to include comple scenarios or not
6. Finally, select the desired config file
7. Pass the path to this config file to --exp-config argument in the above commands

#Note: the project's structure is based on [Sound-Spaces](https://github.com/facebookresearch/sound-spaces) Simulator repository  
