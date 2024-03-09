# Maneuver-Conditioned Decision Transformer for Tactical In-Flight Decision-Making
This repo is the PyTorch implementation of our paper "Maneuver-Conditioned Decision Transformer for Tactical In-Flight Decision-Making".

## Install
We tested the code in Ubuntu 20.04. 
 - We recommend using Anaconda to create a virtual environment.

```
conda create --name M-DT python=3.8.5
conda activate M-DT
```
 - Our experiments require JSBSim as well as JSBSim-py. Install them by following the instructions in the [JSBSim-repo](https://github.com/JSBSim-Team/jsbsim).

 - We also implemented an environment using Mujoco-py to ensure that our model operates in a typical benchmark environment. It require MuJoCo as well as mujoco-py. Install them by following the instructions in the mujoco-py repo.
 - Install benchmark environments and dependencies with the following commands:
```
# install dependencies
pip install -r requirements.txt

# install benchmark environments
./install_benchmark_envs.sh
```

## Data
 - After our related ongoing project is finalized, we will upload the offline dataset.
 
 - Instead, you can experiment with benchmark dataset. [Google Drive link](https://drive.google.com/drive/folders/1six767uD8yfdgoGIYW86sJY-fmMdYq7e?usp=sharing).
 - Download the "data" folder.

```
wget -O data.zip 'https://drive.google.com/uc?export=download&id=1rZufm-XRq1Ig-56DejkQUX1si_WzCGBe&confirm=True' 
unzip data.zip
rm data.zip
```
 - Organize folders as follows.
```
.
├── configs
├── data
│   ├── ant_dir
│   ├── cheetah_dir
│   ├── cheetah_vel
│   └── ML1-pick-place-v2
├── envs
├── model_saved
├── src
└── ...
```

## Commands
- The main function allows for input to set up the benchmark environment and the settings in the configuration file. # default: AirCombatEnvironment, #choices: ['ant_dir, 'cheetah_dir', 'cheetah_vel'] 

### Training
```
# For training M-DT on benchmark environment
python main.py --task ant_dir --hyperparam_path ant-MDT.json
```
### Validation
```
# Test M-DT on benchmark environment
python main.py --base_path [path-to-checkpoint] --rollout True --task ant_dir --hyperparam_path ant-MDT.json
```

### Related Repositories
The original code framework is rendered from [Prompt-DT repo](https://github.com/mxu34/prompt-dt) (which is also rendered from [decision transformer](https://github.com/kzl/decision-transformer)).


