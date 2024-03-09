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

 - Install environments and dependencies with the following commands:
```
# install dependencies
pip install -r requirements.txt

# install environments
./install_envs.sh
```

## Data
 - We share example datasets via this [Google Drive link](https://drive.google.com/drive/folders/1six767uD8yfdgoGIYW86sJY-fmMdYq7e?usp=sharing).
 - Download the "data" folder.

```
wget -O data.zip 'https://drive.google.com/uc?export=download&id=1rZufm-XRq1Ig-56DejkQUX1si_WzCGBe&confirm=True' 
unzip data.zip
rm data.zip
```
 - Organize folders as follows.

## Commands

### Training

### Validation

### Test

### Related Repositories
The original code framework is rendered from [Prompt-DT repo](https://github.com/mxu34/prompt-dt) (which is also rendered from [decision transformer](https://github.com/kzl/decision-transformer)).


