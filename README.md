# <p align=center> Enhancing Fog-Free Infrared and Visible Image Fusion: An Aggregation-to-Decoupling Approach</p>

## Continuously updating!  
Note: Still under review, this code repository is not yet fully complete.

## 1.Quick Start

### Install
This repository is built in PyTorch 1.12.0 and Python 3.8
Follow these intructions

[//]: # (1. Clone our repository)

[//]: # (```)

[//]: # (git clone  https://github.com/zhoushen1/MEASNet)

[//]: # (cd MEASNet)

[//]: # (```)
1.Create conda environment
The Conda environment used can be recreated using the ```env.yml``` file
```
conda env create -f env.yml
```
### Datasets
FOGIV: [FOGIV](https://pan.baidu.com/s/1lER7xj6Lzw64E0AuvbkVAA?pwd=ajjj)

M3FD: [M3FD](https://pan.baidu.com/s/1m4DLqnywOoWFuRQbJdQL3Q?pwd=375k)

MSRS: [MSRS](https://github.com/Linfeng-Tang/MSRS)


The training data should be placed in ``` dataset/{dataset_name}```.

The testing data should be placed in ```dataset/{dataset_name}/test/```. 

## 2.Training
After preparing the training data in ```dataset/{dataset_name}``` directory, use 
```
python train.py
```
## 3.Testing

After preparing the testing data in ```dataset/{dataset_name}/test/``` directory, use
```
python test.py
```

## Results
You can download visual results from (Link：https://pan.baidu.com/s/1GHmqP9himlZ_yo9h2AYCCQ?pwd=o2kp code：o2kp)
# Contact:
    Don't hesitate to contact me if you meet any problems when using this code.
    Xianghe Bi
    Email: xianghe001919@163.com

## If this work is helpful to you, please cite it as：
```bibtex
This code is associated with the manuscript submitted to The Visual Computer.
We kindly request that researchers cite this work if used.
