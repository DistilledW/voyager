# webGS
A cloud and personal computer co-rendering methods 

## [Project page](https://github.com/DistilledW/webGS) | [Paper](https://github.com/DistilledW/webGS) 

This repository contains three parts including train, cloud and client. 

## Setup
### Prerequisite
### Python Evironment for optimization
## Running the method


# Getting Started
This repository uses ubuntu22.04, cuda12.4 

```sh
conda create -n h3dgs python=3.10
conda activate h3dgs
pip install -r requirements.txt
cd submodules
pip install simple-knn/
pip install fast_hier/
pip install flashTreeTraversal/
pip install flashLocal/ 
```


```sh
conda activate h3dgs
cd /path/to/voyager/cloud 
bash run.sh 

conda activate h3dgs
cd /path/to/voyager/client
bash run.sh
```