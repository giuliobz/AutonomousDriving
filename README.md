# Autonomous Driving

The aim of this project is to predict the trajectory from a series of frames to emulate the autonomous drive in a radio controlled machine.

## Table of contents

Run:
    - [Preprocessing](datasets/image_dataset/README.md)
    - [Create csv files](datasets/csv_dataset/README.md)
    - [Training and Testing](models/README.md)
    - [Video creation](videos/README.md)

## Dependacies

- PyTorch
- numpy
- opencv
- mathplotlib
- pil

For detailed steps to install PyTorch, follow the [PyTorch installation instruction](https://pytorch.org/get-started/locally/). A typical user can install PyTorch using the following commands:

```bash
# Create virtual environment
conda create -n pytorch python=3.7

# Activate virtual environment
conda activate pytorch

# Install PyTorch
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch

```

The other packet can be installed using pip:

```bash

conda activate pytorch
pip install numpy
pip install mathplotlib
pip install Pillow==2.2.2
pip install opencv-python
pip install tqtm

```
    
