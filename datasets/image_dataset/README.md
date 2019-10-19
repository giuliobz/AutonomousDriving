## Recommended Directory Structure for Training and Evaluation

Be sure you have the same dataset structure for both original and processed data. These structures will be used in the testing phase.
Another recommendation : call **Dataset** the processed one and **Dataset_real** the original one and put them in the same folder like this:

```bash

image_dataset
│
├── Dataset                                    
│   ├── train  
│       ├── seq1
│           ├── left   
│           ├── depth 
│   ├── valid 
│       ├── seq2
│           ├── left
│           ├── depth 
│   ├── test
│       ├── seq3
│           ├── left
│           ├── depth 
│
├── Dataset_real                                    
│   ├── train  
│       ├── seq1
│           ├── left  
│           ├── depth  
└── ...

```

Inside the left and depth folder is recommended to have:

- left: is the folder contain the frames
- trajectories.json: a json file where tehre are all the trajectory relative to the frames contained in the left folder

## Pre processing

The Dataset used for the experiment is preprocessed and saved with the structure described above using the following command:

```bash

python tools/preprocess.py  --type= specify the type of the image folder: train - test - validation

```




