# Running the Train and Test Job


To make sure that a local training job can be run, remember to execute all work from AutonomousDriving/Pytorch/ folder, to have the correct paths where the results will be saved.

## Training job

Now there are three types of method a user can use:

- using the model that work with RGB single image
- using the model that work with RGB and Depth single image
- using the model that work with RGB sequence of image

The comand used for these three models is the same, change the main.py file of the model we will use:

```bash
python models/main.py   --train_path = path to the train csv
                        --valid_path = path to the validation csv
                        --epochs = define the number of epochs
                        --val_period = define how often do the validation test
                        --device = select the GPU to use
                        --model_type= choose from one of the tree model : single frame (single), multi frame (multi) and depth
```

The only things that change in using the depth model are the train and valid path where we have to specify the path with the keyword **depth** instead of the word **nodepth**. The weight create from the network will be saved in saved_models.

During the train is create the tensorboard file, saved in **tensorboard_runs**, wich the user can see with the tensorboard command:

```bash
tensorboard — logdir=tensorboard_runs/ — port 6006
```

## Testng  Job

Like is describe previously for the Training Job, for the Test we have to use the same command:

```bash
python models/main.py   --trest_path = path to the test csv
                        --weight_path = path to the path file create during the train job 
                        --device = select the GPU to use         
                             
```

At the end of the test job is created a csv file containing the prediction, the ground truth and the path to the image and save int test_results. This file will be used to [create the video](./files/video_creation.md)

