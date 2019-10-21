## Create csv files

In order to use the entire model is necessary create the csv file with the following comand:

```bash
python tools/create_csv_file.py     --input_path=input direcotry of the image
                                    --depth= True if you want to create csv for depth model
                                    --type=choose the dataset: train, validation, test
                                    --len_seq = lenght of the sequence
```

This script create a csv file and store it in AutonomousDriving/csv_dataset in one of the tree inside folder depending on the type and on the usage of depth or not.
