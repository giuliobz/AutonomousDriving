import cv2, os, math, tqdm
import pandas as pd
import numpy as np
from utils.read_csv import load_data_for_video


def save_video(video_name, csv_path, len_seq, model_type):
    
    data_df = pd.read_csv(csv_path, error_bad_lines=False, names=["image_path", "predicted", "real"]) 

    image_path = data_df['image_path'].values
    predicted, real = load_data_for_video(csv_path, len_seq)
    video_img = []
    
    for i in tqdm.tqdm(range(len(image_path))):
        
        img = cv2.imread(image_path[i].strip(), -1)
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        height, width, _ = img.shape
        size = (width,height)

        x, y = 600, 450  # (x, y) coordinate punto corrente
        cv2.circle(img, (x, y), 3, (0, 255, 0), 1)
        
        for j, item in enumerate(predicted[i]):
            x = math.ceil(item[0])
            y = math.ceil(item[1])
            img[y - 1:y + 1, x - 1:x + 1] = [0, 0, 255]
            if (j < (len(predicted[i]) - 1)):
                next_x = math.ceil(predicted[i][j + 1][0])
                next_y = math.ceil(predicted[i][j + 1][1])
                cv2.line(img, (x, y), (next_x, next_y), (0, 0, 255), 2)

        for j, item in enumerate(real[i]):
            x = int(item[0])
            y = int(item[1])
            img[y - 1:y + 1, x - 1:x + 1] = [255, 255, 255]
            if (j < (len(real[i]) - 1)):
                next_x = int(real[i][j + 1][0])
                next_y = int(real[i][j + 1][1])
                cv2.line(img, (x, y), (next_x, next_y), (255, 255, 255), 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, 'CNN+LSTM', (100, 100), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(img, 'Ground Truth', (100, 150), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, 'Starting Point', (100, 200), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        video_img.append(img)


    writer = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'DIVX'), 10, size)
    for i in tqdm.tqdm(range(len(video_img))):
        writer.write(video_img[i])
    writer.release()   
    print()
    print("Create video and store it in " + video_name)
    

