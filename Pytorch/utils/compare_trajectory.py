import sys 
sys.path.insert(0, '/home/biondibazzanti/AutonomousDriving/Pytorch')
from utils.read_csv import convert_to_vector
from utils.trajectory_images import save_compare_video, save_video_with_trajectory
from utils.pixel_to_coordinate import pixel2coordinate
import pandas as pd
import numpy as np
import matplotlib.image as mpimg
import cv2, math, tqdm, os, re, json, argparse

def adjust_trajectory_small(trajectory):
    K = 0.125
    offset = 60
    tra = []
    for element in trajectory:
        tmp = []
        for coord in element:
            x = coord[0]
            y = coord[1]
            tmp.append([round(x/K), round((y/K) + offset)])
        tra.append(tmp)
    return np.asarray(tra)


def take_one_seq(images, seq):
    seq_images = []
    for element in images:
        if seq in element:
            seq_images.append(element)

    return np.asarray(seq_images)
'''
def compare_video(first_path, second_path, name):

    third_path = '/home/biondibazzanti/AutonomousDriving/saved_models/trajectories-depth-model.csv'
    print(first_path)
    print(second_path)
    if first_path == None or second_path == None:
        print("Error: specify the path of the trajectories' file")
        exit()
    
    if '.avi' not in name:
        print("Error: insert .avi in the video name")
        exit()

    if '.csv' in first_path and '.csv' in second_path:
        # caso in cui si voglia comparare due nostri video

        print('sono qui')

        data_first = pd.read_csv(first_path, error_bad_lines=False, names=["img", "predicted", "real"])
        data_second = pd.read_csv(first_path, error_bad_lines=False, names=["img", "predicted", "real"])
        data_third = pd.read_csv(third_path, error_bad_lines=False, names=["img", "predicted", "real"])

        predicted_first = data_first['predicted'].values
        predicted_second = data_second['predicted'].values
        predicted_third = data_third['predicted'].values
        r = data_first['real'].values
        images = data_first['img'].values

        f_predicted = np.zeros([len(images), 30, 2])
        s_predicted = np.zeros([len(images), 30, 2])
        t_predicted = np.zeros([len(images), 30, 2])
        real = np.zeros([len(images), 30, 2])

        for i in range(len(images)):
            
            f_predicted[i] = convert_to_vector(predicted_first[i], video=True)
            s_predicted[i] = convert_to_vector(predicted_second[i], video=True)
            t_predicted[i] = convert_to_vector(predicted_third[i], video=True)
            real[i] = convert_to_vector(r[i], video=True)

    elif '.csv' in first_path and '.json' in second_path:
        
        # Questo elif e il secondo sono il caso in cui si vogliono confrontare con gli altri

        data_first = pd.read_csv(first_path, error_bad_lines=False, names=["img", "predicted", "real"])

        predicted_first = data_first['predicted'].values
        r = data_first['real'].values
        images = data_first['img'].values

        seq = re.search("_(.+?)_", second_path)

        images = take_one_seq(images, seq.group(1))

        f_predicted = np.zeros([len(images), 30, 2])
        real = np.zeros([len(images), 30, 2])

        for i in range(len(images)):
            f_predicted[i] = convert_to_vector(predicted_first[i], video=True)
            real[i] = convert_to_vector(r[i], video=True)


        with open(second_path) as j:
            data = json.load(j)
            s_predicted = np.zeros([len(data), 30, 2])
            for i, elem in enumerate(data):
                s_predicted[i] = elem
        
        s_predicted = adjust_trajectory_small(s_predicted)
            

    elif '.json' in first_path and '.csv' in second_path:

        data_second = pd.read_csv(second_path, error_bad_lines=False, names=["img", "predicted", "real"])

        predicted_second = data_second['predicted'].values
        r = data_second['real'].values
        images = data_second['img'].values

        seq = re.search("_(.+?)_", second_path)

        images = take_one_seq(images, seq.group(1))

        s_predicted = np.zeros([len(images), 30, 2])
        real = np.zeros([len(images), 30, 2])

        for i in range(len(images)):

            s_predicted[i] = convert_to_vector(predicted_second[i], video=True)
            real[i] = convert_to_vector(r[i], video=True)


        with open(first_path) as j:
            data = json.load(j)
            f_predicted = np.zeros([len(data), 30, 2])
            for i, elem in enumerate(data):

                f_predicted[i] = elem
        
        f_predicted = adjust_trajectory_small(f_predicted)

    path = os.getcwd() # da terminale mi prende la cartella da cui si sta lavorando NB scrivere di lavorare sempre dalla cartella AutonomousDriving/Pytorch

    save_compare_video(path + '/results/video_compare/' + name, images, f_predicted, s_predicted, t_predicted, real)
'''

def compare_video(name):

    first_path = '/home/biondibazzanti/archieve/single-frame-input/weight_850_lr_0.01_20-buono/summarize/summarize_test_weight_26.pth.csv'
    second_path = '/home/biondibazzanti/AutonomousDriving/saved_models/trajectories-multiple-frame.csv'
    third_path = '/home/biondibazzanti/AutonomousDriving/saved_models/trajectories-depth-model.csv'

    data_first = pd.read_csv(first_path, error_bad_lines=False, names=["img", "predicted", "real"])
    data_second = pd.read_csv(second_path, error_bad_lines=False, names=["img", "predicted", "real"])
    data_third = pd.read_csv(third_path, error_bad_lines=False, names=["img", "predicted", "real"])

    predicted_first = data_first['predicted'].values
    predicted_second = data_second['predicted'].values
    predicted_third = data_third['predicted'].values
    r = data_first['real'].values
    images = data_second['img'].values


    f_predicted = np.zeros([len(images), 30, 2])
    s_predicted = np.zeros([len(images), 30, 2])
    t_predicted = np.zeros([len(images), 30, 2])
    real = np.zeros([len(images), 30, 2])

    i = 4
    while( i < len(images)):     
        f_predicted[i] = convert_to_vector(predicted_first[i], video=True)
        s_predicted[i] = convert_to_vector(predicted_second[i], video=True)
        t_predicted[i] = convert_to_vector(predicted_third[i], video=True)
        real[i] = convert_to_vector(r[i], video=True)
        i+=1


    save_compare_video(name, images, f_predicted, s_predicted, t_predicted, real)


def compute_distances(first_path):

    data_df = pd.read_csv(first_path, error_bad_lines=False, names=["img", "predicted", "real"])

    img = data_df['img'].values
    predicted = data_df['predicted'].values
    real = data_df['real'].values

    output = np.zeros([len(img), 7])
    
    for i in range(len(img)):
        depth_path = img[i].replace("Dataset_real","Dataset")
        depth_path = depth_path.replace("left", "depth")
        pred = convert_to_vector(predicted[i], video=True)
        rel = convert_to_vector(real[i], video=True)
        output[i] = pixel2coordinate(pred, rel, depth_path)
    
    result = np.mean(output, axis=0)

    print("Distances for {} is : {:.2f} cm".format(1,  result[0]/10))
    print("Distances for {} is : {:.2f} cm".format(5,  result[1]/10))
    print("Distances for {} is : {:.2f} cm".format(10, result[2]/10))
    print("Distances for {} is : {:.2f} cm".format(15, result[3]/10))
    print("Distances for {} is : {:.2f} cm".format(20, result[4]/10))
    print("Distances for {} is : {:.2f} cm".format(25, result[5]/10))
    print("Distances for {} is : {:.2f} cm".format(30, result[6]/10))



##################################################################################################################################################

def main():

    parser = argparse.ArgumentParser(description="Compare prediction in video")
    parser.add_argument("--first_path",     dest="first",               default=None,                            help="path of the first file with prediction and image")
    parser.add_argument("--second_path",    dest="second",              default=None,                            help="path of the second file with prediction and image")
    parser.add_argument("--video_name",     dest='name',                default='compare_trajectory.avi',        help="name of the video, put .avi in the name")
    parser.add_argument("--video",          dest='video',               default='True',                          help='decide if you make video or not')

    args = parser.parse_args()
    first_path = args.first
    second_path = args.second
    name = args.name
    video = args.video

    if video == 'True':
        compare_video(name)

    else:
        compute_distances(first_path)


if __name__ == '__main__':
    main()