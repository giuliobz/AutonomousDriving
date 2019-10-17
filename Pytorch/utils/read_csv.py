import pandas as pd
import numpy as np
import matplotlib.image as mpimg
import cv2, math, tqdm, os, re


IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 90, 160, 3
DEPTH_HEIGHT, DEPTH_WIDTH = 720, 1280

def reshape_depth(img):
    SCALE_PERCENT = 12.5
    height = img.shape[0]
    width = img.shape[1]
    crop_img = img[60:-25, :]  # Ritaglio l'imagine
    dim = (IMAGE_WIDTH, IMAGE_HEIGHT)
    depth = cv2.resize(crop_img, dim, interpolation=cv2.INTER_AREA)
    depth[depth > 10000] = 10000
    depth = depth / 5000 - 1   
    depth = np.reshape(depth, (IMAGE_HEIGHT, IMAGE_WIDTH, 1))

    return depth

def load_image(image_file, data_dir, depth=False):


    img = cv2.imread(os.path.join(data_dir, image_file.strip()), -1)

    if depth == False: 

        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    elif depth == True:

        img = reshape_depth(img)

    return img

def convert_to_vector(string, video=False):

    tmp = list(string)
    i = 0
    final = []

    while (i != len(tmp)):
        if (tmp[i] == '['):
            pair = []
            number = ' '
            while (tmp[i] != ']'):
                if tmp[i] == '[' or tmp[i] == ' ':
                    i += 1
                elif tmp[i] == ',':

                    if video == True:
                        n = float(number)
                    elif video == False:
                        n = int(number)

                    number = ' '
                    pair.append(n)
                    i += 1
                else:
                    number += tmp[i]
                    i += 1
            if (number != ' '):

                if video == True:
                    pair.append(float(number))
                elif video == False:
                    pair.append(int(number))

                final.append(pair)
        i += 1
        
    final = np.asarray(final)


    return final

########################  CREATE DATA FOR SEQUENCE FOR SEQUENCE #############################

def load_data_sequence_nodepth(dir, len_sequence):
    """
        Load dataset and sequenze 
    """

    data_df = pd.read_csv(dir , error_bad_lines=False, names=["data_dir", "path", "future_point"])

    data_dir = data_df['data_dir'].values
    path = data_df['path'].values
    data_dimension = len(path)

    images_c = np.zeros([data_dimension, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
    tensor_list = np.zeros([data_dimension, len_sequence, 2])

    for i in tqdm.tqdm(range(data_dimension)):
        if (os.path.exists(path[i])):
            images_c[i] = load_image(image_file = path[i], data_dir = data_dir[i])
            tensor_list[i] = convert_to_vector(string = data_df["future_point"][i])

    return images_c, tensor_list, path

########################  CREATE DATA FOR VIDEO #############################

def load_data_for_video(dir, len_seq):

    data_df = pd.read_csv(dir , error_bad_lines=False, names=["image_path", "predicted", "real"])

    image_path = data_df['image_path'].values
    data_dimension = len(image_path)
    
    predicted = np.zeros([data_dimension, len_seq, 2])
    real = np.zeros([data_dimension, len_seq, 2])

    for i in range(data_dimension):
        if (os.path.exists(image_path[i])):
            predicted[i] = convert_to_vector(string = data_df['predicted'][i], video = True)
            real[i] = convert_to_vector(string = data_df["real"][i], video = True)


    return predicted, real

########################  CREATE DATA FOR SEQUENCE WITH DEPTH #############################

def load_data_sequence_depth(dir, len_sequence):

    data_df = pd.read_csv(dir , error_bad_lines=False, names=["data_dir", "path", "depth", "future_point"])

    data_dir = data_df['data_dir'].values
    path = data_df['path'].values
    depth = data_df['depth'].values
    data_dimension = len(path)

    images = np.zeros([data_dimension, IMAGE_HEIGHT, IMAGE_WIDTH, 4])
    tensor_list = np.zeros([data_dimension, len_sequence, 2])

    for i in tqdm.tqdm(range(data_dimension)):
        if (os.path.exists(path[i]) and os.path.exists(depth[i])):
            img = load_image(image_file = path[i], data_dir = data_dir[i])
            img = img / 127.5 - 1.0
            depth_image = load_image(image_file = depth[i], data_dir = data_dir[i], depth=True)
            images[i] = np.concatenate((img, depth_image), axis=2)
            tensor_list[i] = convert_to_vector(string = data_df["future_point"][i])

    return images, tensor_list, path

######################################################################################################################

def load_data_sequence_nodepth_2(dir, len_sequence, tipo):


    data_df = pd.read_csv(dir , error_bad_lines=False, names=["data_dir", "path", "future_point"])
    data_dir = '/home/biondibazzanti/Dataset/' + tipo +'/'
    path = data_df['path'].values
    data_dimension = len(path)

    sequence = {}

    for i, img in tqdm.tqdm(enumerate(path)):
        seq = re.search(tipo + '/(.+?)/', img)
        seq = seq.group(1)
        if seq in sequence.keys():
            sequence[seq].append(img)
            sequence[seq + '_cord'].append(convert_to_vector(string = data_df["future_point"][i]))
        else:
            sequence[seq] =  [img]
            sequence[seq + '_cord'] = [convert_to_vector(string = data_df["future_point"][i])]

    images = []
    coordinates = []
    paths = []
    for seq in tqdm.tqdm(sequence.keys()):
        if 'cord'not in seq:
            i = 4
            data_dir = data_dir + seq + '/'
            while(i < len(sequence[seq])):
                paths.append(sequence[seq][i])
                img = load_image(image_file = sequence[seq][i - 4], data_dir = data_dir) 
                tmp = np.concatenate((img, load_image(image_file = sequence[seq][i - 3], data_dir = data_dir)), axis=2)
                tmp = np.concatenate((tmp, load_image(image_file = sequence[seq][i - 2], data_dir = data_dir)), axis=2)
                tmp = np.concatenate((tmp, load_image(image_file = sequence[seq][i - 1], data_dir = data_dir)), axis=2)
                images.append(np.concatenate((tmp, load_image(image_file = sequence[seq][i], data_dir = data_dir)), axis=2))
                coordinates.append(sequence[seq + '_cord'][i])
                i += 1
                
    return np.asarray(images), np.asarray(coordinates), np.asarray(paths)


    

