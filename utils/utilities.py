import torch, math, csv, cv2
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from models.model import NET
from utils.read_csv import load_data_multiframe, load_data_depth, load_data_singleframe
from torch.autograd import Variable
from torchvision.utils import make_grid
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

##############################################################################################################

def export_plot_from_tensorboard(event_path, save_path):
    event_acc = EventAccumulator(event_path)
    event_acc.Reload()
    for element in event_acc.Tags()['scalars']:
        _, step_nums, vals = zip(*event_acc.Scalars(element))
        plt.plot(list(step_nums), list(vals))
        tmp = element.split("/")
        plt.savefig(save_path + "/" + tmp[1] + '.png')
        plt.clf()
    
##############################################################################################################

def depth(y, x, depth):
    
    val = depth[ y-1 : y +2, x-1 : x+2]
    var = np.mean(val)

    return var

def pixel2coordinate(output, label, depth_path):
    SCALE_PERCENT = 12.5
    K = float(SCALE_PERCENT / 100)
    offset = 60
    fx = 699.224
    fy = 699.224
    cx = 652.652
    cy = 364.663

    img_depth = cv2.imread(depth_path, -1)
    img_depth[img_depth > 10000] = 100000
    distances = []
    out = []

    index = [0, 4, 9, 14, 19, 24, 29]

    new_out = torch.zeros([len(output), 2], requires_grad=True, dtype=torch.double)
    new_labels = torch.zeros([len(label), 2], requires_grad=True, dtype=torch.double)
    for i in range(len(output)):
        x_o = int(output[i][0].item())
        y_o = int(output[i][1].item())
        z_o = int(depth(y_o, x_o, img_depth))

        x_l = int(label[i][0].item())
        y_l = int(label[i][1].item())
        z_l = int(img_depth[y_l][x_l])
                
        x_o_d = int((x_o - cx) * z_o / fx)
        y_o_d = int((y_o - cy) * z_o / fy)

        new_out[i][0] = x_o_d
        new_out[i][1] = y_o_d
        
        x_l_d = int((x_l - cx) * z_l / fx)
        y_l_d = int((y_l - cy) * z_l / fy)
        
        new_labels[i][0] = x_l_d
        new_labels[i][1] = y_l_d

        distances.append(np.sqrt( (x_o_d - x_l_d)**2 + (y_o_d - y_l_d)**2 ))
    
    
    for element in index:
        out.append(distances[element])

    return np.asarray(out)

##############################################################################################################

def initialize_model(model_type, cfg, mode):

    if mode == 'train':
        batch_size = cfg.TRAIN.BATCH_SIZE
        len_seq = cfg.TRAIN.LEN_SEQUENCES
    elif mode == 'test':
        batch_size = cfg.TEST.BATCH_SIZE
        len_seq = cfg.TEST.LEN_SEQUENCES
    else:
        print('Error occurrent')
        exit()
    
    model = NET(len_seq=len_seq, batch_size=batch_size, hidden_dimension=cfg.DIMENSION[model_type], num_layers=cfg.LAYERS, in_channels=cfg.IN_CHANNELS[model_type])
 
    criterion = nn.MSELoss()

    if mode == 'train':
        
        if cfg.TRAIN.OPTIMIZER == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.LEARNING_RATE)
        elif cfg.TRAIN.OPTIMIZER == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), lr=cfg.TRAIN.LEARNING_RATE, momentum=cfg.TRAIN.MOMENTUM)
        elif cfg.TRAIN.OPTIMIZER == 'RMS':
            optimizer = torch.optim.RMSprop(model.parameters(), lr=cfg.TRAIN.LEARNING_RATE, alpha=cfg.TRAIN.ALPHA, momentum=cfg.TRAIN.MOMENTUM)
    
        return model, criterion, optimizer

    else:
        
        return model, criterion

def load_dataset(len_sequence, model_type, train_path=None, valid_path=None, test_path=None):
    
    if test_path == None:
        print("Load Train Set and vaidation set")
        if model_type == 'multi':
            imm_train, train_coordinates,  _ = load_data_multiframe(train_path, len_sequence)
            imm_valid, valid_coordinates,  _ = load_data_multiframe(valid_path, len_sequence)
        elif model_type == 'depth':
            imm_train, train_coordinates,  _ = load_data_depth(train_path, len_sequence)
            imm_valid, valid_coordinates,  _ = load_data_depth(valid_path, len_sequence)
        elif model_type == 'single':
            imm_train, train_coordinates,  _ = load_data_singleframe(train_path, len_sequence)
            imm_valid, valid_coordinates,  _ = load_data_singleframe(valid_path, len_sequence)
 
        train_images = np.moveaxis(imm_train, -1, 1)
        valid_images = np.moveaxis(imm_valid, -1, 1)

        return train_images, valid_images, train_coordinates, valid_coordinates

    else:
        
        print("Load Test Set")

        if model_type == 'multi':
            imm_test, test_coordinates, image_path = load_data_multiframe(test_path, len_sequence)
        elif model_type == 'depth':
            imm_test, test_coordinates, image_path = load_data_depth(test_path, len_sequence)
        elif model_type == 'single':
            imm_test, test_coordinates, image_path = load_data_singleframe(test_path, len_sequence)

        test_images = np.moveaxis(imm_test, -1, 1)

        return test_images, test_coordinates, image_path

##############################################################################################################