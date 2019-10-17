import cv2 
import numpy as np
import torch, sys
sys.path.insert(0, '/home/biondibazzanti/AutonomousDriving/Pytorch')
from utils.create_csv_file  import *
from utils.read_csv         import load_data_sequence_depth

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

    


