import torch, math, csv
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torchvision.utils import make_grid
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


##########################################################################################################

class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


##########################################################################################################

def create_weight_matrix(batch_size, len_seq):
    weighted_matrix = np.zeros((batch_size, len_seq, 2))

    for i in range(batch_size):
        for j in range(len_seq):
            if j < math.ceil(len_seq/3):
                weighted_matrix[i][j] = 0.2
            elif j < 2*math.ceil(len_seq/3):
                weighted_matrix[i][j] = 0.3
            else:
                weighted_matrix[i][j] = 0.5
                
    return Variable(torch.from_numpy(weighted_matrix))


###########################################################################################################


def save_plot_loss(dictionary, save_path):
    fig, ax = plt.subplots()
    for key in dictionary.keys():
        loss = sum(dictionary[key])/len(dictionary[key])
        ax.plot(key, loss)

    ax.set(xlabel='epochs', ylabel='losses')  
    ax.grid()
    fig.savefig(save_path)


##############################################################################################################


def new_evaluation(out, label, depth, num_correct, epsilon):
    SCALE_PERCENT = 12.5
    K = float(SCALE_PERCENT / 100)
    offset = 60
    fx = 699.224
    fy = 699.224
    cx = 652.652
    cy = 364.663
    new_x_present = []
    new_y_present = []
    x_present = int(600 * K)
    y_present = int((450 - offset) * K)
    for images in depth:
        z_present = images[x_present][y_present].item()
        new_x_present.append(float((x_present - cx) * z_present / fx))
        new_y_present.append(float((y_present - cy) * z_present / fy))
    
    for i in range(len(new_x_present)):
        num_seq = 0
        out_dist = 0
        label_dist = 0
        for j in range(len(out[i])):
            tmp1 = (out[i][j][0] - new_x_present[i])**2
            tmp2 = (out[i][j][1] - new_y_present[i])**2
            out_dist = math.sqrt(tmp1 + tmp2)

            tmp1 = (label[i][j][0] - new_x_present[i])**2
            tmp2 = (label[i][j][1] - new_y_present[i])**2
            label_dist = math.sqrt(tmp1 + tmp2)

            if abs(out_dist - label_dist) <= epsilon:
                num_seq += 1
        
        if num_seq >= math.ceil(len(out[i]) * 0.8):
            num_correct +=1
    
    return num_correct  


#############################################################################################################

def evaluate_result(out, label, num_correct, epsilon):
    
    for i in range(len(out)):
        x_diff = out[i][0] - label[i][0]
        if x_diff < epsilon:
            y_diff = out[i][1] - label[i][0]                
            if y_diff < epsilon:
                num_correct += 1

    return num_correct

##############################################################################################################

 
def adapt_learning_rate(optimizer, lr, epoch, limit, dec_period, lr_decay=0.1):

    if dec_period > 1:
        #lr = lr * (lr_decay ** (epoch // 10))
        lr = lr * lr_decay
    else:
        lr = lr / (1 + 0.05 * epoch)

    if lr <= limit:
        return optimizer, limit
    else:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return optimizer, limit

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
    
################################################################################################################

def evaluate(img_loader, model, batch_size, criterion, losses, device, l, 
                writer=None, paths=None, csv_file=None, weighted_matrix=None, epoch=None, avg_meters=None, mode='val'):

    i = epoch
    model.eval()
    current_path = 0

    #hidden = model.init_hidden(batch_size, device)
    with torch.no_grad():

        for image, labels in img_loader:

            hidden = model.init_hidden(batch_size, device)
            hidden = tuple([each.data for each in hidden])
            inputs = image.to(device)

            out, hidden = model(inputs.float(), hidden)
                                      
            if l == 'weighted':
                loss = criterion(out, labels.float(), weighted_matrix.float())
            else:
                loss = criterion(out, labels.float())
                            
            if mode == 'val':

                if i in losses:
                    losses[i].append(loss.item())
                else:
                    losses[i] = [loss.item()]
                
                avg_meters.update(loss.item(), batch_size)
                
            elif mode == 'test':

                losses.append(loss.item())
                with open(csv_file, 'a') as result_file:
                    filewriter = csv.writer(result_file)

                    for i in range(len(out)):
                        predicted = out[i].detach().numpy()
                        real = labels[i].detach().numpy()
                        path = paths[current_path].replace("Dataset","Dataset_real")
                        lines = [path, predicted.tolist(), real.tolist()]
                        filewriter.writerow(lines)
                        current_path += 1



################################################################################################################