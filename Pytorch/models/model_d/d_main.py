import sys 
sys.path.insert(0, '/home/biondibazzanti/AutonomousDriving/Pytorch')
from utils.create_csv_file      import *
from utils.read_csv             import load_data_sequence_depth
import torch
import torch.nn                 as nn
from models.model_d.d_model            import NET, Weighted_MSE
from models.model_d.d_train              import train
from models.model_d.d_test               import test
from torch.utils.data           import TensorDataset, DataLoader
from torch.autograd             import Variable
import numpy                    as np
import argparse, os
from datetime                   import datetime

########################################################################################################################
# INIT NETWORK AND LOAD VALIDATION AND TRAIN SET
########################################################################################################################

def Initialize_Model(batch_size, lr, len_seq, hidden_dimension, loss, num_layers, opt):
    model = NET(len_seq=len_seq, batch_size=batch_size, hidden_dimension=hidden_dimension, num_layers=num_layers)
    if loss == 'MSE':   
        criterion = nn.MSELoss()
        #criterion = nn.L1Loss()
    elif loss == 'weighted':
        criterion = Weighted_MSE()

    if opt == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))
    elif opt == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=float(lr), momentum=0.9)
    elif opt == 'RMS':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=float(lr), alpha=0.90, momentum=0.0)

    return model, criterion, optimizer


def load_Dataset(len_sequence, train_path=None, valid_path=None, test_path=None):
    if test_path == None:
        print("Load Train Set")
        imm_train, train_coordinates,  _ = load_data_sequence_depth(train_path, len_sequence)
        print()
        print("Load Validation Set")
        imm_valid, valid_coordinates,  _ = load_data_sequence_depth(valid_path, len_sequence)
 
        train_images = np.moveaxis(imm_train, -1, 1)
        valid_images = np.moveaxis(imm_valid, -1, 1)

        return train_images, valid_images, train_coordinates, valid_coordinates
    else:
        print("Load Test Set")
        imm_test, test_coordinates, image_path = load_data_sequence_depth(test_path, len_sequence)
        test_images = np.moveaxis(imm_test, -1, 1)

        return test_images, test_coordinates, image_path


########################################################################################################################
# STARTING THE RAINING OF THE NET
########################################################################################################################


def main():

    parser = argparse.ArgumentParser(description="Train the CNN and LSTM model")
    parser.add_argument("--train_path",     dest="train",               default=None,                            help="path of the train csv file")
    parser.add_argument("--valid_path",     dest="valid",               default=None,                            help="path of the validation csv file")
    parser.add_argument("--weight_path",    dest="weight",              default='/home/biondibazzanti/weights/', help="path to save the weight")
    parser.add_argument("--test_path",      dest="test",                default=None,                            help="path of the test csv file")
    parser.add_argument("--model_path",     dest="model",               default=None,                            help="path of the model weight")
    parser.add_argument("--epochs",         dest="epochs",              default=1000,                            help="number of epochs")
    parser.add_argument("--val_period",     dest="period",              default=1,                               help="choose when use the validation")
    parser.add_argument("--lr",             dest="learn",               default=0.01,                            help="decide the learning rate")
    parser.add_argument("--batch_size",     dest="batch",               default=20,                              help="decide the batch size")
    parser.add_argument("--device",         dest="device",              default=1,                               help="choose GPU")
    parser.add_argument("--epsilon",        dest="epsilon",             default=100,                             help="epsilon value for accuracy of model")
    parser.add_argument("--len_seq",        dest="sequence",            default=30,                              help="define the len of the sequence")
    parser.add_argument("--hidden_dim",     dest="hidden_dimension",    default=128,                             help="define the dimension of states in LSTM")
    parser.add_argument("--loss",           dest="loss",                default='MSE',                           help="MSE or weighted for weightedMSE")
    parser.add_argument("--opt",            dest="optimizer",           default='SGD',                           help="choose one Adam, SGD, RMSprop")
    parser.add_argument("--num_layers",     dest="num_layers",          default=2,                               help="layer's number of the LSTM")
    parser.add_argument("--dec_per",        dest="dec_period",          default=20,                              help="decrement period of lr")
    parser.add_argument("--note",           dest="note",                default=None,                            help="notebook")
 
    args                = parser.parse_args()

    len_seq             = int(args.sequence)
    batch_size          = int(args.batch)
    learning_rate       = float(args.learn)
    dev                 = str(args.device)
    hidden_dimension    = int(args.hidden_dimension)
    loss                = args.loss
    num_layers          = int(args.num_layers)
    opt                 = args.optimizer
    dec_period          = int(args.dec_period)

    if (args.train == None and args.test == None):
        print("you have to decide : do train or test")
        exit()

    ####################################################################################################################
    # TRAIN PHASE
    ####################################################################################################################
    if args.test == None:

        if (args.valid == None or args.weight == None):
            print("please insert valid  or weight path ")
            exit()
        else:
            
            epoch = int(args.epochs)
            val_period = int(args.period)

            # current date and time
            now = datetime.now()
            timestamp = datetime.timestamp(now)

            save_weight_path = args.weight + 'weight_' + str(epoch) + '_lenseq_' + str(args.sequence) + '_depth_' + str(timestamp) 
            
            clip = 5
            print()
            print("SUMMARIZE : ")
            print()
            print("train data path: {}"     .format(args.train))
            print("valid data path: {}"     .format(args.valid))
            print("weight save path: {}"    .format(save_weight_path))
            print("epoch: {}"               .format(epoch))
            print("validation period: {}"   .format(val_period))
            print("batch size: {}"          .format(batch_size))
            print("learning rate: {}"       .format(learning_rate))
            print("GPU device: {}"          .format(dev))
            print("len_seq: {}"             .format(len_seq))
            print("hidden_dimension: {}"    .format(hidden_dimension))
            print("Loss Function: {}"       .format(loss))
            print("Optimizer: {}"           .format(opt))
            print("Decrement period: {}"    .format(dec_period))
            print("num_layers: {}"          .format(num_layers))
            print("To see tensorboardX use the following comand from the remote server")
            print("---------> tensorboard --logdir=/home/biondibazzanti/tensorboard_runs/weight_" + '_lenseq_' + str(args.sequence) + '_' + str(timestamp) + " --host 150.217.35.230 --port 16006")
            print()
            
            if not os.path.exists(save_weight_path):
                os.mkdir(save_weight_path)
            
            if not os.path.exists(save_weight_path + "/summarize"):
                os.mkdir(save_weight_path + "/summarize")

            if not os.path.exists(save_weight_path + "/video"):
                os.mkdir(save_weight_path + "/video")

            with open(save_weight_path + "/SUMMARIZE.txt", "w+") as f:
                
                f.write("SUMMARIZE \n\r")
                f.write("\n\r")
                f.write("train data path: {} \n\r"  .format(args.train))
                f.write("valid data path: {} \n\r"  .format(args.valid))
                f.write("weight save path: {} \n\r" .format(save_weight_path))
                f.write("epoch: {} \n\r"            .format(epoch))
                f.write("validation period: {} \n\r".format(val_period))
                f.write("batch size: {} \n\r"       .format(batch_size))
                f.write("learning rate: {} \n\r"    .format(learning_rate))
                f.write("GPU device: {} \n\r"       .format(dev))
                f.write("len_seq: {} \n\r"          .format(len_seq))
                f.write("hidden_dimension: {} \n\r" .format(hidden_dimension))
                f.write("Loss Function: {} \n\r"    .format(loss))
                f.write("Optimizer: {} \n\r"        .format(opt))
                f.write("Decrement period: {} \n\r" .format(dec_period))
                f.write("num_layers: {} \n\r"       .format(num_layers))
                f.write("To see tensorboardX use the following comand from the remote server \n\r")
                f.write("---------> tensorboard --logdir=/home/biondibazzanti/tensorboard_runs/weight_" + '_lenseq_' + str(args.sequence) + '_' + str(timestamp) + " --host 150.217.35.230 --port 16006 \n\r")
                f.write("To be notice: {}"          .format(args.note))

            if not os.path.exists(args.train):
                print('Creating CSV for training set with sequence with dimension {}'.format(len_seq))
                type = 'train'
                path = '/home/biondibazzanti/Dataset/'  + type + "/"
                crete_csv_sequence(path, type, len_seq)

            if not os.path.exists(args.valid):
                print('Creating CSV for validation set with sequence with dimension {}'.format(len_seq))
                type = 'validation'
                path = '/home/biondibazzanti/Dataset/' + type + "/"
                crete_csv_sequence(path, type, len_seq)

            train_images, valid_images, train_coordinates, valid_coordinates = load_Dataset(len_seq, args.train, args.valid)

            model, criterion, optimizer = Initialize_Model(batch_size, learning_rate, len_seq, hidden_dimension, loss, num_layers, opt)

            #print(model)
            train_data  = TensorDataset(torch.from_numpy(train_images), torch.from_numpy(train_coordinates))
            val_data    = TensorDataset(torch.from_numpy(valid_images), torch.from_numpy(valid_coordinates))

            train_loader    = DataLoader(train_data,    shuffle=True,   batch_size=batch_size, drop_last=True)
            val_loader      = DataLoader(val_data,      shuffle=False,  batch_size=batch_size, drop_last=True)

            train(model, criterion, optimizer, train_loader, val_loader, batch_size, epoch, val_period, clip, save_weight_path, dev, loss, learning_rate, dec_period, opt)


    ####################################################################################################################
    # TEST PHASE
    ####################################################################################################################

    if args.train == None:
        if (args.model == None):
            print("please insert the path to load the model for the test")
            exit()
        else:

            epsilon = float(args.epsilon)

            print()
            print("SUMMARIZE : ")
            print()
            print("test data path: {}"      .format(args.test))
            print("model path: {}"          .format(args.model))
            print("batch size: {}"          .format(batch_size))
            print("hidden dimension: {}"    .format(hidden_dimension))
            print("GPU device: {}"          .format(dev))
            print("len_seq: {}"             .format(len_seq))
            print("Loss Function: {}"       .format(loss))
            print()
            
            if not os.path.exists(args.test):
                print('Creating CSV for test set with sequence with dimension {}'.format(len_seq))
                type = 'test'
                path = '/home/biondibazzanti/Dataset/' + type + "/"
                crete_csv_sequence(path, type, len_seq)

            test_images, test_coordinates, image_path = load_Dataset(test_path=args.test, len_sequence=len_seq)

            model, criterion, _ = Initialize_Model(batch_size, learning_rate, len_seq, hidden_dimension, loss, num_layers=num_layers, opt=opt)

            print(model)
            
            test_data = TensorDataset(torch.from_numpy(test_images), torch.from_numpy(test_coordinates))

            test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size, drop_last=True)

            test(model, criterion, args.model, batch_size, test_loader, image_path, epsilon, dev, len_seq, loss)


if __name__ == '__main__':
    main()
