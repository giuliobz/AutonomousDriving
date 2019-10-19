from tools.create_csv_file  import crete_csv_sequence
from utils.read_csv         import load_data_multiframe_nd
from utils.utilities        import initialize_model, load_dataset
import torch, argparse, os
from models.train           import train
from models.test            import test
from config.config          import cfg
from torch.utils.data       import TensorDataset, DataLoader
from torch.autograd         import Variable
import numpy                as np
from datetime               import datetime



########################################################################################################################
# STARTING THE RAINING OF THE NET
########################################################################################################################


def main():

    parser = argparse.ArgumentParser(description="Train the CNN and LSTM model")
    parser.add_argument("--train_path",     dest="train",               default=None,                            help="path of the train csv file")
    parser.add_argument("--valid_path",     dest="valid",               default=None,                            help="path of the validation csv file")
    parser.add_argument("--test_path",      dest="test",                default=None,                            help="path of the test csv file")
    parser.add_argument("--model_path",     dest="model",               default=None,                            help="path of the model weight")
    parser.add_argument("--epochs",         dest="epochs",              default=200,                             help="number of epochs")
    parser.add_argument("--val_period",     dest="period",              default=1,                               help="choose when use the validation")
    parser.add_argument("--device",         dest="device",              default=0,                               help="choose GPU")
    parser.add_argument("--moel_type",           dest="type",                default='single',                        help="define the model to use: sigle-frame, multi-frame or depth")
 
    args                = parser.parse_args()


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

            # current date and time
            now = datetime.now()
            timestamp = datetime.timestamp(now)

            save_weight_path = cfg.SAVE_WEIGHT_PATH[args.type] + 'weight_' + args.epochs + '_lenseq_' + args.sequence + '_' + str(timestamp)
            
            print()
            print("SUMMARIZE : ")
            print()
            print("train data path: {}"     .format(args.train))
            print("valid data path: {}"     .format(args.valid))
            print("weight save path: {}"    .format(save_weight_path))
            print("epoch: {}"               .format(args.epochs))
            print("validation period: {}"   .format(args.period))
            print("batch size: {}"          .format(cfg.TRAIN.BATCH_SIZE))
            print("learning rate: {}"       .format(cfg.TRAIN.LEARNING_RATE))
            print("GPU device: {}"          .format(args.device))
            print("len_seq: {}"             .format(cfg.TRAIN.LEN_SEQUENCES))
            print("hidden_dimension: {}"    .format(cfg.DIMENTION[args.type]))
            print("Loss Function: {}"       .format(cfg.TRAIN.LOSS))
            print("Optimizer: {}"           .format(cfg.TRAIN.OPTIMIZER))
            print("Decrement period: {}"    .format(cfg.TRAIN.DEC_PERIOD))
            print("num_layers: {}"          .format(cfg.TRAIN.LAYERS))
            print("you are working with {} model"          .format(args.type))
            print("To use tensorboardX log this --logdir : " + cfg.TENSORBOARD_PATH + "/weight_" + '_lenseq_' + str(args.sequence) + '_' + str(timestamp))
            print()
            
            if not os.path.exists(save_weight_path):
                os.mkdir(save_weight_path)

            train_images, valid_images, train_coordinates, valid_coordinates = load_dataset(len_sequence=cfg.TRAIN.LEN_SEQUENCES, model_type=args.type, train_path=args.train, valid_path=args.valid)

            model, criterion, optimizer = initialize_model(model_type=args.type, cfg=cfg, mode='train')

            train_data  = TensorDataset(torch.from_numpy(train_images), torch.from_numpy(train_coordinates))
            val_data    = TensorDataset(torch.from_numpy(valid_images), torch.from_numpy(valid_coordinates))

            train_loader    = DataLoader(train_data,    shuffle=cfg.TRAIN.SHUFFLE_T,   batch_size=cfg.TRAIN.BATCH_SIZE, drop_last=True)
            val_loader      = DataLoader(val_data,      shuffle=cfg.TRAIN.SHUFFLE_V,   batch_size=cfg.TRAIN.BATCH_SIZE, drop_last=True)

            train(model=model, criterion=criterion, optimizer=optimizer, train_loader=train_loader, val_loader=val_loader, epochs=args.epochs, val_period=args.period, save_weights=save_weight_path, dev=args.device, cfg=cfg)


    ####################################################################################################################
    # TEST PHASE
    ####################################################################################################################

    if args.train == None:
        if (args.model == None):
            print("please insert the path to load the model for the test")
            exit()
        else:

            print()
            print("SUMMARIZE : ")
            print()
            print("test data path: {}"      .format(args.test))
            print("model path: {}"          .format(args.model))
            print("batch size: {}"          .format(cfg.TRAIN.BATCH_SIZE))
            print("hidden dimension: {}"    .format(cfg.DIMENTION[args.type]))
            print("GPU device: {}"          .format(args.device))
            print("len_seq: {}"             .format(cfg.LEN_SEQUENCES))
            print("Loss Function: {}"       .format(cfg.TRAIN.LOSS))
            print("you are working with {} model"          .format(args.type))
            print()

            test_images, test_coordinates, image_path = load_dataset(test_path=args.test, len_sequence=cfg.LEN_SEQUENCES)

            model, criterion, _ = initialize_model(model_type=args.type, cfg=cfg, mode='test')
            
            test_data = TensorDataset(torch.from_numpy(test_images), torch.from_numpy(test_coordinates))

            test_loader = DataLoader(test_data, shuffle=cfg.TEST.SHUFFLE, batch_size=cfg.TEST.BATCH_SIZE, drop_last=True)

            test(model=model, criterion=criterion, model_path=args.model, test_loader=test_loader, paths=image_path, dev=args.device, model_type=args.type)


if __name__ == '__main__':
    main()
