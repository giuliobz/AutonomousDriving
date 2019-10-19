import csv, json, re, os, argparse
from config.config import cfg
SCALE_PERCENT = 12.5
K = float(SCALE_PERCENT / 100)
offset = 60
b = True


def crete_csv_sequence(seq_path, depth, data_type, len_seq):

    if depth == 'True':
        save_path = cfg.DATASET_PATH + depth + '/' + data_type + '_' + str(len_seq) + '_sequence_d.csv'
    elif depth == 'False':
        save_path = cfg.DATASET_PATH + depth + '/' + data_type + '_' + str(len_seq) + '_sequence.csv'
    else:
        print('Choose True or False')
        exit()

    with open(save_path, 'w') as csvfile:
        filewriter = csv.writer(csvfile)
        for seq in os.listdir(seq_path):
            i = 0
            tmp_cord = []
            path = seq_path + seq + "/"
            data = json.load(open(path + "trajectories.json"))
            for frame in data.keys():
                if i >= 250:
                    cord = []
                    if (len(data[frame]["object_0"]["future"]) >= len_seq):
                        for k in range(0, len_seq):
                            cord.append(data[frame]["object_0"]["future"][k])   

                        string = frame.split("_")
                        result = re.match('\d+', string[1])

                        number = (int(result.group()) + 1)

                        if number in range(0, 10):
                            n = '00000' + str(number)
                        elif number in range(10, 100):
                            n = '0000' + str(number)
                        elif number in range(100, 1000):
                            n = '000' + str(number)
                        else:
                            n = '00' + str(number)
                        
                        if depth == 'True':
                            img_path = path + "left/left" + n + ".png"
                            depth_path =  path + "depth/depth" + n + ".png"
                            
                            if (os.path.exists(img_path) and os.path.exists(depth_path)):
                                lines = [path, img_path, depth_path, cord]
                                filewriter.writerow(lines)
                        else: 
                            
                            img_path = path + "left/left" + n + ".png"
                            
                            if (os.path.exists(img_path) and os.path.exists(depth_path)):
                                lines = [path, img_path, cord]
                                filewriter.writerow(lines)
                else:
                    i += 1


def main():

    parser = argparse.ArgumentParser(description="Create the CSV file from image")
    parser.add_argument("--input_path",     dest="input",               default=None,                            help="path of the image")
    parser.add_argument("--depth",          dest="depth",               default='False',                         help="choose if use depth or not ")
    parser.add_argument("--type",           dest="data_type",           default=None,                            help="choose the dataset: train, validation, test ")
    parser.add_argument("--len_seq",        dest="len_seq",             default=None,                            help="Define the lenght of sequences")

    args = parser.parse_args()

    crete_csv_sequence(seq_path = args.input, depth = args.depth, data_type = args.data_type, len_seq = args.len_seq)


if __name__=='__main__':
    main()

