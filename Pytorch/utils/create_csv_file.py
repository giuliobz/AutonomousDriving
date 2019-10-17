import csv
import json
import re
import os

SCALE_PERCENT = 12.5
K = float(SCALE_PERCENT / 100)
offset = 60
b = True


def crete_csv_sequence(seq_path, tipo, len_sequence):

        with open('/home/biondibazzanti/Image-and-video-Analysis/Pytorch/CSV_file/depth/' + tipo + '/' + tipo + '_' + str(len_sequence) + '_sequence_d.csv', 'w') as csvfile:
            filewriter = csv.writer(csvfile)
            for seq in os.listdir(seq_path):
                i = 0
                tmp_cord = []
                path = seq_path + seq + "/"
                data = json.load(open(path + "trajectories.json"))
                for frame in data.keys():
                    if i >= 250:
                        cord = []
                        if (len(data[frame]["object_0"]["future"]) >= len_sequence):
                            for k in range(0, len_sequence):
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
                                
                            img_path = path + "left/left" + n + ".png"
                            depth_path =  path + "depth/depth" + n + ".png"
                            
                            if (os.path.exists(img_path) and os.path.exists(depth_path)):
                                lines = [path, img_path, depth_path, cord]
                                filewriter.writerow(lines)

                    else:
                        i += 1


def crete_csv_sequence_nodepth(seq_path, tipo, len_sequence):

    with open('/home/biondibazzanti/AutonomousDriving/Pytorch/CSV_file/nodepth/' + tipo + '/' + tipo + '_' + str(len_sequence) + '_sequence.csv', 'w') as csvfile:
        filewriter = csv.writer(csvfile)
        for seq in os.listdir(seq_path):
            i = 0
            tmp_cord = []
            path = seq_path + seq + "/"
            data = json.load(open(path + "trajectories.json"))
            for frame in data.keys():
                if i >= 250:
                    cord = []
                    if (len(data[frame]["object_0"]["future"]) >= len_sequence):
                        for k in range(0, len_sequence):
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
                                
                        img_path = path + "left/left" + n + ".png"
                            
                        if (os.path.exists(img_path)):
                            lines = [path, img_path, cord]
                            filewriter.writerow(lines)

                else:
                    i += 1


def main():
    
    tipo = "test"
    
    path_turn = "/home/biondibazzanti/ds-turn/" + tipo + "/"
    path_data = "/home/biondibazzanti/Dataset/" + tipo + "/"

    #mi identifica la lunghezza della sequenza 
    len_sequence = 10
    
    #crete_csv_sequence(path, type, len_sequence)

    crete_csv_sequence_nodepth(path_data, tipo, len_sequence)


if __name__=='__main__':
    main()

