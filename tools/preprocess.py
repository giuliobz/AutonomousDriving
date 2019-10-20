import os, cv2, shutil, argparse, sys, tqdm
sys.path.insert(1, os.path.dirname(os.path.abspath('__file__')))
from PIL import Image
from config.config import cfg

SCALE_PERCENT = 12.5

def adjustImages(data_type):

    dir_name = os.path.dirname(os.path.abspath('__file__'))
    
    input_path = dir_name + cfg.DATASET_PATH['real'] + data_type + '/'
    output_path = dir_name + cfg.DATASET_PATH['modify'] + data_type + '/'
    
    num_seq = len(os.listdir(input_path))
    i = 1
    output = output_path + type + "/"

    if not os.path.exists(output):
        os.makedirs(output)

    for seq in sorted(os.listdir(input_path)):
        if seq != ".DS_Store":
            image_path = input_path + seq + "/left/"
            output = output_path + data_type + "/" + seq + "/"

            if not os.path.exists(output):
                os.makedirs(output)

            shutil.copy(input_path + seq + "/trajectories.json", output)

            output = output + "left/"

            if not os.path.exists(output):
                os.makedirs(output)

            print('*' * 100)
            print('Analisi Sequenza {n_seq}/{num_seq} in corso ...'.format(n_seq=i,num_seq=num_seq))
            print("Generazione Nuove Immagini dall'Input:", image_path)
            for image in tqdm.tqdm(sorted(os.listdir(image_path))):
                if (image.endswith(".png")):  # Controlla che si tratti di un'immagine
                    img = cv2.imread(image_path + '{img_name}'.format(img_name=image), cv2.IMREAD_UNCHANGED)
                    height = img.shape[0]
                    width = img.shape[1]
                    crop_img = img[60:-25, :, :]  # Ritaglio l'imagine
                    new_width = int(width * SCALE_PERCENT / 100)
                    new_height = int(height * SCALE_PERCENT / 100)
                    dim = (new_width, new_height)
                    resized = cv2.resize(crop_img, dim, interpolation=cv2.INTER_AREA)

                    cv2.imwrite(os.path.join(output, image), resized)
            print('Output Nuove Immagini:', output)
            print('*' * 100)
            i += 1

def adjustDepth(data_type):

    dir_name = os.path.dirname(os.path.abspath(__file__))

    input_path = dir_name + cfg.DATASET_PATH['real'] + data_type + '/'
    output_path = dir_name + cfg.DATASET_PATH['modify'] + data_type + '/'
    

    num_seq = len(os.listdir(input_path))
    i = 1

    if not os.path.exists(output):
        os.makedirs(output)

    for seq in sorted(os.listdir(input_path)):
        if seq != ".DS_Store":
            depths_input = input_path + seq + "/depth/"
            output = output_path + data_type + "/" + seq + "/"

            if not os.path.exists(output):
                os.makedirs(output)

            output = output + "depth/"

            if not os.path.exists(output):
                os.makedirs(output)

        print('*' * 100)
        print('Analisi Sequenza {n_seq}/{num_seq} in corso ...'.format(n_seq=i,num_seq=num_seq))
        print("Generazione Nuove Immagini dall'Input:", depths_input)
        for image in tqdm.tqdm(sorted(os.listdir(depths_input))):
            if (image.endswith('png')):  # Controlla che si tratti di un'immagine
                img = cv2.imread(depths_input + '{img_name}'.format(img_name=image), cv2.IMREAD_UNCHANGED)
                height = img.shape[0]
                width = img.shape[1]
                crop_img = img[60:-25, :]  # Ritaglio l'imagine
                new_width = int(width * SCALE_PERCENT / 100)
                new_height = int(height * SCALE_PERCENT / 100)
                dim = (new_width, new_height)
                resized = cv2.resize(crop_img, dim, interpolation=cv2.INTER_AREA)

                cv2.imwrite(os.path.join(output, image), resized)
        print('Output Nuove Immagini:', output)
        print('*' * 100)
        i+=1


def main():

    parser = argparse.ArgumentParser(description="Compare prediction in video")
    parser.add_argument("--type",                dest='data_type',                default=None,        help="specify the type of the image folder: train - test - validation")
    
    args = parser.parse_args()

    adjustImages(data_type = args.data_type)
    adjustDepth(data_type = args.data_type)

if __name__ == '__main__':
    main()




