import os, cv2, shutil, argparse
from PIL import Image

SCALE_PERCENT = 12.5

def adjustImages(input_path, output_path, type):
    num_seq = len(os.listdir(input_path))
    i = 1
    output = output_path + type + "/"

    if not os.path.exists(output):
        os.makedirs(output)

    for seq in sorted(os.listdir(input_path)):
        if seq != ".DS_Store":
            image_path = input_path + seq + "/left/"
            output = output_path + type + "/" + seq + "/"

            if not os.path.exists(output):
                os.makedirs(output)

            shutil.copy(input_path + seq + "/trajectories.json", output)

            output = output + "left/"

            if not os.path.exists(output):
                os.makedirs(output)

            print('*' * 100)
            print('Analisi Sequenza {n_seq}/{num_seq} in corso ...'.format(n_seq=i,num_seq=num_seq))
            print("Generazione Nuove Immagini dall'Input:", image_path)
            for image in sorted(os.listdir(image_path)):
                if (image.endswith(".png")):  # Controlla che si tratti di un'immagine
                    img = cv2.imread(image_path + '{img_name}'.format(img_name=image), cv2.IMREAD_UNCHANGED)
                    height = img.shape[0]
                    width = img.shape[1]
                    crop_img = img[60:-25, :, :]  # Ritaglio l'imagine
                    new_width = int(width * SCALE_PERCENT / 100)
                    new_height = int(height * SCALE_PERCENT / 100)
                    dim = (new_width, new_height)
                    resized = cv2.resize(crop_img, dim, interpolation=cv2.INTER_AREA)

                    print('Sequenza seq{n_seq}, Image {img} Saving'.format(n_seq=i, img=image))

                    cv2.imwrite(os.path.join(output, image), resized)
            print('Output Nuove Immagini:', output)
            print('*' * 100)
            i += 1

def adjustDepth(input_path, output_path, type):
    num_seq = len(os.listdir(input_path))
    i = 1
    output = output_path + type + "/"

    if not os.path.exists(output):
        os.makedirs(output)

    for seq in sorted(os.listdir(input_path)):
        if seq != ".DS_Store":
            depths_input = input_path + seq + "/depth/"
            output = output_path + type + "/" + seq + "/"

            if not os.path.exists(output):
                os.makedirs(output)

            output = output + "depth/"

            if not os.path.exists(output):
                os.makedirs(output)

        print('*' * 100)
        print('Analisi Sequenza {n_seq}/{num_seq} in corso ...'.format(n_seq=i,num_seq=num_seq))
        print("Generazione Nuove Immagini dall'Input:", depths_input)
        for image in sorted(os.listdir(depths_input)):
            if (image.endswith('png')):  # Controlla che si tratti di un'immagine
                img = cv2.imread(depths_input + '{img_name}'.format(img_name=image), cv2.IMREAD_UNCHANGED)
                height = img.shape[0]
                width = img.shape[1]
                crop_img = img[60:-25, :]  # Ritaglio l'imagine
                new_width = int(width * SCALE_PERCENT / 100)
                new_height = int(height * SCALE_PERCENT / 100)
                dim = (new_width, new_height)
                resized = cv2.resize(crop_img, dim, interpolation=cv2.INTER_AREA)

                print('Sequenza seq{n_seq}, Image {img} Saving'.format(n_seq=i, img=image))

                cv2.imwrite(os.path.join(output, image), resized)
        print('Output Nuove Immagini:', output)
        print('*' * 100)
        i+=1

def RGBA_to_RGB(input_image):
    for folder in os.listdir(input_image):
        out = input_image + folder + "/left/"

        if not os.path.exists(out):
            os.makedirs(out)
        i = 1
        for image in os.listdir(input_image + folder + "/RGBA/"):
            png = Image.open(input_image + folder + "/RGBA/" + image)
            background = Image.new("RGB", png.size, (255, 255, 255))
            background.paste(png, mask=png.split()[3])
            background.save(out + "left{number:06}.png".format(number=i))
            i += 1


def main():

    parser = argparse.ArgumentParser(description="Compare prediction in video")
    parser.add_argument("--input_directory",     dest="input",               default=None,                            help="path of the input directory")
    parser.add_argument("--output_directory",    dest="output",              default=None,                            help="path of the output directory")
    parser.add_argument("--type",                dest='type',                default=None,                            help="specify the type of the folder of the image: train - test - validation")
    
    args = parser.parse_args()
    input_directory = args.input
    output_image = args.output
    tipo = args.type

    adjustImages(input_directory + tipo + "/", output_image, tipo)
    adjustDepth(input_directory + tipo + "/", output_image, tipo)

if __name__ == '__main__':
    main()




