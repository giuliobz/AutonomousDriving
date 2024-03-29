import argparse, os, sys
sys.path.insert(1, os.path.dirname(os.path.abspath('__file__')))
from config.config import cfg
from utils.trajectory_images import save_video


def main():

    parser = argparse.ArgumentParser(description="Compare prediction in video")
    parser.add_argument("--input_path",     dest="input",               default=None,                            help="path to csv file with prediction and image")
    parser.add_argument("--video_name",     dest='name',                default='compare_trajectory.avi',        help="name of the video, put .avi in the name")
    parser.add_argument("--model_type",     dest='model_type',          default='compare_trajectory.avi',        help="name of the video, put .avi in the name")

    args = parser.parse_args()
    dir_name = os.path.dirname(os.path.abspath('__file__'))

    save_video(video_name=dir_name + cfg.SAVE_VIDEO_PATH[args.model_type] + args.name, csv_path=args.input, len_seq=cfg.TRAIN.LEN_SEQUENCES, model_type=args.model_type)


if __name__ == '__main__':
    main()