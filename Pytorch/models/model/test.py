import sys, os, math, csv, tqdm, torch
sys.path.insert(0, '/home/biondibazzanti/AutonomousDriving/Pytorch')
from utils.trajectory_images    import save_video_with_trajectory
from utils.pixel_to_coordinate  import pixel2coordinate
from utils.utilities            import create_weight_matrix, new_evaluation, evaluate_result
import numpy                    as np
import time

def test(model, criterion, model_path, batch_size, test_loader, paths, dev, len_seq, l):
    
    num_correct = 0
    test_losses = []
    distances = []
    split = model_path.split('/')
    tmp = split[5].split('_')
    name = tmp[0] + "_" + tmp[2]
    name = name.split('.')
    csv_file = "/home/biondibazzanti/" + split[3] + "/" + split[4] + "/summarize/summarize_test_"+ name[0] +".csv"
    video_name = "/home/biondibazzanti/" + split[3] + "/" + split[4] + "/video/video_trajectory_"+ name[0] +".avi"
    
    if os.path.exists(csv_file):
        os.remove(csv_file)
    
    model.load_state_dict(torch.load(model_path))

    device = torch.device('cuda:' + str(dev))
    model.to(device)
    model.eval()

    print("starting testig the model") 

    if l == 'weighted':
        weighted_matrix = create_weight_matrix(batch_size, model.len_seq)

    current_path = 0

    with open(csv_file, 'w') as result_file:
        with torch.no_grad():
            filewriter = csv.writer(result_file)
            for inputs, labels in test_loader:
                
                start = time.time()
                image = inputs.to(device)                
                output = model(image.float(), device)
                end = time.time()
                delta = 1 / (end - start) 

                print(delta)
                
                if l == 'weighted':
                    test_loss = criterion(output, labels.float(), weighted_matrix.float())
                else:
                    test_loss = criterion(output, labels.float())

                test_losses.append(test_loss.item())

                for i in range(len(output)):
                    predicted = output[i].detach().numpy()
                    real = labels[i].detach().numpy()
                    path = paths[current_path].replace("Dataset","Dataset_real")
                    lines = [path, predicted.tolist(), real.tolist()]
                    filewriter.writerow(lines)
                    current_path += 1
            
    print()
    print("saved resume csv file in " + csv_file)   
    print()
    print("Test loss: {:.3f}".format(np.mean(test_losses)))
    print('*' * 100)
    print('Starting Video Creation')

    save_video_with_trajectory(video_name, csv_file, len_seq)