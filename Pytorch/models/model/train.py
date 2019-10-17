import sys, os, math, torch
sys.path.insert(1, '/home/biondibazzanti/AutonomousDriving/Pytorch/models')
from utils.pixel_to_coordinate  import pixel2coordinate
from utils.utilities            import create_weight_matrix, save_plot_loss, AverageMeter, adapt_learning_rate, export_plot_from_tensorboard
from torch.optim.lr_scheduler   import StepLR
from torch.utils.tensorboard    import SummaryWriter
import torch.nn                 as nn
import numpy                    as np
from os                         import listdir
from os.path                    import isfile, join
import cv2
from torchvision.utils          import make_grid

def train(model, criterion, optimizer, train_loader, val_loader, batch_size, epochs, val_period, clip, save_weights, dev, l, learning_rate, dec_period, opt):
    
    limit = 1e-07
    global_min_val_loss = np.Inf
    t_losses = AverageMeter() 
    iteration = 1

    path_run = save_weights.split("/")
    event_log_path = "/home/biondibazzanti/tensorboard_runs/" + path_run[4]
    writer = SummaryWriter(event_log_path)
    val_losses = {}
    train_losses = {}
    
    device = torch.device('cuda:' + dev)
    
    torch.cuda.reset_max_memory_allocated(device)
    model.to(device)

    if l == 'weighted':
        weighted_matrix = create_weight_matrix(batch_size, model.len_seq)

    print()
    print("Starting training the model")

    for i in range(epochs):
        
        print('*' * 100)
        print("Epoch : {}".format(i + 1))
        model.train()
        scheduler = StepLR(optimizer, step_size=dec_period, gamma=0.1)

        for image, labels in train_loader:
            optimizer.zero_grad()

            train_inputs = image.to(device).float()
            out = model(train_inputs, device)

            if l == 'weighted':
                loss = criterion(out, labels.float(), weighted_matrix.float())
            else:
                #loss = torch.sqrt(criterion(out, labels.float()))
                loss = criterion(out, labels.float())

            if i in train_losses:
                train_losses[i].append(loss.item())
            else:
                train_losses[i] = [loss.item()]

            # tensorboard utilities
            t_losses.update(loss.item(), batch_size)
            writer.add_scalar('Training/train_loss_value', loss.item(), iteration)
            iteration += 1

            # compute gradients and optimizer step
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

        writer.add_scalar('Training/train_global_loss', sum(train_losses[i])/len(train_losses[i]), i + 1)
        print("Epoch: {}/{}".format(i + 1, epochs),
                "Loss : {}".format(sum(train_losses[i])/len(train_losses[i])))


        #### Validation step

        if (i + 1) % val_period == 0:
            
            print()
            print("Starting valid test")
            model.eval()

            with torch.no_grad():

                for val_image, val_labels in val_loader:

                    val_image = val_image.to(device)
                    val_out = model(val_image.float(), device)

                    if l == 'weighted':
                        val_loss = criterion(val_out, val_labels.float(), weighted_matrix.float())
                    else:
                        val_loss = criterion(val_out, val_labels.float())
                    
                    if i in val_losses:
                        val_losses[i].append(val_loss.item())
                    else:
                        val_losses[i] = [val_loss.item()]

                    writer.add_scalar('Validation/valid_loss_value', val_loss.item(), iteration)
            
            writer.add_scalar('Validation/valid_global_loss', sum(val_losses[i])/len(val_losses[i]), i + 1)
            
            print("End valid test")
            print("Train Loss: {:.3f} - ".format(sum(train_losses[i])/len(train_losses[i])),
                    "Validation Loss: {:.3f}".format(sum(val_losses[i])/len(val_losses[i])))
            
            torch.save(model.state_dict(), save_weights + '/weight_' + str(sum(val_losses[i])/len(val_losses[i])) +  '_'  + str(i + 1) + '.pth')
            
    
        #if (i+1) % 10 == 0:
        #    image_acaso(writer, val_out, val_labels, i + 1)

        #if (i+1) % dec_period == 0:
        #    optimizer, limit = adapt_learning_rate(optimizer, learning_rate, i+1, limit, dec_period)
        scheduler.step()    

    writer.close()
    print()
    print("Save losses graphs")
    onlyfile = [f for f in listdir(event_log_path) if isfile(join(event_log_path, f))]
    export_plot_from_tensorboard(event_log_path + "/" + onlyfile[0], save_weights)


