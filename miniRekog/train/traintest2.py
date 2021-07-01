import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR, OneCycleLR, MultiStepLR, CyclicLR
from ..utils import fileutils
import wandb
from tqdm import tqdm
from torchsummary import summary
#from torchlars import LARS
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import torchvision
from torch.optim.optimizer import Optimizer
from torch._six import inf
from sklearn.metrics import confusion_matrix
import seaborn as sns
from fastprogress.fastprogress import master_bar, progress_bar


def unnormalize(img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    st = torch.FloatTensor(std).view(3,1,1)
    mt = torch.FloatTensor(mean).view(3,1,1)
    img_unnorm = (img_tensor*st)+mt
    return np.transpose(img_unnorm, (1, 2, 0))

def show_misclassfied_images_classwise(model, imageloader, classes):
    epoch_test_acc,epoch_test_loss,error_images, preds, actuals = plot_misclassified(None, 
                                                                    model.to(torch.device("cuda")), 
                                                                    torch.device("cuda"), 
                                                                    imageloader, 
                                                                    classes,1)
    print(epoch_test_acc,epoch_test_loss)
    #fileutils.show_sample_images(error_images, labels, dataloader.classes)


    for class_idx in range(len(classes)):
        fig = plt.figure(figsize=(30,30))
        for idx,pos in enumerate(np.where(actuals == class_idx)[0][:10]):
            ax = fig.add_subplot(1, 10, idx+1, xticks=[], yticks=[])
            #plt.imshow(np.transpose(error_images[pos].cpu().numpy(), (1, 2, 0)))
            plt.imshow(unnormalize(error_images[pos].cpu()))
            ax.set(ylabel="Pred="+classes[np.int(preds[pos])], xlabel="Actual="+classes[np.int(actuals[pos])])
    return error_images, preds, actuals


def show_misclassfied_images(model, imageloader, classes):
    epoch_test_acc,epoch_test_loss,error_images, preds, actuals = plot_misclassified(None, 
                                                                    model.to(torch.device("cuda")), 
                                                                    torch.device("cuda"), 
                                                                    imageloader, 
                                                                    classes,1)
    print(epoch_test_acc,epoch_test_loss)
    #fileutils.show_sample_images(error_images, labels, dataloader.classes)


    fig = plt.figure(figsize=(20,20))
    for idx in np.arange(25):
        ax = fig.add_subplot(5, 5, idx+1, xticks=[], yticks=[])
        #plt.imshow(np.transpose(error_images[idx].cpu().numpy(), (1, 2, 0)))
        plt.imshow(unnormalize(error_images[idx].cpu()))
        ax.set(ylabel="Pred="+classes[np.int(preds[idx])], xlabel="Actual="+classes[np.int(actuals[idx])])
    return error_images, preds, actuals


def return_traced_model(model, input_size):
    return torch.jit.trace(model, input_size)

def model_builder(model_class=None, weights_path=None, local_device=torch.device("cpu")):
    if (model_class == None):
        print("Please provide the model object to be used")
        return
    local_model = model_class()#.to(local_device)
    try:
        if (weights_path != None):
            checkpoint = torch.load(weights_path)
            local_model.load_state_dict(checkpoint['model_state_dict'])
                #torch.load(weights_path, map_location=local_device))           
    except:
        print("Some execption occured during loading the model")
    return local_model.to(local_device)

def model_builder2(model_class=None, weights_path=None, local_device=torch.device("cpu")):
    best_acc=0.0
    if (model_class == None):
        print("Please provide the model object to be used")
        return
    local_model = model_class#.to(local_device)
    try:
        if (weights_path != None):
            checkpoint = torch.load(weights_path)
            best_acc = checkpoint['test_acc']
            local_model.load_state_dict(checkpoint['model_state_dict'])
            
                #torch.load(weights_path, map_location=local_device))           
    except:
        print("Some execption occured during loading the model")
    return local_model.to(local_device),best_acc

def get_classacc_conf_matrix(model, image_loader, classes, device=torch.device("cpu")):
    #basemodelclass.CIFARModelDepthDilate()
    class_correct = list(0. for i in range(len(classes)))
    class_total = list(0. for i in range(len(classes)))
    class_acc_map = {}
    #model=model.to(device)
    y_true = []
    y_pred = []
    with torch.no_grad():
        for images, labels in image_loader:
            y_true.extend(labels.numpy())
            try:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                c = (predicted == labels).squeeze()
                #print(count(c))
                #print(c)
                for i in range(images.shape[0]):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1
                y_pred.extend(predicted.to("cpu").numpy())
            except:
                print("Exception:",predicted, labels)
                continue
    
    for i in range(len(classes)):
        class_accuracy = 100 * class_correct[i] / class_total[i]
        print('Accuracy of %5s : %2d %%' % (classes[i], class_accuracy))
        class_acc_map[classes[i]]=class_accuracy
    
    if(len(y_true) != len(y_pred)):
        print("Predicted:{} vs Actual:{} counts mismatch".format(len(y_pred),len(y_true)))
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]
        
    conf_matrix = confusion_matrix(y_true, y_pred)
    ax = sns.heatmap(conf_matrix,annot=True,fmt='d',cmap='Blues')
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    
    return class_acc_map, conf_matrix

def classwise_accuracy(model, image_loader, classes, device=torch.device("cpu")):
    
    #basemodelclass.CIFARModelDepthDilate()
    class_correct = list(0. for i in range(len(classes)))
    class_total = list(0. for i in range(len(classes)))
    class_acc_map = {}
    with torch.no_grad():
        for data in image_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(images.shape[0]):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    
    for i in range(len(classes)):
        class_accuracy = 100 * class_correct[i] / class_total[i]
        print('Accuracy of %5s : %2d %%' % (classes[i], class_accuracy))
        class_acc_map[classes[i]]=class_accuracy
    
    return class_acc_map

"""
    This function can be used for CyclicLR based training
"""
def train_cyclic_lr(args, model, device, 
          train_loader, optimizer, scheduler, criterion,
          epoch_number,l1_loss=False, l1_beta = 0):
    model.train()
    pbar = tqdm(train_loader)
    train_loss = 0
    train_accuracy = 0
    #scheduler = CyclicLR(optimizer, base_lr=1e-3, max_lr=0.1, mode='triangular', gamma=1., scale_fn='triangular',step_size_up=200)
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        train_accuracy += pred.eq(target.view_as(pred)).sum().item()

        loss = criterion(output, target)#F.nll_loss(output, target)
        if l1_loss == True:
            l1_crit = nn.L1Loss(size_average=False)
            reg_loss = 0
            for param in model.parameters():
                target = torch.zeros_like(param)    
                reg_loss += l1_crit(param, target)
            loss += (l1_beta * reg_loss)
        loss.backward()
        optimizer.step()
        scheduler.step()

        pbar.set_description(desc= f'loss={loss.item()} batch_id={batch_idx}')
        train_loss += loss.item()
    
    train_accuracy = (100. * train_accuracy) / len(train_loader.dataset)
    train_loss /= len(train_loader.dataset)
    return train_accuracy, train_loss

def plot_misclassified(args, model, device, test_loader,classes,epoch_number):
    model.eval()
    test_loss = 0
    correct = 0
    preds = np.array([])
    actuals = np.array([])
    error_images = []
    total_misclassified = 0
    total_rounds = 0
    with torch.no_grad():
        for data, target in test_loader:
            #print(len(data))
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            orig_labels = target.cpu().numpy()
            pred_labels = pred.squeeze().cpu().numpy()
            mislabeled_index = np.where(orig_labels != pred_labels)[0]
            total_rounds+=1
            if (mislabeled_index.shape[0] > 0):
                #print(mislabeled_index)
                for iCount in range(len(mislabeled_index)):            
                    #print(transforms.Normalize()data[offset]((-0.1307,), (1/0.3081,)))
                    #plt.imshow(data[offset].cpu().numpy().squeeze(), cmap='gray_r')
                    offset = mislabeled_index[iCount]
                    error_images.append(data[offset])#.cpu()#.numpy().squeeze())
                    preds=np.append(preds, pred_labels[offset])
                    actuals = np.append(actuals, orig_labels[offset])
                    total_misclassified += 1
                #error_images.append(data[mislabeled_index].cpu().numpy())#,axis=1)
                #preds=np.append(preds, pred_labels[mislabeled_index])
                

        #example_images.append(wandb.Image(
        #        data[0], caption="Pred: {} Truth: {}".format(classes[pred[0].item()], classes[target[0]])))
    #print("Total images worked on:",total_rounds)
    test_loss /= len(test_loader.dataset)
    test_accuracy = (100. * correct) / len(test_loader.dataset)
    print((total_misclassified))
    print(preds.shape)

    fig = plt.figure(figsize=(10,10))

    return test_accuracy, test_loss, error_images, preds, actuals #error_images#, data(np.where(orig_labels != pred_labels))
        

"""
    Training loop for Mono MaskDepth
"""
def train_monomaskdepth(model, 
          device, 
          train_loader, 
          optimizer,           
          criterion=nn.MSELoss(),
          ):
    model.train()
    pbar = tqdm(train_loader)
    train_loss = 0
    #train_accuracy = nn.MSELoss()
    for batch_idx, dataset in enumerate(pbar):
        data = dataset['input'].to(device)
        gt_mask = dataset['output'][0].to(device) ### Ground truth for mask
        gt_depth = dataset['output'][1].to(device) ### Ground truth for depth
        optimizer.zero_grad()
        pred_mask, pred_depth = torch.split(model(data), 3,dim=1) ### Model will output in form of (6, h, w)
        # get the index of the max log-probability
        #train_accuracy += pred.eq(target.view_as(pred)).sum().item()

        loss = (torch.sqrt(criterion(gt_mask, pred_mask)) + torch.sqrt(criterion(gt_depth, pred_depth)))/2
        loss.backward()
        optimizer.step()
        pbar.set_description(desc= f'loss={loss.item()} batch_id={batch_idx}')
        train_loss += loss.item()

    return (train_loss/batch_idx)


def train(args, model, device, 
          train_loader, 
          optimizer, scheduler,
          criterion, epoch_number,
          l1_loss=False, l1_beta = 0, 
          batch_step=False,
          masterProgbar=None):
    model.train()
    if masterProgbar is None:
        pbar = tqdm(train_loader)
    else:
        pbar = progress_bar(train_loader, parent=masterProgbar)
    train_loss = 0
    train_accuracy = 0
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        train_accuracy += pred.eq(target.view_as(pred)).sum().item()

        loss = criterion(output, target)#F.nll_loss(output, target)
        if l1_loss == True:
            l1_crit = nn.L1Loss(size_average=False)
            reg_loss = 0
            for param in model.parameters():
                target = torch.zeros_like(param)    
                reg_loss += l1_crit(param, target)
            loss += (l1_beta * reg_loss)
        loss.backward()
        optimizer.step()
        ### Specifically for CyclicLR. TODO: Add isinstance check for scheduler as well.
        if (batch_step == True and scheduler is not None):
            scheduler.step()
        if masterProgbar is None:
            pbar.set_description(desc= f'loss={loss.item()} batch_id={batch_idx}')
        else:
            masterProgbar.child.comment = f'loss={loss.item()} batch_id={batch_idx}'
        train_loss += loss.item()
    
    ### For other LR schedulers
    #if(batch_step != True and epoch_number > args.start_lr and scheduler is not None):
    #    scheduler.step(train_loss)
        
    train_accuracy = (100. * train_accuracy) / len(train_loader.dataset)
    train_loss /= len(train_loader)
    return train_accuracy, train_loss

def test(args, model, device, test_loader, criterion, classes,epoch_number):
    model.eval()
    test_loss = 0
    correct = 0
    example_images = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()#, reduction='sum')
            #.item() #F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
        #example_images.append(wandb.Image(
        #        data[0], caption="Pred: {} Truth: {}".format(classes[pred[0].item()], classes[target[0]])))

    test_loss /= len(test_loader.dataset)
    test_accuracy = (100. * correct) / len(test_loader.dataset)
    return test_accuracy, test_loss

#optimizer=optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
def execute_model(model_class, 
                  hyperparams, 
                  train_loader, 
                  test_loader, 
                  device,
                  classes,
                  optimizer_in=optim.SGD, 
                  wandb_param = None,
                  criterion=nn.CrossEntropyLoss,
                  scheduler=None,
                  prev_saved_model=None,
                  save_best=False, 
                  batch_step=False, 
                  lars_mode=False,
                  **kwargs):
    
    if wandb_param is None:
        hyperparams.set('run_name',fileutils.rand_run_name())
        wandb.init(config=hyperparams, project=hyperparams['project'])
    
    #wandb.watch_called = False # Re-run the model without restarting the runtime, 
    # unnecessary after our next release
    config = wandb.config
    model_path = fileutils.generate_model_save_path(rand_string=config.run_name)
    print("Model saved to: ",model_path)
    #print("Hyper Params:")
    #print(config)
    use_cuda = not config.no_cuda and torch.cuda.is_available()
    best_acc = 0.0
    #device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    
    # Set random seeds and deterministic pytorch for reproducibility
    # random.seed(config.seed)       # python random seed
    torch.manual_seed(config.seed) # pytorch random seed
    # numpy.random.seed(config.seed) # numpy random seed
    torch.backends.cudnn.deterministic = True

    # Initialize our model, recursively go over all modules and 
    # convert their parameters and buffers to CUDA tensors (if device is set to cuda)
    if(prev_saved_model != None):
        # model = model_builder(model_class, 
        #                       weights_path=prev_saved_model,    
        #                       local_device=device)
        model,best_acc = model_builder2(model_class, 
                      weights_path=prev_saved_model,    
                      local_device=device)
        print("Model loaded from ", prev_saved_model, " with previous accuracy:",best_acc)
    else:
        #model = model_class(config.dropout).to(device)
        model = model_class.to(device)
    
    summary(model.to(device),input_size=(3, 224, 224))
    optimizer = optimizer_in(model.parameters(), lr=config.lr,momentum=config.momentum,
                           weight_decay=config.weight_decay) #

    optimizer=optimizer_in(model.parameters(), 
                        lr=config.lr,
                        momentum=config.momentum,
                        weight_decay=config.weight_decay)
    masterProgbar = master_bar(range(1, config.epochs+1))
    for epoch in masterProgbar:
        #epoch_train_acc,epoch_train_loss = train(config, model, device, train_loader, optimizer,criterion(), epoch)

        epoch_train_acc,epoch_train_loss = train(config, model, device, 
                                                train_loader, optimizer,scheduler, 
                                                criterion(), epoch,
                                                batch_step=batch_step,
                                                masterProgbar=masterProgbar)   
        epoch_test_acc,epoch_test_loss = test(config, model, 
                                                device, test_loader,
                                                criterion(reduction='sum'), 
                                                classes,epoch
                                                )
        last_lr = scheduler.get_last_lr()[0]
        training_results_string = f'Epoch: {epoch:.0f}, Train set:, Average loss: {epoch_train_loss:.4f}, Accuracy: {epoch_train_acc:.3f}%, lr:{last_lr}'
        test_results_string = f'Epoch: {epoch:.0f}, Test set, Average loss: {epoch_test_loss:.4f}, Accuracy: {epoch_test_acc:.3f}%'
                                
        # print('\nEpoch: {:.0f} Train set: Average loss: {:.4f}, Accuracy: {:.3f}%, lr:{}'.format(
        # epoch, epoch_train_loss, epoch_train_acc,last_lr))
        # print('Epoch: {:.0f} Test set: Average loss: {:.4f}, Accuracy: {:.3f}%'.format(
        # epoch, epoch_test_loss, epoch_test_acc))
        masterProgbar.write(f'{training_results_string}\n{test_results_string}')
        
        wandb.log({ "Train Accuracy": epoch_train_acc, 
                   "Train Loss": epoch_train_loss, 
                   "Test Accuracy":epoch_test_acc, 
                   "Test Loss": epoch_test_loss,
                   #"Learning Rate": config.lr})
                   "Learning Rate": last_lr})
        
        if(save_best == True and epoch_test_acc > best_acc):
            print(f"Model saved as Test Accuracy increased from {best_acc} to {epoch_test_acc} at epoch {epoch}")
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc':epoch_test_acc,
                'epoch':epoch
                }, model_path)
            best_acc = epoch_test_acc

        if (scheduler != None and 
            epoch > config.start_lr and 
            batch_step == False):
            # print("Non CyclicLR Case")
            scheduler.step(epoch_test_loss)
        
    print("Final model save path:",model_path," best Accuracy:",best_acc)
    wandb.save(model_path)
    return model_path