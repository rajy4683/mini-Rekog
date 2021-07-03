"""
    This file contains non-core utility functions 
    used in the overall project
"""

import random
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F


def show_sample_images(images, labels, classes, max_count=25):
    print(images.shape, torch.mean(images,[0,2,3]),torch.std(images,[0,2,3]))
    fig = plt.figure(figsize=(10,10))
    for idx in np.arange(max_count):
        ax = fig.add_subplot(5, 5, idx+1, xticks=[], yticks=[])
        plt.imshow(np.transpose(images[idx].cpu().numpy(), (1, 2, 0)))
        ax.set(xlabel="Actual="+classes[labels[idx].cpu().numpy()])


def rand_run_name():
    ran = random.randrange(10**80)
    myhex = "%064x" % ran
    #limit string to 64 characters
    myhex = myhex[:10]
    return myhex

def generate_model_save_path(base="/content/model_saves", rand_string=None):
    if rand_string == None:
        rand_string=rand_run_name()
    file_name = "model-"+rand_string+".pt"
    return os.path.join(base,file_name)

# functions to show an image
def imshow_labels(img,labels):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    fig = plt.figure(figsize=(10,10))
    #plt.figsize = (10,20)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

# functions to show an image
def imshow(img,labels):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    #print(npimg.shape)
    fig = plt.figure(figsize=(10,10))
    #plt.figsize = (10,20)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

# get some random training images
def get_image_samples(imageloader, classes,count=32,seed=0xDEADBEEF):
    torch.manual_seed(seed)
    dataiter = iter(imageloader)
    images, labels = dataiter.next()
    # show images
    print(images.shape, torch.mean(images,[0,2,3]),torch.std(images,[0,2,3]))
    imshow(torchvision.utils.make_grid(images[:count], nrow=8),labels[:count])
    # print labels
    print(' '.join('%5s' % classes[labels[j]] for j in range(count)))



# get some random training images
def get_image_samples(imageloader, classes,count=32, seed=0xDEADBEEF):
    torch.manual_seed(seed)
    dataiter = iter(imageloader)
    images, labels = dataiter.next()
    # show images
    print(images.shape, torch.mean(images,[0,2,3]),torch.std(images,[0,2,3]))
    imshow(torchvision.utils.make_grid(images[:count], nrow=8),labels[:count])
    # print labels
    print(' '.join('%5s' % classes[labels[j]] for j in range(count)))


def plot_graphs(df_array, legend_arr, columns=['Test Accuracy'], xlabel="Epochs", ylabel="Accuracy"):
    fig, ax = plt.subplots(figsize=(15, 6))
    for i in range(len(df_array)):
        for col in columns:
            ax.plot(range(df_array[i].shape[0]),
                    df_array[i][col])
    # ax.plot(range(40),
    #         base_metrics_dataframe['Test Accuracy'],
    #         'g',
    #         color='blue')
    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.legend(legend_arr)
    plt.show()

"""
    Split any dataframe into training and test sets as per split_pct
"""
def split_df(df_data_list, split_pct=0.7):
    full_set = df_data_list.index.values
    train_set = df_data_list.sample(frac=split_pct).index.values
    test_set  = [ index_val for index_val in full_set if index_val not in train_set]

    df_train = df_data_list.loc[train_set].reset_index()
    df_train.drop('index',axis=1,inplace=True)

    df_test = df_data_list.loc[test_set].reset_index()
    df_test.drop('index',axis=1,inplace=True)
    return df_train, df_test

#### Function to plot misclassified images from dataset loader
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
            #torch.
            #np.append(preds, pred.squeeze().cpu())
            orig_labels = target.cpu().numpy()
            pred_labels = pred.squeeze().cpu().numpy()
            #print(orig_labels.shape, pred_labels.shape)
            mislabeled_index = np.where(orig_labels != pred_labels)[0]
            #print(orig_labels)
            #print(pred_labels)
            total_rounds+=1
            if (mislabeled_index.shape[0] > 0):
                #print(mislabeled_index)
                for iCount in range(len(mislabeled_index)):
                    #print(transforms.Normalize()data[offset]((-0.1307,), (1/0.3081,)))
                    #plt.imshow(data[offset].cpu().numpy().squeeze(), cmap='gray_r')
                    offset = mislabeled_index[iCount]
                    error_images.append(data[offset].cpu().numpy().squeeze())
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
    #print('\nEpoch: {:.0f} Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
    #    epoch_number, test_loss, correct, len(test_loader.dataset),
    #    100. * correct / len(test_loader.dataset)))
    #test_acc.append(test_accuracy)
    #test_losses.append(test_loss)

    # figure = plt.figure()
    # num_of_images = total_misclassified
    # for index in range(1, num_of_images):
    #     plt.subplot(5, 13, index)
    #     plt.axis('off')
    #     plt.imshow(error_images[index], cmap='gray_r')
    #     #plt.label
    fig = plt.figure(figsize=(10,10))
    for idx in np.arange(25):
        ax = fig.add_subplot(5, 5, idx+1, xticks=[], yticks=[])
        plt.imshow(error_images[idx], cmap='gray_r')
        #ax.set_title("Pred="+str(np.int(preds[idx])))
        ax.set(ylabel="Pred="+str(np.int(preds[idx])), xlabel="Actual="+str(np.int(actuals[idx])))

    return test_accuracy, test_loss, None#error_images#, data(np.where(orig_labels != pred_labels))


def plot_misclassified_rgb(args, 
                             model, 
                             device, 
                             test_loader,
                             classes,
                             epoch_number,
                             max_images=20):
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
            #torch.
            #np.append(preds, pred.squeeze().cpu())
            orig_labels = target.cpu().numpy()
            pred_labels = pred.squeeze().cpu().numpy()
            #print(orig_labels.shape, pred_labels.shape)
            mislabeled_index = np.where(orig_labels != pred_labels)[0]
            #print(orig_labels)
            #print(pred_labels)
            total_rounds+=1
            if (mislabeled_index.shape[0] > 0):
                #print(mislabeled_index)
                for iCount in range(len(mislabeled_index)):
                    #print(transforms.Normalize()data[offset]((-0.1307,), (1/0.3081,)))
                    #plt.imshow(data[offset].cpu().numpy().squeeze(), cmap='gray_r')
                    offset = mislabeled_index[iCount]
                    error_images.append(data[offset].cpu().numpy().squeeze())
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
    for idx in np.arange(max_images):
        ax = fig.add_subplot(5, int(max_images/5), idx+1, xticks=[], yticks=[])
        # plt.imshow(error_images[idx], cmap='gray_r')
        npimg = np.transpose(error_images[idx], (1, 2, 0))
        plt.imshow(npimg)
        #ax.set_title("Pred="+str(np.int(preds[idx])))
        ax.set(ylabel="Pred="+str(np.int(preds[idx])), xlabel="Actual="+str(np.int(actuals[idx])))

    return test_accuracy, test_loss, error_images[:max_images]#, data(np.where(orig_labels != pred_labels))
