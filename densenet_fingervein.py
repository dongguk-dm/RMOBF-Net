
from __future__ import print_function
from __future__ import division
from multiprocessing import Process, freeze_support
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import models, transforms
import matplotlib.pyplot as plt
import sys
import time
import os
import copy
import random

from glob import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader

######### Set Seeds ###########
random_seed = 1234
torch.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU

# Top level data directory. Here we assume the format of the directory conforms
#   to the ImageFolder structure
data_dir = ""
save_dir = ""
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


def split_dataset():
    class_dir = ("auth", "impo")
    

    # GET DATA
    list_auth = np.array(sorted(glob(os.path.join(data_dir, class_dir[0], "*.*"))))
    list_impo = np.array(sorted(glob(os.path.join(data_dir, class_dir[1], "*.*"))))
    label_auth = np.zeros(list_auth.shape)
    label_impo = np.ones(list_impo.shape)

    # SPLIT
    split_ratio = int(label_auth.shape[0] * 0.8)

    # TRAIN DATA
    train_list  = np.append(list_auth[:split_ratio], list_impo[:split_ratio])
    train_label = np.append(label_auth[:split_ratio], label_impo[:split_ratio])

    # VALID DATA
    valid_list  = np.append(list_auth[split_ratio:], list_impo[split_ratio:])
    valid_label = np.append(label_auth[split_ratio:], label_impo[split_ratio:]) 

    return train_list, train_label, valid_list, valid_label

class DatasetTrain(Dataset):
    def __init__(self, imgs, labels):

        self.train_images = imgs
        self.train_labels = torch.tensor(labels, dtype=torch.int64)
        self.transforms = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    
    def __len__(self):
        return len(self.train_images)
    
    def __getitem__(self, idx):
        fname = self.train_images[idx]
        labels = self.train_labels[idx]

        img = Image.open(fname).convert("RGB")
        img = self.transforms(img)

        return img, labels, fname

class DatasetValid(Dataset):
    def __init__(self, imgs, labels):

        self.train_images = imgs
        self.train_labels = torch.tensor(labels, dtype=torch.int64)
        self.transforms = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    
    def __len__(self):
        return len(self.train_images)
    
    def __getitem__(self, idx):
        fname = self.train_images[idx]
        labels = self.train_labels[idx]

        img = Image.open(fname).convert("RGB")
        img = self.transforms(img)

        return img, labels, fname


def build_model(num_class):
    densenet = models.densenet161(pretrained=True)
    num_filters = densenet.classifier.in_features
    densenet.classifier = nn.Linear(num_filters, num_class)

    for name, param in densenet.named_parameters():
        print(name, "REQUIRE UPDATE : ", param.requires_grad)
    
    return densenet


def train(model, train_loader, valid_loder, loss_function, optimizer, num_epochs, device):
    

    best_acc = 0.0
    for epoch in range(0, num_epochs):
        since = time.time()
        epoch_loss, epoch_acc = 0, 0
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('=' * 20)

        model.train()

        train_loss, train_acc = 0, 0


        ############################## TRAIN ##############################
        for iter, (image, label, filename) in enumerate(train_loader):
            image = image.to(device)
            label = label.to(device)

            optimizer.zero_grad()

            output = model(image)


            loss = loss_function(output, label)


            _, preds = torch.max(output, 1)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc  += torch.sum(preds == label.data)

            if iter % 100 == 0:
                print("EPOCH : {}, ITER : {}, LOSS : {:.4f}".format(epoch, iter, loss.item()))
        
        epoch_loss = train_loss / len(train_loader)
        epoch_acc = train_acc.double() / len(train_loader.dataset)
        print('=' * 20)
        print('TRAIN EPOCH : {} Loss: {:.10f} Acc: {:.10f}'.format(epoch, epoch_loss, epoch_acc))

        ############################## VALID ##############################
        epoch_loss, epoch_acc = 0, 0
        valid_loss, valid_acc = 0, 0

        model.eval()
        for iter, (image, label, filename) in enumerate(valid_loader):
            image = image.to(device)
            label = label.to(device)

            with torch.no_grad():
                output = model(image)

                loss = loss_function(output, label)
                _, preds = torch.max(output, 1)

                valid_loss += loss.item()
                valid_acc += torch.sum(preds == label.data)

        
        epoch_loss = valid_loss / len(valid_loader)
        epoch_acc = valid_acc.double() / len(valid_loder.dataset)

        if epoch_acc > best_acc:
            print("**BEST ACC UPDATE**")
            best_acc = epoch_acc
            torch.save(model.state_dict(), os.path.join(save_dir, "best.pth"))


        print('VALID EPOCH : {} Loss: {:.10f} Acc: {:.10f}'.format(epoch, epoch_loss, epoch_acc))
        print('=' * 20)
        time_elapsed = time.time() - since
        print('EPOCH COMPLETE in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

        print("SAVE_MODEL....")
        torch.save(model.state_dict(), os.path.join(save_dir, "epoch_%s.pth" %(epoch)))

        print()
        print()





if __name__ == "__main__":

    num_classes = 2
    num_epochs = 30
    batch_size = 16

    train_list, train_label, valid_list, valid_label = split_dataset()
    train_dataset = DatasetTrain(train_list, train_label)
    valid_dataset = DatasetTrain(valid_list, valid_label)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    print("NUMBER of TRAINING IMAGES : ", len(train_dataset))
    print("NUMBER of VALIDATION IMAGES : ", len(valid_dataset))


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes)
    model = model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.0001)
    loss_function = nn.CrossEntropyLoss()

    train(model, train_loader, valid_loader, loss_function, optimizer, num_epochs, device)





