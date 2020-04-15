import os
import sys
import numpy as np
import cv2
import concurrent
from random import randint
from tqdm import tqdm
import time
from PIL import Image,ImageOps,ImageEnhance

from sklearn.model_selection import train_test_split

from torch.nn import Parameter
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import Dataset,DataLoader,Subset

import matplotlib.pyplot as plt

sys.path.insert(0, r'C:\Users\w.rogers\Tools_for_DICOM')
from process_dicom import listfolders

def absoluteFilePaths(directory):
    files = []
    for dirpath,_,filenames in os.walk(directory):
        for f in filenames:
            if f[-4:] == '.bmp':
                files.append(os.path.abspath(os.path.join(dirpath, f)))
    return files

def normalize(img):
    norm = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return norm
         
def get_images(img_path = r'C:\Users\w.rogers\Desktop\Data\Lung1', norm = False, max = False):
    files = absoluteFilePaths(img_path)
    images  = np.array([cv2.imread(file, 0) for file in files])
    if max:
        images = images[:max]
    if norm:
        images = np.array([normalize(image) for image in images])        
    return images

class LungDataset(Dataset):
    def __init__(self, path=None, init_transform=None, batch_transform=None, split=False, train=True, max=None):
        
        if path is not None:
            self.imgs = get_images(path, max=max)
        else:
            self.imgs = get_images(max=max)
            
        if init_transform:
            #print("Resizing images of type", type(self.imgs[0]), "and length", len(self.imgs), "...")
            self.imgs = [Image.fromarray(img) for img in self.imgs]
            self.imgs = [init_transform(i) for i in self.imgs]

        if split:
            X_train, X_test = train_test_split(self.imgs, test_size=.25, train_size=.75, random_state=86)
            if train:
                self.imgs = X_train
            else:
                self.imgs = X_test            
        
        self.batch_transform = batch_transform
        
        print("\n... LUNG1 Dataset Intialized with", len(self.imgs), "scans")
            
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self,idx):
        img =  self.imgs[idx]
        #print('-----------------------------')
        #print(imgs.shape, imgs.min(), imgs.max())
        #n_imgs = len(imgs)

        #n = randint(1, n_imgs-2)
        #imgs = imgs[n-1:n+2, :, :]
        #print(imgs.shape)
        #imgs = np.rollaxis(imgs, -1)
        #print(imgs.shape)
        #imgs = np.rollaxis(imgs, -1)
        #print(imgs.shape)

        if self.batch_transform:
            #print("doing transform ...")
            img = self.batch_transform(img)
        #print(imgs.shape, imgs_transformed.size())
        return img   
    
def load_images(batch_size=32, split=False, train=True):
    while True:
        for ii, data in enumerate(create_data_loader(batch_size, split=split, train=train)):
            yield data

def create_data_loader(batch_size, split=False, train=True):
    
    transform = transforms.Compose([#transforms.RandomCrop(img_size),
                                     #transforms.RandomHorizontalFlip(p=0.5),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0],std=[1])
                                    ])
    
    train_set = LungDataset(split=split, batch_transform=transform)
    
    train_loader = DataLoader(train_set,
                              shuffle=True, batch_size=batch_size,
                              num_workers=0, pin_memory=True)
    
    #print("The length of the training set is", len(train_set))
    return train_loader


if __name__ == "__main__":
    #imgs = get_images()
    #print(len(imgs), imgs[0].min(), imgs[0].max())
    #plt.imshow(imgs[0])
    n=8
    gen = load_images(batch_size=n)
    img_batch = next(gen).detach().cpu().numpy()
    for i, img in enumerate(img_batch):
        plt.subplot(1, n, i+1)
        plt.imshow(img.squeeze())
    plt.show()
    print(img.min(), img.max())








