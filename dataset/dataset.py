from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

# PaintDir = os.path.abspath("../" + os.curdir + "/data/monet_jpg/")
# ImageDir = os.path.abspath("../" + os.curdir + "/data/photo_jpg/")


class DataGenerator():
    def __init__(self, path):
        self.path = path
        
        self.data = self.load_data()


    def load_data(self):
        paintings = []       
        for filename in os.listdir(self.path):
            f = os.path.join(self.path, filename)
            
            if os.path.isfile(f):
                painting = np.array(Image.open(f))                
                paintings.append(painting)
                
        paintings = np.array(paintings)
        
        return paintings
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        #Shape of outcomes has to be 3, 256, 256
        #y-values will be added later in training class      
        painting = self.data[idx]
        x = np.zeros((painting.shape[2], painting.shape[0], painting.shape[1]))
        for i in range(painting.shape[2]):
            x[i] = painting[:,:,i]
                       
        x = torch.from_numpy(x)
        
        return x.long()
    
    
    
# dataClass = DataGenerator(ImageDir)
# Length = dataClass.__len__()
# Paint = dataClass.data
# x = dataClass.__getitem__(7037)

    
# test_ds, _ = torch.utils.data.random_split(dataClass, lengths=(Length,0))
# test_dl = DataLoader(test_ds, batch_size=2)


    
# plt.imshow(Paint[0])

    
    
    