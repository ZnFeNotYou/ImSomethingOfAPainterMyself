from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

ImageDir = os.path.abspath("../" + os.curdir + "/data/monet_jpg/")

class DataGenerator():
    def __init__(self, imagepath):
        self.imagepath = imagepath
        self.paintings = self.load_data()
        
    def load_data(self):
        paintings = []       
        for filename in os.listdir(self.imagepath):
            f = os.path.join(self.imagepath, filename)
            
            if os.path.isfile(f):
                painting = np.array(Image.open(f))                
                paintings.append(painting)
                
        paintings = np.array(paintings)
        
        return paintings
    
    def __len__(self):
        return len(self.paintings)
    
    def __getitem__(self, idx):
        #Shape of outcomes has to be 3, 256, 256
        #y-values will be added later in training class
        painting = self.paintings[idx]
        x = np.zeros((painting.shape[2], painting.shape[0], painting.shape[1]))
        
        for i in range(painting.shape[2]):
            x[i] = painting[:,:,i]
                       
        x = torch.from_numpy(x)
        
        return x.long()
    
    
    
dataClass = DataGenerator(ImageDir)
Length = dataClass.__len__()
Paint = dataClass.paintings
x = dataClass.__getitem__(10)

    
test_ds, _ = torch.utils.data.random_split(dataClass, lengths=(Length,0))
test_dl = DataLoader(test_ds, batch_size=2)

for x in test_dl:
    a = 1
    
plt.imshow(Paint[0])

    
    
    