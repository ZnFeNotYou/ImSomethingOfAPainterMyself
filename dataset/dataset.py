from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

ImageDir = '/home/notyou/Kaggle_Challenges/Data_ImSomethingOfAPainterMyself/monet_jpg/'



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
        x = torch.tensor(self.paintings[idx]) #Shape is 3,256,256
        #What is the y? Is it 0 or 1 for true/false?
        y = 1        
        y = torch.tensor(y)
        
        return x.long(), y.float()
    
    
    
dataClass = DataGenerator(ImageDir)
Length = dataClass.__len__()
Paint = dataClass.paintings
    
test_ds = torch.utils.data.random_split(dataClass, lengths=(Length,0))
test_dl = DataLoader(test_ds)

plt.imshow(Paint[0])



    
    
    