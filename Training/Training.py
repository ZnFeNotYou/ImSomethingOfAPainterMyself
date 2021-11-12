import csv
import time
import torch
from torch import nn, optim
from torchvision import transforms, datasets
from dataset import dataset


class TrainGAN():
    def __init__(self, Paths, Generator, Discriminator, lossFn, Epoch_BatchSize, CUDA=False):
        #Paths
        self.Imagepath = Paths['Imagepath']
        self.Modelpath = Paths['Modelpath']
        self.Filename = Paths['Filename']
        
        #Networks
        self.Generator = Generator
        self.Discriminator = Discriminator
        self.lossFn = lossFn 
        
        self.Epoch = Epoch_BatchSize[0]
        self.BatchSize = Epoch_BatchSize[1]
        
        self.CUDA= bool(CUDA)

    def realDataTarget(self, size):
        #Tensor containing ones - CUDA by default disabled
        data = torch.ones((size, 1))
        
        if self.CUDA: return data.cuda()            
        return data
    
    
    def fakeDataTarget(self, size):
        #Tensor containing zeros - CUDA by default disabled
        data = torch.zeros((size, 1))
        
        if self.CUDA: data.cuda()            
        return data
    
    
    def createNoise(self, size):
        #Creating random noise for generating fake data - CUDA by default disabled
        noise = torch.randn(size, 100) #Why size 100?
        if self.CUDA: return noise.cuda()
        return noise
    
    
    def trainDiscriminator(self, optimizer, realData, fakeData): 
        optimizer.zero_grad()
        
        #Train on real data
        predictionReal = self.Discriminator(realData)
        errorReal = self.lossFn(predictionReal, self.realDataTarget(predictionReal.size(0)))
        errorReal.backward()
        
        #Train on fake data
        predictionFake = self.Discriminator(fakeData)
        errorFake = self.lossFn(predictionFake, self.fakeDataTarget(predictionFake.size(0)))
        errorFake.backward()
        
        optimizer.step()
        
        return errorReal, errorFake 
    
    
    def trainGenerator(self, optimizer, fakeData):
        optimizer.zero_grad()
        
        #Generate fake data and train
        prediction =  self.Discriminator(fakeData)
        error = self.lossFn(prediction, self.realDataTarget(prediction.size(0)))
        error.backward()
        optimizer.step()
        
        return error
    
    
    def trainModel(self, optimizerD, optimizerG):
        print('=='*20)
        print('Start Training')
        start_time = time.time()
        
        #Loading data 
        dataClass = dataset.DataGenerator(self.Imagepath)
        Length = dataClass.__len__()
        
        data_ds, _ = torch.utils.data.random_split(dataClass, (Length, 0))
        data_dl = torch.utils.data.DataLoader(data_ds, batch_size=self.BatchSize)
        
        for epoch in range(self.Epoch):
            print('--'*20)
            print('Epoch: %i/%i'%(epoch, self.Epoch-1))
            epoch_start_time = time.time()
            for realData in data_dl:
                if self.CUDA: realData = realData.cuda()
                
                #Firstly training Discriminator
                fakeData = self.Generator(self.createNoise(realData.size(0))).detach()                
                errorRealDisc, errorFakeDisc = self.trainDiscriminator(optimizerD, realData, fakeData)
                
                #Secondly, the Generator
                fakeData = self.Generator(self.createNoise(realData.size(0)))
                errorGen = self.trainGenerator(optimizerG, fakeData)
                
            epoch_time = time.time() - epoch_start_time
            hours, mins, sec = self.time(epoch_time) #### Copy self.time() function
            
            print('Generator Loss: %.f'%errorGen)
            print('Discriminator Loss Fake Data: %.f'%errorFakeDisc)
            print('Discriminator Loss Real Data: %.f'%errorRealDisc)
            print('Epoch time: %i min %i sec'%(mins, sec))
            
        total_time = time.time() - start_time
        hours, mins, sec = self.time(total_time)
        print('=='*20)
        print('Total time needed: %i h, %i min %i sec'%(hours, mins, sec))
        print('=='*20)
        
        if epoch%5 == 0 or epoch == (self.Epoch-1):
            #Saving Models
            torch.save(self.Generator.state_dict(), self.Modelpath+'_'+self.Filename+'_Generator.pt')
            torch.save(self.Discriminator.state_dict(), self.Modelpath+'_'+self.Filename+'_Discriminator.pt')
            
            #Write Log output
            a = 1
            
        

        

                
                
                
                
                
                
                
                
        
        
        
    
    
    
    






































