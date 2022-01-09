import csv
import time
import torch
import numpy as np

from dataset import dataset
from IPython.display import clear_output


class TrainGAN():
    def __init__(self, Paths, Generator, Discriminator, lossFn, Epoch_BatchSize, CUDA=False):
        #Paths
        self.Imagepath = Paths['Imagepath']
        self.Paintingpath = Paths['Paintingpath']
        self.Modelpath = Paths['Modelpath']
        self.Filename = Paths['Filename']
        
        #Networks
        self.Generator = Generator
        self.Discriminator = Discriminator
        self.lossFn = lossFn 
        
        self.Epoch = Epoch_BatchSize[0]
        self.PreTrain = Epoch_BatchSize[1]
        self.BatchSize = int(Epoch_BatchSize[2])
        
        self.CUDA= bool(CUDA)


    def loadModel(self, model):
        if self.CUDA:
            return model.cuda()
        else:
            return model
        
    
    def time(self, total_time):
        hours = int(total_time/3600)
        mins = int(total_time/60 - hours*60)
        sec = int(total_time - hours*3600 - mins*60)   

        return hours, mins, sec    

    
    def trainDiscriminatorReal(self, optimizer, realData): 
        optimizer.zero_grad()
        
        #Train on real data
        predictionReal = self.Discriminator(realData)
        errorReal = self.lossFn(predictionReal, torch.ones_like(predictionReal))
        errorReal.backward()   
        optimizer.step()
        
        return errorReal 
    
    
    def trainDiscriminatorFake(self, optimizer, fakeData): 
        optimizer.zero_grad()

        #Train on fake data
        predictionFake = self.Discriminator(fakeData)
        errorFake = self.lossFn(predictionFake, torch.zeros_like(predictionFake))
        errorFake.backward()
        optimizer.step()
        
        return errorFake 
    
    
    def trainGenerator(self, optimizer, fakeData):
        optimizer.zero_grad()
        
        #Generate fake data and train
        prediction =  self.Discriminator(fakeData)
        error = self.lossFn(prediction, torch.ones_like(prediction))
        error.backward()
        optimizer.step()
        
        return error
    
    
    def trainModel(self, optimizerD, optimizerG):
        print('=='*20)
        print('Start Datacollection')
        start_time = time.time()
        
        #Loading data 
        dataClassImage = dataset.DataGenerator(self.Imagepath)
        ImageLength = dataClassImage.__len__()
        
        dataClassPaint = dataset.DataGenerator(self.Paintingpath)
        PaintLength = dataClassPaint.__len__()
        
        image_ds, _ = torch.utils.data.random_split(dataClassImage, (ImageLength, 0))
        image_dl = torch.utils.data.DataLoader(image_ds, batch_size=self.BatchSize)
        
        paint_ds, _ = torch.utils.data.random_split(dataClassPaint, (PaintLength, 0))
        paint_dl = torch.utils.data.DataLoader(paint_ds, batch_size=self.BatchSize)
        
        RealDisc, FakeDisc, Gen = [], [], []
            
        print('=='*20)
        print('Start Pretraining')
        for pretrain in range(self.PreTrain):
            print('--'*20)
            print('PreTrain: %i/%i'%(pretrain, self.PreTrain-1))
            pretrain_start_time = time.time()
            
            for imageData in image_dl:
                if self.CUDA: imageData = imageData.cuda()
                
                fakeData = self.Generator(imageData)
                errorFake = self.trainDiscriminatorFake(optimizerD, fakeData.detach())
                
            for paintData in paint_dl:
                if self.CUDA: paintData = paintData.cuda()
                
                errorReal = self.trainDiscriminatorReal(optimizerD, paintData)
                                
            clear_output(wait = True)
            
            errorReal, errorFake = errorReal.detach().cpu().numpy(), errorFake.detach().cpu().numpy()
            RealDisc.append(errorReal)
            FakeDisc.append(errorFake)
            Gen.append(0)
            
        pretrain_time = time.time() - pretrain_start_time
        hours, mins, sec = self.time(pretrain_time)
        
        print('Duration of pretraining: %i h %i min %i sec'%(hours, mins, sec))
        print('=='*20)
        
        for epoch in range(self.Epoch):
            print('Start Training')
            print('--'*20)
            print('Epoch: %i/%i'%(epoch, self.Epoch-1))
            epoch_start_time = time.time()
            for imageData in image_dl:
                if self.CUDA: imageData = imageData.cuda()
                
                #Start with Discriminator
                fakeData = self.Generator(imageData)
                errorFakeDisc = self.trainDiscriminatorFake(optimizerD, fakeData.detach())
                
                #Continue with Generator
                errorGen = self.trainGenerator(optimizerG, fakeData)
                
            for paintData in paint_dl:
                if self.CUDA: paintData = paintData.cuda()
                
                errorRealDisc = self.trainDiscriminatorReal(optimizerD, paintData)
                
            epoch_time = time.time() - epoch_start_time
            hours, mins, sec = self.time(epoch_time) #### Copy self.time() function
            
            errorGen, errorRealDisc, errorFakeDisc = Gen.detach().cpu().numpy(), errorRealDisc.detach().cpu().numpy(), errorFakeDisc.detach().cpu().numpy()
            print('Generator Loss: %.f'%errorGen)
            print('Discriminator Loss Fake Data: %.f'%errorFakeDisc)
            print('Discriminator Loss Real Data: %.f'%errorRealDisc)
            print('Epoch time: %i min %i sec'%(mins, sec))
            
            RealDisc.append(errorRealDisc)
            FakeDisc.append(errorFakeDisc)
            Gen.append(errorGen)
        
        if epoch%5 == 0 or epoch == (self.Epoch-1):
            #Saving Models
            torch.save(self.Generator.state_dict(), self.Modelpath+'_'+self.Filename+'_Generator.pt')
            torch.save(self.Discriminator.state_dict(), self.Modelpath+'_'+self.Filename+'_Discriminator.pt')
            
            #Miscellaneous
            Header = ['PreTrain:', self.PreTrain, 'Epochs:', self.Epoch, 'BatchSize:', self.BatchSize]
            MeanLoss = ['Biggest Generator Train Loss:', np.array(Gen).max()]
            Lossfunction = ['Loss Function: ', self.lossFn]
            ColumnOrder = ['Gen', 'DiscReal', 'DiscFake']
                    
            with open(self.modelpath+'Loss_'+self.filename+'.csv', 'w') as f:
                writer = csv.writer(f)
                    
                writer.writerow(Header)
                writer.writerow(MeanLoss)
                writer.writerow(Lossfunction)
                writer.writerow(ColumnOrder)
                for i in range(len(Gen)):
                    row = np.array([Gen[i], RealDisc[i], FakeDisc[i]])
                    writer.writerow(row)
                    
        total_time = time.time() - start_time
        hours, mins, sec = self.time(total_time)
        print('=='*20)
        print('Total time needed: %i h, %i min %i sec'%(hours, mins, sec))
        print('=='*20)
            
        

        

                
                
                
                
                
                
                
                
        
        
        
    
    
    
    






































