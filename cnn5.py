import numpy as np
import torch #pytorch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim
import cv2
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets
from torch.utils.data.sampler import SubsetRandomSampler
import os #used to get directory
import random


# number of subprocesses to use for data loading
num_workers = 0
# how many samples per batch to load
batch_size = 20
# percentage of training set to use as validation
valid_size = 0.2

train_transform=transforms.Compose([
      transforms.ToTensor()
])

test_tranform=transforms.Compose([
      transforms.ToTensor()
])

# choose the training and test datasets
train_data = datasets.CIFAR10('data', train=True,
                              download=True,transform=train_transform)
test_data = datasets.CIFAR10('data', train=False,
                             download=True,transform=test_tranform)

# obtain training indices that will be used for validation
num_train = len(train_data)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(valid_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]

# define samplers for obtaining training and validation batches
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

# prepare data loaders (combine dataset and sampler)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
    sampler=train_sampler, num_workers=num_workers)
valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
    sampler=valid_sampler, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
    num_workers=num_workers)



# obtain one batch of training images
dataiter = iter(train_loader)
images, labels = next(dataiter)
images = images.numpy() # convert images to numpy for display
images.shape # (number of examples: 20, number of channels: 3, pixel sizes: 32x32)

one_image=images[0]

# Convert the image to grayscale if needed (assuming it's originally RGB)
one_image = np.mean(one_image, axis=0)

# Save the image to a file (e.g., in PNG format)
cv2.imwrite('saved_image.png', one_image * 255)  # multiply by 255 for proper scaling

# Now, you can use cv2.imread to read the saved image
file_path = 'saved_image.png'
true_np = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

true_np= true_np/255 
# create gaussian noise
x, y = true_np.shape
#gets dimensions of image f
mean = 0
#set gaussian distribution to 0
#primarily affect brightness level of noise
var = 0.01
#sets gaussian distribution variance to 0.01
sigma = np.sqrt(var)
#calcs standard deviation from variance
n = np.random.normal(loc=mean, 
                     scale=sigma, 
                     size=(x,y))

#generates gaussian noise
#2d array n
#loc parameter=mean
#scale parameter=sigma
#size=dimensions of image

# add a gaussian noise
noisy_np = true_np + n

def cv2disp(name, image, xpos, ypos,resize=None) :
    if resize is not None:
        image=cv2.resize(image,resize)
        cv2.imshow(name, image *1.0/(np.max(image)+1e-15)); cv2.moveWindow(name, xpos, ypos)
#window name,image,xpos and ypos of window, normalize so max value is 1, safety officer(1e-15)

nxd=true_np.shape[0]


cv2disp('True', true_np, 0 ,0,resize=(400,400) )#display true image in top left
cv2disp('Noisy', noisy_np, nxd, 0,resize=(400,400))#display noisy image next window beside true image

class Convolution_NxN(nn.Module):#conv with NxN kernel
    def __init__(self, kernel_size):
            super(Convolution_NxN, self).__init__() #need to do init for the nn.Module
            self.conv1kernel= nn.Conv2d(1, 1, kernel_size, padding=(int(kernel_size/2), int(kernel_size/2)), bias=False)
             #1 input image, 1 output image, kernel size 2D array, pad edges with 0's
            self.conv1kernel.weight.data.fill_(1.0/(kernel_size*kernel_size))
            #count up elements and fill with weights
        
    def forward(self, x):#forward function on input image x
          x=self.conv1kernel(x)
          return x
    
noisy_torch=torch.from_numpy(noisy_np).float().unsqueeze(0).unsqueeze(0)
#torch tensor version of noisy array, creating 2 extra dimensions with unsqueeze
#for convolution nxn
noisy_torch_conv = torch.from_numpy(noisy_np).float().unsqueeze(0).unsqueeze(0)
#for cnn
noisy_torch_cnn = torch.from_numpy(noisy_np).float().unsqueeze(0)



conv_fixed_kernel= Convolution_NxN(31)#31x31 kernel

kernel_torch=list(conv_fixed_kernel.parameters())[0]#torch tensor of kernel and parameters
cv2disp('Kernel', cv2.resize( np.squeeze(kernel_torch.detach().numpy()), (nxd,nxd), interpolation=0), 2*nxd, 0,resize=(400,400))
#resize to be same size as other windows, and go to next nearest neighbour window interpolation

conv_out_torch= conv_fixed_kernel(noisy_torch)#output of conv
conv_out_np=np.squeeze(conv_out_torch.detach().numpy())#detach and specify you want numpy array

cv2disp('Output Conv after average kernel', conv_out_np, 0, 0,resize=(400,400))#output of conv window


conv_trained_kernel= Convolution_NxN(31)#training kernel
loss_function=nn.MSELoss()
true_torch= torch.from_numpy(true_np).float().unsqueeze(0).unsqueeze(0)
#target is true image
optimiser_conv= torch.optim.Adam(conv_trained_kernel.parameters(), lr=0.003)


for epoch in range(50):
      output_conv=conv_trained_kernel(noisy_torch_conv)
      loss_conv=loss_function(output_conv, true_torch)
      loss_conv.backward()#computes dloss/dx for every parameter x which has requires_grad=True
      optimiser_conv.step()#does update of parameters
      optimiser_conv.zero_grad()#go back to zero to redo again

      kernel_torch=list(conv_trained_kernel.parameters())[0]
      cv2disp('TrainedKernel', cv2.resize( np.squeeze(kernel_torch.detach().numpy()), (nxd,nxd), interpolation=0), 1*nxd, 30+nxd,resize=(400,400))

      cnn_ouput=np.squeeze(output_conv.detach().numpy())
      cv2.putText(cnn_ouput, 'Ep %d' % epoch, (2, 30), 0, 1, int(np.max(cnn_ouput)+1), 1, cv2.LINE_AA)
      #epoch number text
      cv2disp('Ouput after conv', cnn_ouput, nxd*2, nxd+30,resize=(400,400))
      cv2.waitKey(1)

class CNN(nn.Module):#proper cnn 
      def __init__(self):
            super(CNN, self).__init__()
            self.CNN=nn.Sequential(
                  nn.Conv2d(1,8,7, padding=(3,3)), nn.PReLU(),
                  nn.Conv2d(8,8,7, padding=(3,3)), nn.PReLU(),
                  nn.Conv2d(8,8,7, padding=(3,3)), nn.PReLU(),
                  nn.Conv2d(8,8,7, padding=(3,3)), nn.PReLU(),
                  nn.Conv2d(8,1,7, padding=(3,3)), nn.PReLU(),
                  #sequential function, 1 input image, 8 output image/kernels of size 7, parametric relu, keeps negative values 
            )
      def forward(self,x):
            x=self.CNN(x)
            return x

#NEWLY ADDED
#NEWLY ADDED
#NEWLY ADDED



cnn_to_train=CNN()
loss_function=nn.MSELoss()
optimiser= torch.optim.Adam(cnn_to_train.parameters(), lr=0.003)

for epoch in range(50):
      output_cnn=cnn_to_train(noisy_torch_cnn)
      # Modify the target tensor size to match the output size
      target_cnn = true_torch.squeeze(1)

      loss_cnn=loss_function(output_cnn, target_cnn)
      loss_cnn.backward()#computes dloss/dx for every parameter x which has requires_grad=True
      optimiser.step()#does update
      optimiser.zero_grad()

      cnn_ouput=np.squeeze(output_cnn.detach().numpy())
      cv2.putText(cnn_ouput, 'Ep %d' % epoch, (2, 30), 0, 1, int(np.max(cnn_ouput)+1), 1, cv2.LINE_AA)
      cv2disp('Ouput after conv', cnn_ouput, 0, 0,resize=(400,400))
      cv2.waitKey(1)



# Convert the output of the trained kernel to numpy array
conv_trained_output_np = np.squeeze(conv_trained_kernel(noisy_torch).detach().numpy())

# Convert the output of the trained CNN to numpy array
cnn_trained_output_np = np.squeeze(cnn_to_train(noisy_torch).detach().numpy())


# Calculate MSE between the original and denoised images
mse_conv_trained = mean_squared_error(true_np.flatten(), conv_trained_output_np.flatten())
mse_cnn_trained = mean_squared_error(true_np.flatten(), cnn_trained_output_np.flatten())

#notes, on tennis(cnn1), conv better than cnn,0.0195,0.0266
#notes, on firetruck(cnn5), conv better than cnn, 0.0075, 0.0082
#notes, on cars(cnn2), cnn better than conv, 0.010,0.005
#notes, on fire hydrant(cnn3), cnn better than conv, 0.016, 0.012
#notes, on road sign (cnn4), cnn better than conv, 0.0095, 0.0055


# Print MSE values
print('MSE between original and denoised image using trained Convolutional Kernel:', mse_conv_trained)
print('MSE between original and denoised image using trained CNN:', mse_cnn_trained)



cv2.waitKey(0)
