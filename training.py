# ====================================================================================================================================== #
#                                                        CNN - IMAGE CLASSIFICATION                                                      #
# ====================================================================================================================================== #
# Author: Miguel Marines


# ====================================================================================================================================== #
#                                                               LIBRARIES                                                                #
# ====================================================================================================================================== #
import torch
import torch.nn as nn                       # Contains modules and classes to create and train neural networks.


from torch.utils.data import DataLoader     # Converts the dataset to an iterable object (to make the minibatches automatically).
from torch.utils.data import sampler        # Used to create random samples from the data.

import torchvision.datasets as datasets     # Contains the dataset and create dataset objects (with our own objects).

import torchvision.transforms as T          # Contains common image transformations.

from torch.optim import Adam                # Contains a method used for stochastic optimization.
from torch.autograd import Variable         # Contains classes and functions with automatic differentiation of arbitrary scalar valued functions.

import pathlib                              # Used to work with paths.
import glob                                 # Used to find all the pathnames matching a specified pattern.





# ====================================================================================================================================== #
#                                                           SELECT DEVICE TO USE                                                         #
# ====================================================================================================================================== #
# Select whether to use GPU or CPU.
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Print device to use.
print("\nDevice:", str(device))





# ====================================================================================================================================== #
#                                                      TRANSFORM AND NORMALIZE IMAGES                                                    #
# ====================================================================================================================================== #
# Transform images to tensors and normalize them. 
transformer =   T.Compose([
                T.Resize((150, 150)),                         # Resize the input image to the given size (H, W).
                T.ToTensor(),                                 # Convert image to tensor (Tensor: Multi-dimensional matrix containing elements of a single data type).
                T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) # Normalize a tensor image with mean and standard deviation. (Valores de media del RGB)
            ])





# ====================================================================================================================================== #
#                                                                   LOAD DATA                                                            #
# ====================================================================================================================================== #
# Training Path
Train_Path = "/Users/.../dataset/seg_train/seg_train"

# Testing Path
Test_Path = "/Users/.../dataset/seg_test/seg_test"


# Load train data and shuffle it.
train_loader = DataLoader(datasets.ImageFolder(Train_Path, transform = transformer), batch_size = 20, shuffle = True)

# Load test data and shuffle it.
test_loader = DataLoader(datasets.ImageFolder(Test_Path, transform = transformer), batch_size = 8, shuffle = True)





# ====================================================================================================================================== #
#                                                     GET CATEGORIES AND CLASSES                                                         #
# ====================================================================================================================================== #
# Get categories and classes.
root = pathlib.Path(Train_Path)
classes = sorted([j.name.split('/')[-1] for j in root.iterdir()])

# Remove .DS_Store in MAC from classes.
classes.pop(0)

# Print classes.
print("\n\nClases: ", end="")
print(', '.join(classes))





# ====================================================================================================================================== #
#                                                                   MODEL                                                                #
# ====================================================================================================================================== #
# In Channel: Depth of the input data.                                                                                                   #
#             Grayscale Image - Channel dimension 1, because there is only one color channel.                                            #
#             Color Image - Channel dimension would be 3, because we have three color channels (red, green, and blue).                   #
#                                                                                                                                        #
# Out Channel: Number of filters used in a particular convolutional layer of the network.                                                #
#                                                                                                                                        #
# Kernel: Determines the receptive area of the input data that each filter in the layer can "see".                                       #
#                                                                                                                                        #
# Stride: Number of pixels that the convolutional filter is shifted each time it is applied to the input image.                          #
#         When the stride value is set to 1, the filter moves one pixel at a time.                                                       #
#                                                                                                                                        #
# Padding: Extra border pixels to the input data.                                                                                        #
#          It is added to the input data before applying a convolutional operation, to ensure that the output size of the operation      #
#          matches the desired output size.                                                                                              #
#                                                                                                                                        #
# Shape = (Batch Size, Number of Channels, Height, Width)                                                                                #
#                                                                                                                                        #
# Output Size After Convolution Filter:                                                                                                  #
# ((w-f + 2P) / s) + 1     ->      ((width - kernel + (2 * padding)) / stride) + 1     ->      (150 - 3 + (2 * 1) / 1) + 1 = 150         #
# ====================================================================================================================================== #

# Convolutional Neural Network Class
class ConvuntionalNetwork(nn.Module):

    # ================================================================================================================ #
    #                                                     CONSTRTUCTOR                                                 #
    # ================================================================================================================ # 
    def __init__(self, num_classes = 5):

        super(ConvuntionalNetwork, self).__init__()
                
        # Input Shape = (256, 3, 150, 150)

        # 2D Convolution. --------------------------------------------------------------------------------------------
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 12, kernel_size = 3, stride = 1, padding = 1)
        # Shape = (256, 12, 150, 150)

        # Batch Normalization.
        self.bn1 = nn.BatchNorm2d(num_features = 12)
        # Shape = (256, 12, 150, 150)

        # Rectified Linear Unit Function.
        self.relu1 = nn.ReLU()
        # Shape = (256, 12, 150, 150)
        
        # 2D Max Pooling.
        self.pool = nn.MaxPool2d(kernel_size = 2)
        # Reduce the image size by factor 2.
        # Shape = (256, 12, 75, 75)
        

        # 2D Convolution. --------------------------------------------------------------------------------------------
        self.conv2 = nn.Conv2d(in_channels = 12, out_channels = 20, kernel_size = 3, stride = 1, padding = 1)
        # Shape = (256, 20, 75, 75)
       
        # Rectified Linear Unit Function.
        self.relu2 = nn.ReLU()
        # Shape = (256, 20, 75, 75)
        
        
        # 2D Convolution. --------------------------------------------------------------------------------------------
        self.conv3 = nn.Conv2d(in_channels = 20, out_channels = 32, kernel_size = 3, stride = 1, padding = 1)
        # Shape = (256, 32, 75, 75)
        
        # Batch Normalization.
        self.bn3 = nn.BatchNorm2d(num_features = 32)
        # Shape = (256, 32, 75, 75)

        # Rectified Linear Unit Function.
        self.relu3 = nn.ReLU()
        # Shape = (256, 32, 75, 75)
        
        # Fully Contected Layer
        # Linear Transformation
        # Input features =  (Hight, Width, Number of Channels) and output image of the convolutional layer output.
        self.fc = nn.Linear(in_features = 75 * 75 * 32, out_features = num_classes)
    
    
    # ================================================================================================================ #
    #                                               FEED FORWARD FUNCTION                                              #
    # ================================================================================================================ # 
    def forward(self, input):
        
        output = self.conv1(input)              # 2D Convolution.
        output = self.bn1(output)               # Batch Normalization.
        output = self.relu1(output)             # Rectified Linear Unit Function.
        
        output = self.pool(output)              # 2D Max Pooling.
        
        output = self.conv2(output)             # 2D Convolution.
        output = self.relu2(output)             # Rectified Linear Unit Function.
        
        output = self.conv3(output)             # 2D Convolution.
        output = self.bn3(output)               # Batch Normalization.
        output = self.relu3(output)             # Rectified Linear Unit Function.


        output = output.view(-1, 32 * 75 * 75)  # Output in matrix form. Shape (256, 32, 75, 75). Reshape matrix.
        output = self.fc(output)                # Linear Transformation.
        
        # Return output.
        return output





# ====================================================================================================================================== #
#                                                        STARTING TRAINING ELEMENTS                                                      #
# ====================================================================================================================================== #
# Create object of the class Convuntional Network and send it to the using device.
model = ConvuntionalNetwork(num_classes = 5).to(device)

# Optmizer - Adam Optimizer.
optimizer = Adam(model.parameters(), lr = 0.001, weight_decay = 0.0001)

# Loss Function - Cross Entropy
loss_function = nn.CrossEntropyLoss()

# Number of Epochs
num_epochs = 25


# Number of the training and testing images.
train_count = len(glob.glob(Train_Path + '/**/*.jpg'))
test_count = len(glob.glob(Test_Path + '/**/*.jpg'))

# Print number of the training and testing images.
print("\n\nTraining Images: " + str(train_count))
print("Testing Images: " + str(test_count))





# ====================================================================================================================================== #
#                                                   TRAINING MODEL AND SAVING BEST MODEL                                                 #
# ====================================================================================================================================== #
print("\n\n")
# Variable to save the best accuracy.
best_accuracy = 0.0

for epoch in range(num_epochs):
    
    # Evaluation of the training dataset. ---------------------------------------------------------------------------------------------
    # Keeps some layers like dropout, batch normalization, which behave differently depending on the current face.
    model.train()

    train_accuracy = 0.0
    train_loss = 0.0
    
    # Loop images and labes (For loop for the batches inside the train loaders).
    for i, (images, labels) in enumerate(train_loader):
        
        # If devise GPU is being used.
        if torch.cuda.is_available():
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())
            
        optimizer.zero_grad()   # Turn to 0 the gradients at the start of a new batch.
        
        outputs = model(images)                     # Prediction
        loss = loss_function(outputs, labels)       # Compute loss or error using predicted and actual value.
        loss.backward()                             # Back propagation.
        optimizer.step()                            # Update weights and bias based on the gradiants.
        
        
        # Calculate the loss.
        train_loss += loss.cpu().data * images.size(0)
        # Calculate the prediction.
        _,prediction = torch.max(outputs.data,1)
        
        # Calculate train accuracy.
        train_accuracy += int(torch.sum(prediction == labels.data))

    # Final training accuracy for a particular epoch.
    train_accuracy = train_accuracy / train_count
    # Final training loss for a particular epoch.
    train_loss = train_loss / train_count
    
    
    # Evaluation of the testing dataset. ----------------------------------------------------------------------------------------------
    model.eval()
    
    test_accuracy = 0.0

    # Loop images and labes (For loop for the batches inside the train loaders). 
    for i, (images, labels) in enumerate(test_loader):

        # If devise GPU is being used.
        if torch.cuda.is_available():
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())
        
        # Make predictions.
        outputs = model(images)
        # Get category id.
        _,prediction = torch.max(outputs.data,1)
        # Calculate test accuracy.
        test_accuracy += int(torch.sum(prediction == labels.data))
    
    # Final testing accuracy for a particular epoch.
    test_accuracy = test_accuracy / test_count
    

    # Print Results -------------------------------------------------------------------------------------------------------------------
    print('Epoch: ' + str(epoch) + ' Train Accuracy: ' + str(train_accuracy) + ' Test Accuracy: ' + str(test_accuracy))
    

    # Save Best Model -----------------------------------------------------------------------------------------------------------------
    if (test_accuracy > best_accuracy):
        torch.save(model.state_dict(), '/Users/.../CNN/best_checkpoint.model')
        best_accuracy = test_accuracy