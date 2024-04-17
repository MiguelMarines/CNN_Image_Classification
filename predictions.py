# ====================================================================================================================================== #
#                                                        CNN - IMAGE CLASSIFICATION                                                      #
# ====================================================================================================================================== #
# Author: Miguel Marines


# ====================================================================================================================================== #
#                                                               LIBRARIES                                                                #
# ====================================================================================================================================== #
import torch
import torch.nn as nn                       # Contains modules and classes to create and train neural networks.

import torchvision.transforms as T          # Contains common image transformations.

from torch.autograd import Variable         # Contains classes and functions with automatic differentiation of arbitrary scalar valued functions.

import pathlib                              # Used to work with paths.
import glob                                 # Used to find all the pathnames matching a specified pattern.

from PIL import Image                       # Provides a class with the same name which is used to represent a PIL image.





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
#                                                                   PATHS                                                                #
# ====================================================================================================================================== #
# Training Path
Train_Path = "/Users/.../dataset/seg_train"

# Prediction Path
Prediction_Path = "/Users.../dataset/seg_pred"





# ====================================================================================================================================== #
#                                                     GET CATEGORIES AND CLASSES                                                         #
# ====================================================================================================================================== #
# Get categories and classes.
root = pathlib.Path(Train_Path)
classes = sorted([j.name.split('/')[-1] for j in root.iterdir()])

# Remove .DS_Store in MAC from classes.
classes.pop(0)

# Print classes.
# print("\n\nClases: ", end="")
# print(', '.join(classes))





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
#                                                                  CHECK IMAGES                                                          #
# ====================================================================================================================================== #
# Path of the best model obtained from the training.
checkpoint = torch.load('/Users/.../CNN/best_checkpoint.model')

# Load best model from the training and testing.
model = ConvuntionalNetwork(num_classes = 5)    # Create CNN model with the number of classes.
model.load_state_dict(checkpoint)               # Feed the checkpoint inside the model.
model.eval()                                    # Evaluation mode to set droput and batch normalization to evaluation mode and get consisten results.





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
#                                                            PREDICTION FUNCTION                                                         #
# ====================================================================================================================================== #
def prediction(img_path, transformer):
    
    image = Image.open(img_path)
    
    image_tensor = transformer(image).float()
    
    image_tensor = image_tensor.unsqueeze_(0)
    
    # If devise GPU is being used.
    if torch.cuda.is_available():
        image_tensor.cuda()
        
    input = Variable(image_tensor)
    
    output = model(input)
    
    index = output.data.numpy().argmax()
    
    pred = classes[index]
    
    return pred





# Variable to store the common path for the images.
images_path = glob.glob(Prediction_Path + '/*.jpg')

# Dictionary to store predictions.(Image name key and prediction)
pred_dict = {}

# Save in dictionary image name key and prediction.
for i in images_path:
    pred_dict[i[i.rfind('/') + 1:]] = prediction(i, transformer)

# Print dictionary with predictions.
for name in pred_dict.keys():
    # Convert elements to string.
    prediction = [str(pred_dict[name])]
    # Print results.
    print(f"{name} -> {prediction}")