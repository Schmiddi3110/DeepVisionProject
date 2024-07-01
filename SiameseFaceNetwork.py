import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import matplotlib.pyplot as plt

class SiameseFaceNetwork(nn.Module):
    def __init__(self):
        super(SiameseFaceNetwork, self).__init__()

        # Using a pre-trained ResNet model
        self.resnet = models.resnet18(pretrained=True)
        
        # Modify the first convolutional layer to accept 1-channel input
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
        # Get the number of input features for the last fully connected layer
        num_ftrs = self.resnet.fc.in_features
        
        # Remove the final fully connected layer
        self.resnet.fc = nn.Identity()
        
        # Add custom fully connected layers
        self.fc1 = nn.Sequential(
            nn.Linear(num_ftrs, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.5),
            
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            
            nn.Linear(256, 2)
        )
        
    def forward_once(self, x):
        # This function will be called for both images
        # Its output is used to determine the similarity
        output = self.resnet(x)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        # In this function we pass in both images and obtain both vectors
        # which are returned
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        return output1, output2

    def evaluate(self, validation_loader, net, criterion):
        net.eval()
        val_loss = 0.0
        with torch.no_grad():
            for img0, img1, label in validation_loader:
                img0, img1, label = img0.cuda(), img1.cuda(), label.cuda()
                output1, output2 = net(img0, img1)
                loss_contrastive = criterion(output1, output2, label)
                val_loss += loss_contrastive.item()
        return val_loss / len(validation_loader)

    def train_network(self, train_loader, val_loader, net, optimizer, criterion, epochs, patience=5):
        counter = []
        loss_history = [] 
        val_loss_history = []
        iteration_number = 0

        # Early stopping variables
        best_val_loss = float('inf')
        epochs_without_improvement = 0

        # Iterate through the epochs
        for epoch in range(epochs):

            net.train()
            # Iterate over batches
            for i, (img0, img1, label) in enumerate(train_loader, 0):

                # Send the images and labels to CUDA
                img0, img1, label = img0.cuda(), img1.cuda(), label.cuda()

                # Zero the gradients
                optimizer.zero_grad()

                # Pass in the two images into the network and obtain two outputs
                output1, output2 = net(img0, img1)

                # Pass the outputs of the networks and label into the loss function
                loss_contrastive = criterion(output1, output2, label)

                # Calculate the backpropagation
                loss_contrastive.backward()

                # Optimize
                optimizer.step()

                # Every 10 batches, print out the loss and evaluate on validation set
                if i % 10 == 0:
                    print(f"Epoch number {epoch}\n Current loss {loss_contrastive.item()}\n")
                    iteration_number += 10

                    counter.append(iteration_number)
                    loss_history.append(loss_contrastive.item())

                    val_loss = self.evaluate(val_loader, net, criterion)
                    val_loss_history.append(val_loss)
                    print(f"Validation loss after {iteration_number} iterations: {val_loss}\n")
                    
                    # Check for early stopping
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        epochs_without_improvement = 0
                    else:
                        epochs_without_improvement += 1

                    if epochs_without_improvement >= patience:
                        print(f"Early stopping at epoch {epoch} after {epochs_without_improvement} epochs without improvement.")
                        self.show_plot(counter, loss_history, val_loss_history)
                        return

        self.show_plot(counter, loss_history, val_loss_history)

    # Function to plot the loss history
    def show_plot(self, counter, loss_history, val_loss_history):
        plt.figure()
        plt.plot(counter, loss_history, label='Training Loss')
        plt.plot(counter, val_loss_history, label='Validation Loss')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
