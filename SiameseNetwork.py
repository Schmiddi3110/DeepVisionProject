import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

class SiameseNetwork(nn.Module):
    """
    Siamese Network using a pre-trained ResNet model for feature extraction and custom fully connected layers.

    Methods:
        forward_once(x): Passes an input through the network once to get the embedding.
        forward(input1, input2): Passes two inputs through the network to get their embeddings.
        evaluate(validation_loader, net, criterion): Evaluates the network on the validation set.
        train_network(train_loader, val_loader, net, optimizer, criterion, epochs, patience=5): Trains the network with early stopping.
        show_plot(counter, loss_history, val_loss_history): Plots training and validation loss histories.
    """

    def __init__(self):
        """
        Initializes the SiameseNetwork with a pre-trained ResNet18 model and custom fully connected layers.
        """
        super(SiameseNetwork, self).__init__()

        # Using a pre-trained ResNet model
        self.resnet = models.resnet18(weights = models.ResNet18_Weights.IMAGENET1K_V1)
        
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
        """
        Passes an input through the network once to get the embedding.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output embedding.
        """
        output = self.resnet(x)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        """
        Passes two inputs through the network to get their embeddings.

        Args:
            input1 (torch.Tensor): First input tensor.
            input2 (torch.Tensor): Second input tensor.

        Returns:
            tuple: A tuple containing the embeddings of the two inputs.
        """
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        return output1, output2

    def evaluate(self, validation_loader, net, criterion, device):
        """
        Evaluates the network on the validation set.

        Args:
            validation_loader (DataLoader): DataLoader for the validation set.
            net (nn.Module): The Siamese network.
            criterion (nn.Module): Loss function.

        Returns:
            float: The average validation loss.
        """
        net.eval()
        val_loss = 0.0
        with torch.no_grad():
            for img0, img1, label in validation_loader:
                img0, img1, label = img0.to(device), img1.to(device), label.to(device)
                output1, output2 = net(img0, img1)
                loss_contrastive = criterion(output1, output2, label)
                val_loss += loss_contrastive.item()
        return val_loss / len(validation_loader)

    def train_network(self, train_loader, val_loader, net, optimizer, criterion, epochs,device, log_dir='./logs', patience=5):
        """
        Trains the network with early stopping.

        Args:
            train_loader (DataLoader): DataLoader for the training set.
            val_loader (DataLoader): DataLoader for the validation set.
            net (nn.Module): The Siamese network.
            optimizer (torch.optim.Optimizer): Optimizer for training.
            criterion (nn.Module): Loss function.
            epochs (int): Number of epochs to train for.
            log_dir (str, optional): Directory to save TensorBoard logs. Defaults to './logs'.
            patience (int, optional): Number of epochs to wait for improvement before stopping. Defaults to 5.
        """
        writer = SummaryWriter(log_dir=log_dir)
        counter = []
        loss_history = [] 
        val_loss_history = []
        iteration_number = 0

        # Early stopping variables
        best_val_loss = float('inf')
        epochs_without_improvement = 0

        # Iterate through the epochs
        for epoch in tqdm(range(epochs), desc=f"Training the model for {epochs} epochs"):
            net.train()
            running_loss = 0.0
            # Iterate over batches with tqdm progress bar
            for i, (img0, img1, label) in enumerate(train_loader):

                # Send the images and labels to CUDA                
                img0, img1, label = img0.to(device), img1.to(device), label.to(device)

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

                running_loss += loss_contrastive.item()

                # Every 10 batches, print out the loss and evaluate on validation set
                if i % 10 == 0:
                    iteration_number += 10
                    counter.append(iteration_number)
                    loss_history.append(loss_contrastive.item())
                    writer.add_scalar('Training Loss', loss_contrastive.item(), iteration_number)

                    val_loss = self.evaluate(val_loader, net, criterion, device)
                    val_loss_history.append(val_loss)
                    writer.add_scalar('Validation Loss', val_loss, iteration_number)
                    
                    #print(f"Epoch number {epoch}\n Current loss {loss_contrastive.item()}\n Validation loss: {val_loss}\n")
                    
                    # Check for early stopping
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        epochs_without_improvement = 0
                    else:
                        epochs_without_improvement += 1

                    if epochs_without_improvement >= patience:
                        #print(f"Early stopping at epoch {epoch} after {epochs_without_improvement} epochs without improvement.")
                        self.show_plot(counter, loss_history, val_loss_history)
                        writer.close()
                        return

            # Log average loss for the epoch
            avg_loss = running_loss / len(train_loader)
            writer.add_scalar('Average Training Loss', avg_loss, epoch)
            #print(f"Average Training Loss after epoch {epoch}: {avg_loss}")

        self.show_plot(counter, loss_history, val_loss_history)
        writer.close()

    def show_plot(self, counter, loss_history, val_loss_history):
        """
        Plots training and validation loss histories.

        Args:
            counter (list): List of iteration numbers.
            loss_history (list): List of training loss values.
            val_loss_history (list): List of validation loss values.
        """
        plt.figure()
        plt.plot(counter, loss_history, label='Training Loss')
        plt.plot(counter, val_loss_history, label='Validation Loss')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
