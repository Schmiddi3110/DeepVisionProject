import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class ForgeryNetwork(nn.Module):
    def __init__(self):
        super(ForgeryNetwork, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1,64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.Sigmoid(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.Sigmoid(),
            nn.MaxPool2d(2,2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(128),
            nn.Sigmoid(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128,128, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(128),
            nn.Sigmoid(),
            nn.MaxPool2d(2,2),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(128,256, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(256),
            nn.Sigmoid(),
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(256,256, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(256),
            nn.Sigmoid(),
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(256,256, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(256),
            nn.Sigmoid(),
            nn.MaxPool2d(2,2),
        )
        self.conv8 = nn.Sequential(
             nn.Conv2d(256,512, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(512),
            nn.Sigmoid(),
        )
        self.conv9 = nn.Sequential(
            nn.Conv2d(512,512, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(512),
            nn.Sigmoid(),
        )
        self.conv10 = nn.Sequential(
            nn.Conv2d(512,512, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(512),
            nn.Sigmoid(),
            nn.MaxPool2d(2,2),
        )
        self.conv11 = nn.Sequential(
            nn.Conv2d(512,512, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(512),
            nn.Sigmoid(),
        )
        self.conv12 = nn.Sequential(
            nn.Conv2d(512,512, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(512),
            nn.Sigmoid(),
        )
        self.conv13 = nn.Sequential(
            nn.Conv2d(512,512, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(512),
            nn.Sigmoid(),
            nn.MaxPool2d(2,2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7,7))
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(25088, 4096),
            nn.Sigmoid(), 
        )
        self.fc2 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096,4096),
            nn.Sigmoid(),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(4096, 100),
        )
    def forward_once(self,x):
        output = self.conv1(x)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        output = self.conv5(output)
        output = self.conv6(output)
        output = self.conv7(output)
        output = self.conv8(output)
        output = self.conv9(output)
        output = self.conv10(output)
        output = self.conv11(output)
        output = self.conv12(output)
        output = self.conv13(output)
        output = self.avgpool(output)
        output = torch.flatten(output, 1)
        output = self.fc1(output)
        output = self.fc2(output)
        output = self.fc3(output)
        return output
    
    def forward(self,input1, input2):
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