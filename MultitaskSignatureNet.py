import torch
import torch.nn as nn
import torch.nn.functional as F

class SignatureNet(nn.Module):
    def __init__(self):
        super(SignatureNet, self).__init__()
        # Setting up the Sequential of CNN Layers
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=11, stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            
            nn.Conv2d(96, 256, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(256, 384, kernel_size=3, stride=1),
            nn.ReLU(inplace=True)
        )
        
        # Setting up the Fully Connected Layers
        self.fc1 = nn.Sequential(
            nn.Linear(384, 1024),
            nn.ReLU(inplace=True),
            
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            
            nn.Linear(256, 2)
        )

        self.fc_forgery = nn.Sequential(
            nn.Linear(384, 1024),
            nn.ReLU(inplace=True),

            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),

            nn.Linear(256, 1),
        )
        

    def forward_once(self, x, forge_head=False):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output1 = self.fc1(output)

        if forge_head:
            output_forged = self.fc_forgery(output)
            return output1, output_forged
        
        return output1

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2, output_forge = self.forward_once(input2, True)

        return output1, output2, output_forge

    # Loss functions
    def similarity_loss(self, output1, output2):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        return euclidean_distance

    def forgery_loss(self, prediction, target):
        target = target.float().view(-1, 1)  # Ensure target is shaped correctly
        loss = F.binary_cross_entropy_with_logits(prediction, target)
        return loss


    def train_network(self, train_loader, val_loader, net, optimizer, epochs):
        # Training loop
        counter = []
        sim_loss_history = []
        forgery_loss_history = []
        val_loss_history = []
        total_loss_history = []
        iteration_number = 0

        for epoch in range(epochs):
            net.train()

            for i, (input1, input2, forgery_label1) in enumerate(train_loader):
                forgery_label1 = forgery_label1.float().unsqueeze(1)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                output1, output2, forgery_output1 = net(input1, input2)
                
                # Calculate losses
                loss_sim = self.similarity_loss(output1, output2).mean()
                loss_forgery = self.forgery_loss(forgery_output1, forgery_label1).mean()
                
                # Total loss
                total_loss = loss_sim + loss_forgery
                sim_loss_history.append(loss_sim)
                forgery_loss_history.append(loss_forgery)
                total_loss_history.append(total_loss)
                
                # Backward pass and optimize
                total_loss.backward()
                optimizer.step()
                
                # Every 10 batches, print out the loss and evaluate on validation set
                if i % 10 == 0:
                    print(f"Epoch number {epoch}\n Current loss {total_loss}\n")
                    iteration_number += 10

                    counter.append(iteration_number)

                    val_loss = self.evaluate(val_loader, net)
                    val_loss_history.append(val_loss.item())
                    print(f"Validation loss after {iteration_number} iterations: {val_loss}\n")

        self.show_plot(counter, total_loss_history, sim_loss_history, forgery_loss_history, val_loss_history)

    def evaluate(self, validation_loader, net):
        net.eval()
        val_loss = 0.0
        with torch.no_grad():
            for img0, img1, forgery_label in validation_loader:
                output1, output2, forgery = net(img0, img1)

                loss_sim = self.similarity_loss(output1, output2)
                loss_forgery = self.forgery_loss(forgery, forgery_label)
                
                total_loss = loss_sim + loss_forgery
                val_loss += total_loss
        return val_loss / len(validation_loader)

    def show_plot(self, counter, total_loss_history, sim_loss_history, forgery_loss_history, val_loss_history):
        # Implement plotting logic here if needed
        pass
