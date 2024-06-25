import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class SignatureNet(nn.Module):
    def __init__(self):
        super(SignatureNet, self).__init__()
        # Setting up the Sequential of CNN Layers
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=11, stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            nn.BatchNorm2d(96),
            
            nn.Conv2d(96, 256, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.BatchNorm2d(256),

            nn.Conv2d(256, 384, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(384)
        )
        
        # Setting up the Fully Connected Layers
        self.fc1 = nn.Sequential(
            nn.Linear(384, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            
            nn.Linear(256, 2),
            nn.Sigmoid()
        )

        self.fc_forgery = nn.Sequential(
            nn.Linear(384, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),

            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(),

            nn.Linear(256, 1),
            nn.ReLU(inplace=True)
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
    def similarity_loss(self, output1, output2, label, margin=5.0):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                    (label) * torch.pow(torch.clamp(margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive

    def forgery_loss(self, prediction, target):
        target = target.float().view(-1, 1)  
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
                input1, input2 = input1.cuda(), input2.cuda()
                forgery_label1 = forgery_label1.cuda().float().unsqueeze(1)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                output1, output2, forgery_output1 = net(input1, input2)
                
                # Calculate losses
                loss_sim = self.similarity_loss(output1, output2, forgery_label1).mean()
                loss_forgery = self.forgery_loss(forgery_output1, forgery_label1).mean()
                
                # Total loss
                total_loss = loss_sim + 5*loss_forgery
                sim_loss_history.append(loss_sim.item())
                forgery_loss_history.append(loss_forgery.item())
                
                # Backward pass and optimize
                total_loss.backward()
                optimizer.step()
                
                # Every 10 batches, print out the loss and evaluate on validation set
                if i % 10 == 0:
                    print(f"Epoch number {epoch}\n Current loss {total_loss.item()}\n")
                    iteration_number += 10

                    counter.append(iteration_number)
                    total_loss_history.append(total_loss.item())

                    val_loss = self.evaluate(val_loader, net)
                    val_loss_history.append(val_loss)
                    print(f"Validation loss after {iteration_number} iterations: {val_loss}\n")

        self.show_plot(counter, total_loss_history, val_loss_history)

    def evaluate(self, validation_loader, net):
        net.eval()
        val_loss = 0.0
        with torch.no_grad():
            for img0, img1, forgery_label in validation_loader:
                img0, img1 = img0.cuda(), img1.cuda()
                forgery_label = forgery_label.cuda().float().unsqueeze(1)
                output1, output2, forgery = net(img0, img1)

                loss_sim = self.similarity_loss(output1, output2, forgery_label).mean()
                loss_forgery = self.forgery_loss(forgery, forgery_label).mean()
                
                total_loss = loss_sim + 5*loss_forgery
                val_loss += total_loss.item()
        return val_loss / len(validation_loader)


    def show_plot(self, counter, total_loss_history, val_loss_history):
        plt.figure(figsize=(10,5))
        plt.plot(counter, total_loss_history, label='Total Loss')
        plt.plot(counter, val_loss_history, label='Validation Loss')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()


    def compute_accuracy(self, test_loader, net, threshold=0.5):
            net.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for img0, img1, forgery_label in test_loader:
                    img0, img1 = img0.cuda(), img1.cuda()
                    forgery_label = forgery_label.cuda().float().unsqueeze(1)
                    
                    # Forward pass
                    _, _, forgery = net(img0, img1)                    
                    
                    # Apply threshold to get binary predictions
                    forgery_preds = (forgery >= threshold).float()
                    
                    # Count correct predictions
                    correct += (forgery_preds == forgery_label).sum().item()
                    total += forgery_label.size(0)
            
            accuracy = correct / total
            return accuracy
    

    def predict(self, input1, input2):
        """
        Predict the similarity between input1 and input2 and return the forgery prediction.
        
        Args:
            input1: The first input image tensor.
            input2: The second input image tensor.
        
        Returns:
            similarity: The similarity between the input images.
            forgery_prediction: The probability that input2 is a forgery.
        """
        self.eval()  # Set the network to evaluation mode
        
        with torch.no_grad():
            input1, input2 = input1.cuda(), input2.cuda()
            
            output1 = self.forward_once(input1)
            output2, forgery_output = self.forward_once(input2, True)
            
            # Calculate similarity using Euclidean distance
            similarity = F.pairwise_distance(output1, output2, keepdim=True)
            forgery_prediction = (forgery_output >= 0.5).float()
        
        return similarity.cpu(), forgery_prediction.cpu()