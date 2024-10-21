# This Scripy is responsible for model training
from src.data.data_loader import GetData
import torchvision.models as models
from tqdm import tqdm
from torch import nn
import torch

# Load the model
base_model= models.mobilenet_v2(pretrained=True)

# base_model

class MobileNet(nn.Module):
    def __init__(self):
        super(MobileNet, self).__init__()
        self.base_model=models.mobilenet_v2(pretrained=True)
    
    def model(self):
        # freeze the layer
        for layers in self.base_model.parameters():
            layers.requires_grad=False
        
        # Modify the layer
        # Chaneg the last layers
        self.base_model.classifier=nn.Sequential(
            nn.Linear(1280, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2)
        )
        
        return self.base_model

# save checkpoint
def save_checkpoint(state, filename="Models/my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

# training model
def train_model(model,train_loader,nbr_of_epochs,optimizer,loss_fn):
    """
    This script is responsible for training the model
    """
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
        for epoch in range(nbr_of_epochs):
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            for image,label in tqdm(train_loader):
                # set the model in train mode
                model = model.to(device)
                model.train()
                # set the model ,and data to target device
                image,label=image.to(device), label.to(device)
                # do forward pass
                train_score=model(image)

                # calculate the loss
                loss=loss_fn(train_score,label)

                # set the gradient ot 0
                optimizer.zero_grad()

                # do backward
                loss.backward()
                # do step
                optimizer.step()
                # Update the running loss
                train_loss += loss.item()
                _, predicted = torch.max(train_score.data, 1)
                train_total += label.size(0)
                train_correct += (predicted == label).sum().item()

             # Calculate average loss and accuracy
        train_average_loss = train_loss / len(train_loader)
        train_accuracy = train_correct / train_total
        
        if epoch%2==0:
            save_checkpoint({
                "state_dict":model.state_dict(),
                "optimizer":optimizer.state_dict(),
                "epoch":epoch
            })

            print(f"Epoch [{epoch+1}/{nbr_of_epochs}], Train Loss: {train_average_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
    except Exception as e:
        return str(e)

if __name__=="__main__":
    # load the model
    mobile_net=MobileNet()
    model=mobile_net.model()

    # load the optimizer and loss
    loss_fn = nn.CrossEntropyLoss()

    # Optimizer
    optimizer=torch.optim.Adam(model.parameters(),lr=0.001,weight_decay=0.01)

    # load data
    data_loader = GetData(train_data_path='data/raw/chest_xray/train',
                          test_data_path='data/raw/chest_xray/test')

    train_loader, _= data_loader.load_data()

    # train model
    train_model(train_loader=train_loader,nbr_of_epochs=1,model=model,
                loss_fn=loss_fn,optimizer=optimizer)