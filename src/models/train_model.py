# This Scripy is responsible for model training
from src.data.data_loader import GetData
import torchvision.models as models
from dotenv import load_dotenv
import mlflow.pytorch
from torch import nn
# Modify the training loop
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import mlflow
import torch
import warnings
import os

scaler = GradScaler()
warnings.filterwarnings('ignore')
load_dotenv()
dagshub_token = os.getenv('DAGSHUB_TOKEN')

if dagshub_token:
    os.environ['MLFlow_TRACKING_USERNAME']=dagshub_token
    os.environ['MLFlow_TRACKING_PASSWORD']=dagshub_token
    # Set up the MLflow tracking URI with authentication using the token
    mlflow.set_tracking_uri(f'https://{dagshub_token}:@dagshub.com/Sami606713/phenomenia_detection.mlflow')

    print("DagsHub login successful!")
else:
    print("DagsHub token not found. Please set the DAGSHUB_TOKEN environment variable.")

# base_model
class MobileNet(nn.Module):
    def __init__(self):
        super(MobileNet, self).__init__()
        self.base_model=models.mobilenet_v2(weights='IMAGENET1K_V1')
    
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
def train_model(model, train_loader, val_loader, nbr_of_epochs, optimizer, loss_fn):
    """
    This script is responsible for training the model and saving checkpoints when the model is not overfitting.
    """
    try:
        early_stopping_patience = 5  # Stop if val_loss doesn't improve after 5 epochs
        no_improvement_epochs = 0
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        best_val_loss = float('inf')  # Initialize with infinity to keep track of the best validation loss
        
        # Start MLflow run
        with mlflow.start_run(run_name="mobilenet_v2") as run:
            mlflow.log_param("learning_rate", optimizer.param_groups[0]['lr'])
            mlflow.log_param("weight_decay", optimizer.param_groups[0]['weight_decay'])
            mlflow.log_param("epochs", nbr_of_epochs)
            
            for epoch in range(nbr_of_epochs):
                train_loss = 0.0
                train_correct = 0
                train_total = 0
                model.train()  # Ensure the model is in training mode
                
                for image, label in tqdm(train_loader):
                    image, label = image.to(device), label.to(device)
                    
                    with autocast():
                    # Forward pass
                        train_score = model(image)

                        # Calculate loss
                        loss = loss_fn(train_score, label)

                    # Zero the gradient
                    optimizer.zero_grad()

                    # Backward pass and optimize
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    # Update the running loss and accuracy
                    train_loss += loss.item()
                    _, predicted = torch.max(train_score.data, 1)
                    train_total += label.size(0)
                    train_correct += (predicted == label).sum().item()

                # Calculate average loss and accuracy
                train_average_loss = train_loss / len(train_loader)
                train_accuracy = train_correct / train_total

                # Log training metrics to MLflow
                mlflow.log_metric("train_loss", train_average_loss, step=epoch)
                mlflow.log_metric("train_accuracy", train_accuracy, step=epoch)

                # Validation phase to check for overfitting
                test_loss = 0.0
                test_correct = 0
                test_total = 0
                model.eval()  # Switch the model to evaluation mode
                with torch.no_grad():
                    for images, labels in val_loader:
                        images = images.to(device)
                        labels = labels.to(device)

                        # Forward pass
                        outputs = model(images)
                        loss = loss_fn(outputs, labels)

                        # Update the running loss
                        test_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)

                        test_total += labels.size(0)
                        test_correct += (predicted == labels).sum().item()

                val_average_loss = test_loss / len(val_loader)
                test_accuracy = test_correct / test_total

                # Log validation metrics to MLflow
                mlflow.log_metric("val_loss", val_average_loss, step=epoch)

                print(f"Epoch [{epoch+1}/{nbr_of_epochs}], Train Loss: {train_average_loss:.4f}, Val Loss: {val_average_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Test accuracy: {test_accuracy:.4f}")

                # Save the checkpoint if the validation loss improves
                if val_average_loss < best_val_loss:
                    print(f"Validation loss improved ({best_val_loss:.4f} -> {val_average_loss:.4f}), saving model checkpoint...")
                    best_val_loss = val_average_loss  # Update the best validation loss
                    no_improvement_epochs = 0

                    save_checkpoint({
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch
                    })

                    # Log and register the model with MLflow
                    mlflow.pytorch.log_model(model, "best_model")  # Log the model
                    print(f"Model logged to MLflow at run {run.info.run_id}")
                else:
                    no_improvement_epochs += 1  # Increment counter if no improvement
                    if no_improvement_epochs >= early_stopping_patience:
                        print(f"Early stopping at epoch {epoch} due to no improvement in validation loss.")
                        break

            # Register the model at the end of training
            model_uri = f"runs:/{run.info.run_id}/best_model"
            registered_model_name = "MobileNetV2_Chest_XRay_Classifier"
            mlflow.register_model(model_uri, registered_model_name)
            print(f"Model registered under the name '{registered_model_name}'")

    except Exception as e:
        print(str(e))

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

    train_loader, val_loader= data_loader.load_data()

    # train model
    train_model(train_loader=train_loader,val_loader=val_loader,nbr_of_epochs=3,model=model,
                loss_fn=loss_fn,optimizer=optimizer)