import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from src.data.data_transformation import make_transformation
from PIL import Image
import os

class CustomData(Dataset):
    def __init__(self, root_dir, transform=make_transformation()):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = os.listdir(root_dir)
        self.images_path = []
        self.labels = []

        # Valid image extensions
        valid_extensions = ['.jpg', '.jpeg', '.png']

        for label, class_name in enumerate(self.classes):
            class_path = os.path.join(self.root_dir, class_name)
            if not os.path.isdir(class_path):
                continue  # Skip non-directory files
            # Get all files in the class directory
            for image_name in os.listdir(class_path):
                if any(image_name.endswith(ext) for ext in valid_extensions):
                    self.images_path.append(os.path.join(class_path, image_name))
                    self.labels.append(label)

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, idx):
        img_path = self.images_path[idx]
        image = Image.open(img_path).convert('RGB')  # Ensuring the image is in RGB mode

        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


class GetData:
    def __init__(self, train_data_path, test_data_path):
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path

    def load_data(self):
        # Use the correct paths for train and test data
        train_data = CustomData(root_dir=self.train_data_path)
        test_data = CustomData(root_dir=self.test_data_path)

        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

        return train_loader, test_loader


if __name__ == "__main__":
    data_loader = GetData(train_data_path='data/raw/chest_xray/train',
                          test_data_path='data/raw/chest_xray/test')

    train_loader, test_loader = data_loader.load_data()

    # Example of iterating through the data loader to check the image and batch shape
    for images, labels in train_loader:
        print(f"Batch size: {images.shape[0]}, Image shape: {images.shape}, Labels: {labels}")
        break  # Just to verify a single batch
