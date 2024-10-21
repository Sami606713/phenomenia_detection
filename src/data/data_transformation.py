from torchvision import transforms

def make_transformation():
    transform=transforms.Compose([
        transforms.Resize((224, 224)),  # Resize all images to 224x224
        transforms.RandomRotation(degrees=40),  # Randomly rotate images within 40 degrees
        transforms.RandomHorizontalFlip(p=0.5),  # Flip the image horizontally with a probability of 0.5
        transforms.RandomResizedCrop(224),  # Randomly crop and resize images to 224x224
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # Apply random brightness, contrast, etc.
        transforms.ToTensor(),  # Convert to Tensor
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # Normalize based on ImageNet stats
    ])

    return transform