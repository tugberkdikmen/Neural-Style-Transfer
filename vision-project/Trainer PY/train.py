import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models


# Set random seed for reproducibility
torch.manual_seed(42)

# Set device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transforms for data preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
])

# Load the dataset
train_dataset = datasets.ImageFolder('selected', transform=transform)
valid_dataset = datasets.ImageFolder('selected', transform=transform)

# Create data loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=32, shuffle=False)

# Load pre-trained VGG-19 model
model = models.vgg19(pretrained=True)
model.to(device)

# Replace the last fully connected layer for your number of classes
num_classes = 2
model.classifier[6] = nn.Linear(4096, num_classes)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

# Training loop
num_epochs = 3

for epoch in range(num_epochs):
    train_loss = 0.0
    valid_loss = 0.0
    correct = 0
    total = 0

    model.train()  # Set model to training mode
    print ('mod train comp!')

    i = 0
    l = 1 
    for images, labels in train_loader:
        print("images", i, 'label', l )
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)

        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        print ('done')
        l+=1
        i+=1

    print ('train comp!')
    # Validation loop
    model.eval()  # Set model to evaluation mode
    print ('mod eval comp!')

    with torch.no_grad():
        for images, labels in valid_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            valid_loss += loss.item() * images.size(0)

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    print ('eval comp!')

    # Calculate average losses and accuracy
    train_loss /= len(train_loader.dataset)
    valid_loss /= len(valid_loader.dataset)
    accuracy = 100.0 * correct / total

    # Print training statistics
    print(f"Epoch: {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Valid Loss: {valid_loss:.4f} | Accuracy: {accuracy:.2f}%")
    
    # Save the trained model
    save_path = "saved/trained_model6.pth"
    torch.save(model.state_dict(), save_path)