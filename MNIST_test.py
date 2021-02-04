import torch
from torch import nn, optim
from torchvision import datasets, transforms

from logger import Logger

# Hyperparameters
num_epochs = 5
num_classes = 10
batch_size = 100
learning_rate = 1e-3

# MNIST dataset
trainset = datasets.MNIST(
    root="../Data/", 
    train=True, 
    transform=transforms.ToTensor(),
    download=True
)

testset = datasets.MNIST(
    root="../Data/",
    train=False,
    transform=transforms.ToTensor(),
)

# Data loader
trainloader = torch.utils.data.DataLoader(
    dataset=trainset, 
    batch_size=batch_size, 
    shuffle=True
)

testloader = torch.utils.data.DataLoader(
    dataset=testset,
    batch_size=batch_size,
    shuffle=True
)

# Convolution Neural Network
class ConvNet(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Linear(7*7*32, num_classes)

    def forward(self, X):

        out = self.layer1(X)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1) # Flatten
        out = self.fc(out)

        return out
    
model = ConvNet(num_classes)
logger = Logger()

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training
for e in range(num_epochs):
    logger.train(len(trainloader))
    # print("Epoch {}/{}".format(e, num_epochs), end='.')
    for images, labels in trainloader:

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Getting accuracy
        with torch.no_grad():
            correct = torch.argmax(outputs.data, 1) == labels
            logger(model, loss.cpu(), correct.cpu(), learning_rate)

    # Testing
    model.eval()
    logger.eval(len(testloader))
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in testloader:

            outputs = model(images)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels)
            logger(model, loss.cpu(), correct.cpu())
            
logger.flush()