import torch
from torch import nn, optim
from torchvision import datasets, transforms

from logger import Logger

num_epochs = 5
num_classes = 10
batch_size = 100
learning_rate = 1e-3

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
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)

        return out
    
model = ConvNet(num_classes)
logger = Logger() # instantiate logger

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for e in range(num_epochs):
    logger.train(len(trainloader)) # tell logger for training
    for images, labels in trainloader:

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            correct = torch.argmax(outputs.data, 1) == labels
            logger(model, loss.cpu(), correct.cpu(), learning_rate) # progress recording

    model.eval()
    logger.eval(len(testloader)) # tell logger for evaluating
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in testloader:

            outputs = model(images)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels)
            logger(model, loss.cpu(), correct.cpu()) # progress recording
            
logger.flush() # clearing outputs