# Develop a Convolutional Deep Neural Network for Image Classification

### Name: SUJITH A
### Register Number:212224230278
## AIM
To develop a convolutional deep neural network (CNN) for image classification and to verify the response for new images.

##   PROBLEM STATEMENT AND DATASET
Include the Problem Statement and Dataset.

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1: 
Load and Preprocess Data

### STEP 2: 
Get the shape of the first image in the training dataset

### STEP 3: 
Get the shape of the first image in the test dataset

### STEP 4: 
Train the Model

### STEP 5: 
Test the Model

### STEP 6: 
Predict on a Single Image

### STEP 7:
Display the image


## PROGRAM



```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from torchsummary import summary



transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


train_set = torchvision.datasets.FashionMNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

test_set = torchvision.datasets.FashionMNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)


# Check dataset
im, lbl = train_set[0]

print("Image Shape:", im.shape)
print("Training Samples:", len(train_set))
print("Testing Samples:", len(test_set))



trl = DataLoader(
    train_set,
    batch_size=32,
    shuffle=True
)

tstl = DataLoader(
    test_set,
    batch_size=32,
    shuffle=False
)



class CNNclassifier1(nn.Module):

    def __init__(self):
        super().__init__()

        self.c1 = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        self.c2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        )

        self.c3 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            padding=1
        )

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        self.l1 = nn.Linear(
            128 * 3 * 3,
            64
        )

        self.l2 = nn.Linear(
            64,
            32
        )

        self.l3 = nn.Linear(
            32,
            10
        )

    def forward(self, x):

        x = self.pool(
            torch.relu(self.c1(x))
        )

        x = self.pool(
            torch.relu(self.c2(x))
        )

        x = self.pool(
            torch.relu(self.c3(x))
        )

        x = x.view(
            x.size(0),
            -1
        )

        x = torch.relu(self.l1(x))

        x = torch.relu(self.l2(x))

        x = self.l3(x)

        return x


model = CNNclassifier1()

criterion = nn.CrossEntropyLoss()

op = optim.Adam(
    model.parameters(),
    lr=0.001
)



device = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu'
)

print("Using Device:", device)

model.to(device)


summary(
    model,
    input_size=(1, 28, 28)
)


epochs = 3


for i in range(epochs):

    model.train()

    running_loss = 0.0

    for a, b in trl:

        # Move data to device
        a = a.to(device)
        b = b.to(device)

        # Clear previous gradients
        op.zero_grad()

        # Forward pass
        pred = model(a)

        # Calculate loss
        loss = criterion(pred, b)

        # Backpropagation
        loss.backward()

        # Update weights
        op.step()

        running_loss += loss.item()

    print(
        f"Epoch [{i + 1}/{epochs}], "
        f"Loss: {running_loss / len(trl):.4f}"
    )



t = 0
c = 0

act = []
pre = []

model.eval()

with torch.no_grad():

    for img, labels in tstl:

        img = img.to(device)
        labels = labels.to(device)

        output = model(img)

        _, predicted = torch.max(
            output,
            1
        )

        t += labels.size(0)

        c += (
            predicted == labels
        ).sum().item()

        pre.extend(
            predicted.cpu().numpy()
        )

        act.extend(
            labels.cpu().numpy()
        )



accuracy = c / t * 100

print(
    "Accuracy Score:",
    accuracy
)


conf_matrix = confusion_matrix(
    act,
    pre
)

print("Confusion Matrix:")
print(conf_matrix)

class_report = classification_report(
    act,
    pre,
    target_names=test_set.classes
)

print(
    "Classification Report:"
)

print(class_report)



plt.figure(figsize=(10, 8))

sns.heatmap(
    conf_matrix,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=test_set.classes,
    yticklabels=test_set.classes
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.show()


with torch.no_grad():

    img1, label = test_set[0]

    output = model(
        img1.unsqueeze(0).to(device)
    )

    _, pred = torch.max(
        output,
        1
    )

    classes = test_set.classes

    # Convert normalized image back to original range
    img1 = img1 * 0.5 + 0.5

    plt.imshow(
        img1.squeeze(),
        cmap="gray"
    )

    plt.title("Predicted Image")

    plt.axis('off')

    plt.show()

    print(
        f"Actual: {classes[label]}"
    )

    print(
        f"Predicted: {classes[pred.item()]}"
    )



original_dataset = torchvision.datasets.FashionMNIST(
    root='./data',
    train=False,
    download=True,
    transform=None
)

image, label = original_dataset[0]

plt.imshow(
    image,
    cmap='gray'
)

plt.title(
    original_dataset.classes[label]
)

plt.axis('off')

plt.show()
```

### OUTPUT

## Training Loss per Epoch



<img width="389" height="155" alt="image" src="https://github.com/user-attachments/assets/fc4de977-6f77-484d-a66f-78492d6c30cc" />



## Confusion Matrix

<img width="756" height="646" alt="image" src="https://github.com/user-attachments/assets/a4558bf8-4c1e-4602-b019-e03155e1296c" />



## Classification Report

<img width="801" height="438" alt="image" src="https://github.com/user-attachments/assets/0f70d09c-055a-451b-8513-e78068c93d76" />



### New Sample Data Prediction


<img width="663" height="688" alt="image" src="https://github.com/user-attachments/assets/043c3f7a-00ec-4c40-bcb2-9f3690c975f1" />

<img width="590" height="554" alt="image" src="https://github.com/user-attachments/assets/8d71209f-ee62-412f-812d-10a2f6aaef4c" />



## RESULT
    Thus, To develop a convolutional deep neural network (CNN) for image classification and to verify the response for new images is executed and verified successfully.


