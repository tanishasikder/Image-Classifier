import kagglehub
import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import torchvision.models as models
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from torchvision.models import resnet18, ResNet18_Weights
import numpy as np

path = kagglehub.dataset_download("jessicali9530/caltech256")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# There are 256 object categories
num_classes = 256
num_epochs = 10
learning_rate = 0.001

# Transformations for the images
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(), 
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Loading the dataset
dataset = ImageFolder(root=path + "/256_ObjectCategories", transform=transform)

# Splitting into train and test sets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# CNN to extract image features
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Processes the image. Outputs feature vector
        resnet = models.resnet18(pretrained=True)
        # Removes last classification layer
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
    
    def forward(self, x):
        with torch.no_grad():
            features = self.resnet(x)
            # Flatten then return
            return features.view(features.size(0), -1)
    
model = CNN()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
n_total_steps = len(train_loader)

# sklearn models
random_forest = RandomForestClassifier()
decision_tree = DecisionTreeClassifier()
svc = SVC()

train_outputs = []
train_labels = []

for images, labels in train_loader:
    output = model(images)
    train_outputs.append(output.detach().numpy())
    train_labels.append(labels.detach().numpy())

X_train = np.vstack(train_outputs)
y_train = np.hstack(train_labels)

random_forest.fit(X_train, y_train)
decision_tree.fit(X_train, y_train)
svc.fit(X_train, y_train)

test_outputs = []
test_labels = []

for images, labels in test_loader:
    output = model(images)
    test_outputs.append(output.detach().numpy())
    test_labels.append(labels.detach().numpy())

X_test = np.vstack(test_outputs)
y_test = np.hstack(test_labels)

r_pred = random_forest.predict(X_test)
d_pred = decision_tree.predict(X_test)
s_pred = svc.predict(X_test)

print(accuracy_score(y_test, r_pred))
print(accuracy_score(y_test, d_pred))
print(accuracy_score(y_test, s_pred))

