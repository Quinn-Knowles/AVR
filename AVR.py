import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report

class ShipDataset(Dataset):
    def __init__(self, image_dir, label_file, class_names, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.class_names = class_names
        with open(label_file, "r") as f:
            self.labels = json.load(f)
        self.image_names = list(self.labels.keys())

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = torch.zeros(len(self.class_names))
        for name in self.labels[img_name]:
            label[self.class_names.index(name)] = 1.0

        return image, label

# === Config ===
class_names = ["prince_of_wales", "dreadnought", "victory", "queen_elizabeth", "endurance"]
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_dataset = ShipDataset(
    image_dir="training_library",
    label_file="training_library/labels.json",
    class_names=class_names,
    transform=transform
)

test_dataset = ShipDataset(
    image_dir="testing_library",
    label_file="testing_library/labels.json",
    class_names=class_names,
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# === Model ===
model = models.resnet18(pretrained=True)
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, len(class_names)),
    nn.Sigmoid() 
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# === Loss & Optimizer ===
criterion = nn.BCELoss()  # For multi-label classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

# === Training Loop ===
for epoch in range(5):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")


torch.save(model.state_dict(), "ship_classifier.pth") #save model

# === Evaluation ===
model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs).cpu()
        preds = (outputs > 0.5).float()

        y_true.extend(labels.numpy())
        y_pred.extend(preds.numpy())

# === Report ===
print("\nEvaluation: ")
print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
