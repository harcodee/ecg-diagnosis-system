import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
from reportlab.pdfgen import canvas
from datetime import datetime
from PIL import Image

# === Paths ===
train_dir = r"C:\ecg_dataset\ECG_DATA\train"
test_dir = r"C:\ecg_dataset\ECG_DATA\test"
model_save_path = "ecg_model.pth"

# === Transformations ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 3 channels RGB
])

# === Datasets and Loaders ===
train_dataset = datasets.ImageFolder(train_dir, transform=transform)
test_dataset = datasets.ImageFolder(test_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

# === Model (MobileNetV2) ===
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)  # Updated for deprecation
model.classifier[1] = nn.Linear(model.last_channel, len(train_dataset.classes))

device = torch.device("cpu")
model = model.to(device)
model.load_state_dict(torch.load(model_save_path, map_location=device))
print("Loaded trained model from ecg_model.pth")


# === Loss and Optimizer ===
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

""" === Training ===
epochs = 5
for epoch in range(epochs):
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
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")

print(" Training completed.")
torch.save(model.state_dict(), model_save_path)
print(f" Model saved as {model_save_path}")"""

# === Evaluation on Test Set ===
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f" Test Accuracy: {accuracy:.2f}%")

# === PDF Report Generator ===
def generate_pdf_report(image_path, predicted_class, confidence_score):
    pdf_file = "ECG_Diagnosis_Report.pdf"
    c = canvas.Canvas(pdf_file)
    c.setFont("Helvetica-Bold", 20)
    c.drawString(50, 800, "ECG Diagnosis Report")
    c.setFont("Helvetica", 12)
    c.drawString(50, 770, f"Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    c.drawString(50, 740, f"Input Image: {os.path.basename(image_path)}")
    c.drawString(50, 710, f"Predicted Condition: {predicted_class}")
    c.drawString(50, 680, f"Confidence: {confidence_score:.2f}%")
    c.save()
    print(f" PDF report generated: {pdf_file}")

# === Predict Function ===
def predict_ecg_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
        predicted_class = train_dataset.classes[predicted_idx]
        confidence_score = confidence.item() * 100

    print(f" Diagnosis: {predicted_class} ({confidence_score:.2f}%)")
    generate_pdf_report(image_path, predicted_class, confidence_score)

# === Test Prediction Example ===
# Replace this path with your own ECG image path to test
test_image_path = r"C:\ecg_dataset\ECG_DATA\test\ECG Images of Myocardial Infarction Patients (240x12=2880)\MI(60).jpg"
predict_ecg_image(test_image_path)
