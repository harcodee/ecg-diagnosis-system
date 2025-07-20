# ecg-diagnosis-system

A deep learning-based ECG image classification system that predicts cardiovascular conditions from ECG scans using MobileNetV2. This project supports batch evaluation, single image prediction, and generates PDF diagnostic reports.

##  Features

-  Trains a MobileNetV2 model on ECG datasets.
-  Achieves high accuracy on test data.
-  Supports prediction on individual ECG images.
-  Generates professional PDF diagnosis reports.
-  Ready for deployment and further extension.

##  Project Structure

ecg-diagnosis-system/
│
├── ecg_model.pth # Pre-trained model weights
├── import_torch.py # Main Python script
├── requirements.txt # Project dependencies
├── README.md # Project overview
├── .gitignore # Files to ignore in git
└── ECG_Diagnosis_Report.pdf # Sample generated report

##  Requirements

Install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt

Key libraries:
torch
torchvision
pillow
reportlab
matplotlib
numpy

 Usage
🔹 1. Evaluate Model
Run the script to load the pre-trained model and evaluate test accuracy:

python import_torch.py
🔹 2. Predict on a Single ECG Image
Set the test_image_path in import_torch.py to your ECG image file:
test_image_path = r"C:\path\to\your\ecg_image.jpg"
Run the script. The model will:
Print the diagnosis and confidence.
Generate a PDF report (ECG_Diagnosis_Report.pdf).

 Performance
Test Accuracy: 99.89%

Confidence Score: Reported per image prediction.
