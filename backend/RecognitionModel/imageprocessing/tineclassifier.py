import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet50_Weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Rebuild model structure and load weights
model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("tinedetector.pth", map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def predict_tine(image_path):
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img)
        probs = F.softmax(output, dim=1)
        pred = torch.argmax(probs, dim=1).item()

    label = "Tine" if pred == 1 else "Not Tine"
    confidence = probs[0, pred].item()
    return f"{label} (Confidence: {confidence:.2%})"

dataset_folder = "tine_dataset/train"

# Loop through both 'positive' and 'negative' folders
for label_folder in ['positive', 'negative']:
    folder_path = os.path.join(dataset_folder, label_folder)

    # Loop over each file in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".png"):  # Check for PNG files
            image_path = os.path.join(folder_path, filename)
            
            # Classify the image using the tine_classifier (predict_tine)
            result = predict_tine(image_path)

            # Determine if the image was correctly classified
            expected_label = "Tine" if label_folder == "positive" else "Not Tine"
            correct = "Correct" if expected_label in result else "Incorrect"

            # Handle the result (print, save, log, etc.)
            print(f"Result for {filename}: {result} - {correct}")
            
            # Optionally, save the result to a text file
            with open("classification_results.txt", "a") as result_file:
                result_file.write(f"{filename}: {result} - {correct}\n")

print("Classification process complete.")
