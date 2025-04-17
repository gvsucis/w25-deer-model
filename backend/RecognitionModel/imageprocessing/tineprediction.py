import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "tinedetector.pth")
image_path = os.path.join(BASE_DIR, "depth_maps", "Antler6F", "antler6F_angle0.png")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])


model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

img = Image.open(image_path).convert("RGB")
img_tensor = transform(img).unsqueeze(0).to(device)

with torch.no_grad():
    outputs = model(img_tensor)
    _, predicted = torch.max(outputs, 1)

class_names = ['no_tine', 'tine']
predicted_class = class_names[predicted.item()]
print(f"Predicted class: {predicted_class}")
