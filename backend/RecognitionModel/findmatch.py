import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchvision import models
from PIL import Image
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
DATA_DIR = os.path.join(BASE_DIR, "imageprocessing", "depth_maps")  
MODEL_PATH = os.path.join(BASE_DIR, "antler_cnn.pth") 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 47)  
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def predict_antler(image_path):
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img)
        probabilities = F.softmax(output, dim=1)
        pred_class = torch.argmax(probabilities, dim=1).item()

    antler_name = f"Antler{pred_class + 1}F"
    confidence = probabilities[0, pred_class].item()
    
    return f"Closest match: {antler_name} (Confidence: {confidence:.2%})"

  #this will be input from user need to fetch from frontend aka a parameter in the post request
# result = predict_antler(image_path)
# print(result)
