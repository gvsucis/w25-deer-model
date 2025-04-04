#!/usr/bin/env python
"""
Deer parts detection model for scoring deer attributes
"""
import os
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader, Dataset, random_split, Subset
import numpy as np
from PIL import Image, ImageDraw
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

# Get the dataset path relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(SCRIPT_DIR, "antlerScore.v3i.yolov8")

class DeerPartsDataset(Dataset):
    """Dataset for deer parts detection using YOLO format"""
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        
        # Load dataset paths
        self.imgs_path = os.path.join(root, "train", "images")
        self.labels_path = os.path.join(root, "train", "labels")
        
        if not os.path.exists(self.imgs_path) or not os.path.exists(self.labels_path):
            raise ValueError(f"Dataset directories not found: {self.imgs_path} or {self.labels_path}")
        
        self.imgs = list(sorted(os.listdir(self.imgs_path)))
        
        # Get class names from data.yaml
        yaml_path = os.path.join(root, "data.yaml")
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
            self.classes = data['names']
        
        self.num_classes = len(self.classes)
        print(f"Loaded dataset with {len(self.imgs)} images and {self.num_classes} classes: {self.classes}")

    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.imgs_path, self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        
        # Get corresponding label file
        img_id = os.path.splitext(self.imgs[idx])[0]
        label_path = os.path.join(self.labels_path, f"{img_id}.txt")
        
        boxes = []
        labels = []
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    data = line.strip().split(' ')
                    class_id = int(data[0])
                    # YOLO format: (class_id, x_center, y_center, width, height) normalized
                    x_center, y_center, width, height = map(float, data[1:5])
                    
                    # Convert to (x1, y1, x2, y2) format for PyTorch
                    img_width, img_height = img.size
                    x1 = (x_center - width/2) * img_width
                    y1 = (y_center - height/2) * img_height
                    x2 = (x_center + width/2) * img_width
                    y2 = (y_center + height/2) * img_height
                    
                    boxes.append([x1, y1, x2, y2])
                    labels.append(class_id)
        
        # Convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if len(boxes) > 0 else torch.zeros((0,))
        
        # Create target dictionary
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = torch.zeros((len(boxes),), dtype=torch.int64)
        
        if self.transforms:
            img, target = self.transforms(img, target)
        
        return img, target

    def __len__(self):
        return len(self.imgs)

def get_transform(train):
    """Get transforms for training and testing"""
    class Compose:
        def __init__(self, transforms):
            self.transforms = transforms
            
        def __call__(self, image, target):
            for t in self.transforms:
                image, target = t(image, target)
            return image, target
    
    class ToTensor:
        def __call__(self, image, target):
            return torchvision.transforms.functional.to_tensor(image), target
    
    class RandomHorizontalFlip:
        def __init__(self, prob):
            self.prob = prob
            
        def __call__(self, image, target):
            if torch.rand(1) < self.prob:
                height, width = image.shape[-2:]
                image = torchvision.transforms.functional.hflip(image)
                bbox = target["boxes"]
                bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
                target["boxes"] = bbox
            return image, target
    
    transforms = [ToTensor()]
    if train:
        transforms.append(RandomHorizontalFlip(0.5))
    
    return Compose(transforms)

def build_model(num_classes, pretrained=True):
    """Build a Faster R-CNN model"""
    # For quick testing, we can use a smaller backbone
    model = fasterrcnn_resnet50_fpn(pretrained=pretrained, 
                                   trainable_backbone_layers=3)  # Freeze early layers to train faster
    
    # Replace the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model

def train_one_epoch(model, optimizer, data_loader, device):
    """Train the model for one epoch"""
    model.train()
    
    epoch_loss = 0
    num_batches = 0
    
    for images, targets in tqdm(data_loader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # In training mode, model returns a dict of losses
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        epoch_loss += losses.item()
        num_batches += 1
    
    return epoch_loss / num_batches if num_batches > 0 else 0

def evaluate(model, data_loader, device):
    """Evaluate the model"""
    model.eval()
    
    # For evaluation, we'll use our own loss calculation
    from torch.nn.functional import cross_entropy
    
    val_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for images, targets in tqdm(data_loader):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # In eval mode, model returns detection results
            # So we need to compute the loss manually
            outputs = model(images)
            
            # Calculate a simple loss (just for monitoring)
            batch_loss = 0
            for i, target in enumerate(targets):
                if len(target['boxes']) > 0 and len(outputs[i]['boxes']) > 0:
                    # Add a simple loss based on the detection scores
                    batch_loss += (1.0 - outputs[i]['scores'].mean()).item()
                else:
                    # Add a penalty if no detections
                    batch_loss += 1.0
            
            val_loss += batch_loss
            num_batches += 1
    
    return val_loss / num_batches if num_batches > 0 else 0

def predict_image(model, image_path, class_names, device, score_threshold=0.5):
    """
    Predict deer parts in an image
    
    Args:
        model: Trained model
        image_path: Path to the image file
        class_names: List of class names (with 'background' as first)
        device: Device to run inference on
        score_threshold: Minimum confidence score for detections
        
    Returns:
        Tuple of (annotated image, detection results)
    """
    image = Image.open(image_path).convert("RGB")
    transform = torchvision.transforms.ToTensor()
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        prediction = model(img_tensor)
    
    # Process prediction
    boxes = prediction[0]['boxes'].cpu().numpy()
    scores = prediction[0]['scores'].cpu().numpy()
    labels = prediction[0]['labels'].cpu().numpy()
    
    # Draw bounding boxes
    result_img = image.copy()
    draw = ImageDraw.Draw(result_img)
    
    detected_parts = []
    for box, score, label in zip(boxes, scores, labels):
        if score > score_threshold:
            box = box.astype(np.int32)
            draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline="red", width=3)
            
            class_name = class_names[label]
            draw.text((box[0], box[1]), f"{class_name}: {score:.2f}", fill="red")
            
            detected_parts.append({
                'class': class_name,
                'score': float(score),
                'box': box.tolist()
            })
    
    return result_img, detected_parts

def train_model(epochs=10, batch_size=2, learning_rate=0.005):
    """
    Train a deer parts detection model
    
    Args:
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Initial learning rate
        
    Returns:
        Path to the saved model
    """
    # Create output directory
    output_dir = os.path.join(SCRIPT_DIR, "models")
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")
    
    # Create dataset
    dataset = DeerPartsDataset(DATASET_PATH, transforms=get_transform(train=True))
    
    # Split dataset into train/val sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        collate_fn=lambda x: tuple(zip(*x))
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        collate_fn=lambda x: tuple(zip(*x))
    )
    
    # Build model
    num_classes = len(dataset.classes) + 1  # Add background class
    model = build_model(num_classes)
    model.to(device)
    
    # Define optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=0.0005)
    
    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    
    # Train the model
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # Train for one epoch
        train_loss = train_one_epoch(model, optimizer, train_loader, device)
        train_losses.append(train_loss)
        print(f"Train Loss: {train_loss:.4f}")
        
        # Update learning rate
        lr_scheduler.step()
        
        # Evaluate on validation set
        val_loss = evaluate(model, val_loader, device)
        val_losses.append(val_loss)
        print(f"Validation Loss: {val_loss:.4f}")
    
    # Save the final model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(output_dir, f"deer_parts_detector_{timestamp}.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'classes': dataset.classes
    }, model_path)
    print(f"Model saved to {model_path}")
    
    # Plot training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    loss_plot_path = os.path.join(output_dir, f"loss_plot_{timestamp}.png")
    plt.savefig(loss_plot_path)
    print(f"Loss plot saved to {loss_plot_path}")
    
    return model_path

def train_model_quick_test(sample_fraction=0.1, epochs=2, batch_size=4, learning_rate=0.005):
    """
    Train a deer parts detection model quickly for testing purposes
    
    Args:
        sample_fraction: Fraction of the dataset to use (0.1 = 10%)
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Initial learning rate
        
    Returns:
        Path to the saved model
    """
    # Create output directory
    output_dir = os.path.join(SCRIPT_DIR, "models")
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")
    
    # Create dataset
    dataset = DeerPartsDataset(DATASET_PATH, transforms=get_transform(train=True))
    
    # Take only a subset of the data for quick training
    total_samples = len(dataset)
    num_samples = max(int(total_samples * sample_fraction), 10)  # At least 10 samples
    
    # Create subset indices
    indices = torch.randperm(total_samples)[:num_samples].tolist()
    subset_dataset = Subset(dataset, indices)
    
    print(f"Using {num_samples}/{total_samples} images for quick training")
    
    # Split subset into train/val
    train_size = int(0.8 * num_samples)
    val_size = num_samples - train_size
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size+val_size]
    
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    
    # Create data loaders with smaller batch size
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        collate_fn=lambda x: tuple(zip(*x)),
        num_workers=0  # Use single process to avoid overhead
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        collate_fn=lambda x: tuple(zip(*x)),
        num_workers=0  # Use single process to avoid overhead
    )
    
    # Build model with fewer trainable layers for speed
    num_classes = len(dataset.classes) + 1  # Add background class
    model = build_model(num_classes)
    model.to(device)
    
    # Define optimizer with higher learning rate for faster convergence
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=0.0005)
    
    # Skip the learning rate scheduler for quick testing
    
    # Train the model for fewer epochs
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # Train for one epoch
        train_loss = train_one_epoch(model, optimizer, train_loader, device)
        train_losses.append(train_loss)
        print(f"Train Loss: {train_loss:.4f}")
        
        # Evaluate on validation set
        val_loss = evaluate(model, val_loader, device)
        val_losses.append(val_loss)
        print(f"Validation Loss: {val_loss:.4f}")
    
    # Save the model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(output_dir, f"deer_parts_detector_quicktest_{timestamp}.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'classes': dataset.classes
    }, model_path)
    print(f"Model saved to {model_path}")
    
    # Plot training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    loss_plot_path = os.path.join(output_dir, f"loss_plot_quicktest_{timestamp}.png")
    plt.savefig(loss_plot_path)
    print(f"Loss plot saved to {loss_plot_path}")
    
    return model_path

def load_model(model_path=None, device=None):
    """
    Load a trained model
    
    Args:
        model_path: Path to the saved model (if None, uses the latest model in models dir)
        device: Device to load the model on (default: auto-detect)
        
    Returns:
        Tuple of (model, class_names)
    """
    if device is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # If no model path is provided, find the latest model
    if model_path is None:
        models_dir = os.path.join(SCRIPT_DIR, "models")
        if not os.path.exists(models_dir) or not os.listdir(models_dir):
            raise FileNotFoundError("No trained models found. Please train a model first.")
        
        # Get the most recent model file
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
        if not model_files:
            raise FileNotFoundError("No trained models found. Please train a model first.")
        
        latest_model = max(model_files, key=lambda x: os.path.getmtime(os.path.join(models_dir, x)))
        model_path = os.path.join(models_dir, latest_model)
        print(f"Using latest model: {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get classes
    if 'classes' in checkpoint:
        classes = checkpoint['classes']
    else:
        # Default classes if not saved with model
        classes = ['antler', 'ear', 'eye', 'nose', 'forehead']
    
    # Build model with correct number of classes
    num_classes = len(classes) + 1  # Add background
    model = build_model(num_classes, pretrained=False)  # No need for pretrained weights when loading
    
    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    # Return model and class names (with background as first class)
    return model, ['background'] + classes

# Main entry point
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "train":
            print("Training a full deer parts detection model...")
            train_model(epochs=10)
        elif sys.argv[1] == "quick-test":
            print("Running a quick test training...")
            train_model_quick_test(sample_fraction=0.1, epochs=2)
        else:
            print(f"Unknown command: {sys.argv[1]}")
            print("Available commands: train, quick-test")
    else:
        # Example usage for inference
        print("Loading model for inference.")
        print("Use 'python deer_parts_detector.py train' for full training")
        print("Use 'python deer_parts_detector.py quick-test' for a quick test training")
        
        try:
            model, class_names = load_model()
            print(f"Model loaded successfully. Available classes: {class_names[1:]}")
            print("You can now use this model in your application.")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print("Please train a model first using 'python deer_parts_detector.py train'")
            print("or run a quick test with 'python deer_parts_detector.py quick-test'")