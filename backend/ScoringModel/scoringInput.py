#!/usr/bin/env python
"""
Module for using the trained deer parts detection model to score input images
"""
import os
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import matplotlib.pyplot as plt

def load_model(model_path=None, device=None):
    if device is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # If model_path is not specified, find the latest model
    if model_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(script_dir, "models")
        
        if not os.path.exists(models_dir) or not os.listdir(models_dir):
            raise FileNotFoundError(f"No models found in {models_dir}")
        
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
        if not model_files:
            raise FileNotFoundError(f"No model files found in {models_dir}")
        
        # If there are checkpoint files, prefer the final model if it exists
        final_models = [f for f in model_files if not f.startswith("deer_parts_detector_epoch_10")]
        if final_models:
            model_path = os.path.join(models_dir, max(final_models, key=lambda x: os.path.getmtime(os.path.join(models_dir, x))))
        else:
            # Otherwise use the latest checkpoint
            model_path = os.path.join(models_dir, max(model_files, key=lambda x: os.path.getmtime(os.path.join(models_dir, x))))
    
    print(f"Loading model from: {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get classes
    if 'classes' in checkpoint:
        classes = checkpoint['classes']
        print(f"Found {len(classes)} classes in model: {classes}")
    else:
        # Default classes if not saved with model
        classes = ['antler', 'ear', 'eye', 'nose', 'forehead']
        print(f"No classes found in model, using defaults: {classes}")
    
    # Build model with correct number of classes
    num_classes = len(classes) + 1  # Add background
    model = fasterrcnn_resnet50_fpn(pretrained=False)
    
    # Replace classifier with the right number of classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()  # Set model to evaluation mode
    
    return model, ['background'] + classes

def detect_deer_parts(model, image, class_names, device=None, score_threshold=0.3):
    """
    Detect deer parts in an image
    
    Args:
        model: The trained model
        image: PIL Image or path to an image file
        class_names: List of class names (with background as first class)
        device: Device to run inference on (default: same as model)
        score_threshold: Confidence score threshold for detections
        
    Returns:
        Tuple of (annotated image, detections list)
    """
    if device is None:
        device = next(model.parameters()).device
    
    print(f"Running detection on device: {device}")
    
    # Handle input which could be either a path or a PIL Image
    if isinstance(image, str):
        # It's a path to an image
        if not os.path.exists(image):
            raise FileNotFoundError(f"Image not found: {image}")
        image = Image.open(image).convert("RGB")
        print(f"Loaded image with size: {image.size}")
    elif not isinstance(image, Image.Image):
        raise TypeError("Image must be a PIL Image object or a path to an image file")
    
    # Convert to tensor and move to device
    transform = torchvision.transforms.ToTensor()
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    # Run inference
    with torch.no_grad():
        try:
            predictions = model(img_tensor)
            print(f"Raw predictions: {len(predictions)} images processed")
            
            # Print all raw predictions
            print(f"Boxes count: {len(predictions[0]['boxes'])}")
            print(f"Scores: {predictions[0]['scores']}")
            print(f"Labels: {predictions[0]['labels']}")
        except Exception as e:
            print(f"Error during inference: {e}")
            import traceback
            traceback.print_exc()
            return image, []  # Return original image and empty detections
    
    # Extract detection results
    boxes = predictions[0]['boxes'].cpu().numpy()
    scores = predictions[0]['scores'].cpu().numpy()
    labels = predictions[0]['labels'].cpu().numpy()
    
    print(f"Detection results: {len(boxes)} boxes")
    print(f"Detected objects above threshold {score_threshold}:")
    for i, (box, score, label_idx) in enumerate(zip(boxes, scores, labels)):
        if score >= score_threshold:
            class_name = class_names[label_idx]
            print(f"  {i}: {class_name} (score: {score:.4f}) at {box}")
    
    # Draw detections on the image
    result_img = image.copy()
    draw = ImageDraw.Draw(result_img)
    
    # Try to get a font
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    # Create a list to store detection results
    detections = []
    
    # Use different colors for different classes
    colors = [
        "red", "blue", "green", "orange", "purple", 
        "cyan", "magenta", "yellow", "brown", "pink"
    ]
    
    # Draw bounding boxes and labels for detections above threshold
    for i, (box, score, label_idx) in enumerate(zip(boxes, scores, labels)):
        if score >= score_threshold:
            # Convert box coordinates to integers
            box = box.astype(np.int32)
            x1, y1, x2, y2 = box
            
            # Get class name
            class_name = class_names[label_idx]
            
            # Choose color based on class
            color_idx = label_idx % len(colors)
            color = colors[color_idx]
            
            # Draw bounding box
            draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=3)
            
            # Draw label with score
            text = f"{class_name}: {score:.2f}"
            draw.rectangle([(x1, y1-20), (x1 + 100, y1)], fill=color)
            draw.text((x1, y1-20), text, fill="white", font=font)
            
            # Add to detections list
            detections.append({
                'class': class_name,
                'score': float(score),
                'box': box.tolist()
            })
    
    print(f"Returning {len(detections)} detections above threshold {score_threshold}")
    return result_img, detections

def visualize_detections(image, detections, output_path=None):
    """Create a matplotlib visualization of detections"""
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    plt.axis('off')
    
    # Count detections by class
    class_counts = {}
    for det in detections:
        cls = det['class']
        class_counts[cls] = class_counts.get(cls, 0) + 1
    
    # Create title
    title = "Detected deer parts:\n"
    for cls, count in class_counts.items():
        title += f"{cls}: {count}, "
    title = title.rstrip(", ")
    
    plt.title(title)
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Saved visualization to {output_path}")
    else:
        plt.tight_layout()
        plt.show()

# When this script is run directly, use the test image
if __name__ == "__main__":
    # Hard-coded test image
    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_image_path = os.path.join(script_dir, "test2score.jpg")
    
    try:
        # Check if the test image exists
        if not os.path.exists(test_image_path):
            print(f"Test image not found: {test_image_path}")
            print("Please place 'testScore.webp' in the same directory as this script.")
            exit(1)
            
        # Load model - explicitly printing which model is loaded
        print("Loading model...")
        model, class_names = load_model()
        
        # Process image with a lower threshold to see more detections
        print(f"Processing test image: {test_image_path}")
        result_img, detections = detect_deer_parts(model, test_image_path, class_names, score_threshold=0.3)
        
        # Save result
        output_path = os.path.join(script_dir, "testScore_detected.jpg")
        result_img.save(output_path)
        print(f"Result saved to: {output_path}")
        
        # Also save a matplotlib visualization
        vis_path = os.path.join(script_dir, "testScore_visualization.jpg")
        visualize_detections(result_img, detections, vis_path)
        
        # Print detections
        print("\nDetections Summary:")
        for i, d in enumerate(detections):
            print(f"  {i+1}. {d['class']}: {d['score']:.2f} at {d['box']}")
        
        if not detections:
            print("\nWARNING: No detections found. Possible issues:")
            print("1. The model needs more training")
            print("2. The confidence threshold might be too high")
            print("3. The image might not contain recognizable deer parts")
            print("4. The model might not be compatible with this image")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()