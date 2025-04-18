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
    
    if model_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(script_dir, "models")
        
        if not os.path.exists(models_dir) or not os.listdir(models_dir):
            raise FileNotFoundError(f"No models found in {models_dir}")
        
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
        if not model_files:
            raise FileNotFoundError(f"No model files found in {models_dir}")
        
        # current trained model
        final_models = [f for f in model_files if not f.startswith("deer_parts_detector_epoch_10")]
        if final_models:
            model_path = os.path.join(models_dir, max(final_models, key=lambda x: os.path.getmtime(os.path.join(models_dir, x))))
        else:
            # Otherwise use the latest checkpoint
            model_path = os.path.join(models_dir, max(model_files, key=lambda x: os.path.getmtime(os.path.join(models_dir, x))))
    
    print(f"Loading model from: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    if 'classes' in checkpoint:
        classes = checkpoint['classes']
        print(f"Found {len(classes)} classes in model: {classes}")
    else:
        classes = ['']
        print(f"No classes found in model, using defaults: {classes}")
    
    num_classes = len(classes) + 1  
    model = fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()  
    
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
    
    if isinstance(image, str):
        if not os.path.exists(image):
            raise FileNotFoundError(f"Image not found: {image}")
        image = Image.open(image).convert("RGB")
        print(f"Loaded image with size: {image.size}")
    elif not isinstance(image, Image.Image):
        raise TypeError("Image must be a PIL Image object or a path to an image file")
    transform = torchvision.transforms.ToTensor()
    img_tensor = transform(image).unsqueeze(0).to(device)
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
    
    result_img = image.copy()
    draw = ImageDraw.Draw(result_img)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
    detections = []
    colors = [
        "red", "blue", "green", "orange", "purple", 
        "cyan", "magenta", "yellow", "brown", "pink"
    ]
    for i, (box, score, label_idx) in enumerate(zip(boxes, scores, labels)):
        if score >= score_threshold:
            # Convert box coordinates to integers
            box = box.astype(np.int32)
            x1, y1, x2, y2 = box
            class_name = class_names[label_idx]
            color_idx = label_idx % len(colors)
            color = colors[color_idx]
            draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=1)
            
            # Draw label with score
            text = f"{class_name}: {score:.1f}"
            draw.rectangle([(x1, y1-15), (x1 + 80, y1)], fill=color)
            draw.text((x1, y1-15), text, fill="white", font=font)
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
    class_counts = {}
    for det in detections:
        cls = det['class']
        class_counts[cls] = class_counts.get(cls, 0) + 1
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
    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_image_path = os.path.join(script_dir, "test2score.jpg")
    
    try:
        if not os.path.exists(test_image_path):
            print(f"Test image not found: {test_image_path}")
            print("Please place 'testScore.webp' in the same directory as this script.")
            exit(1)
        print("Loading model...")
        model, class_names = load_model()
        print(f"Processing test image: {test_image_path}")
        result_img, detections = detect_deer_parts(model, test_image_path, class_names, score_threshold=0.3)
        output_path = os.path.join(script_dir, "testScore_detected.jpg")
        result_img.save(output_path)
        print(f"Result saved to: {output_path}")
        vis_path = os.path.join(script_dir, "testScore_visualization.jpg")
        visualize_detections(result_img, detections, vis_path)
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