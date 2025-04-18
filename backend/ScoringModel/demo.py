#!/usr/bin/env python
"""
Demo script for deer parts detection with distance measurements between detected parts
"""
import os
import sys
import argparse
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import math
from ScoringModel.scoringInput import load_model, detect_deer_parts

def calculate_distance(point1, point2, scale=1.0):
    """
    Calculate the Euclidean distance between two points
    
    Args:
        point1: (x, y) coordinates of first point
        point2: (x, y) coordinates of second point
        scale: Optional scale factor for the distance (default: 1.0)
        
    Returns:
        Distance in pixels (scaled if scale != 1.0)
    """
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2) * scale

def get_box_center(box):
    """
    Get the center point of a bounding box
    
    Args:
        box: List of [x1, y1, x2, y2] coordinates
        
    Returns:
        (center_x, center_y) tuple
    """
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, (y1 + y2) // 2)

def draw_distances(image, detections, output_path=None, scale_factor=1.0, min_score=0.5):
    """
    Draw lines between detected deer parts with distance measurements
    
    Args:
        image: PIL Image object
        detections: List of detection dictionaries
        output_path: Path to save the visualization (default: None, will display instead)
        scale_factor: Optional scaling for distances (default: 1.0)
        min_score: Minimum confidence score to include in distance calculations
        
    Returns:
        PIL Image with visualization
    """
    # Create a copy of the image to draw on
    result_img = image.copy()
    draw = ImageDraw.Draw(result_img)
    
    # Try to load a font
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        except:
            font = ImageFont.load_default()
    
    # Filter detections by score
    valid_detections = [d for d in detections if d['score'] >= min_score]
    print(f"Using {len(valid_detections)} detections with scores >= {min_score}")
    
    # Get centers of each detection
    detection_centers = []
    for det in valid_detections:
        center = get_box_center(det['box'])
        detection_centers.append({
            'class': det['class'],
            'center': center,
            'score': det['score'],
            'box': det['box']
        })
    
    # Define colors for different pairs
    colors = {
        ('antler', 'antler'): "red",
        ('eye', 'eye'): "blue",
        ('ear', 'ear'): "green",
        ('antler', 'eye'): "purple",
        ('antler', 'ear'): "orange",
        ('antler', 'nose'): "cyan",
        ('eye', 'ear'): "yellow",
        ('eye', 'nose'): "magenta",
        ('ear', 'nose'): "brown",
        ('forehead', 'antler'): "pink",
        ('forehead', 'eye'): "teal",
        ('forehead', 'ear'): "lime",
        ('forehead', 'nose'): "violet"
    }
    
    # Default color for pairs not in the dictionary
    default_color = "gray"
    
    # Dictionary to store all measurements
    measurements = []
    
    # Draw lines between every pair of detections
    for i in range(len(detection_centers)):
        for j in range(i+1, len(detection_centers)):
            center1 = detection_centers[i]['center']
            class1 = detection_centers[i]['class']
            center2 = detection_centers[j]['center']
            class2 = detection_centers[j]['class']
            
            # Get color for this pair of classes
            pair = tuple(sorted([class1, class2]))
            color = colors.get(pair, default_color)
            
            # Calculate distance
            distance = calculate_distance(center1, center2, scale_factor)
            
            # Add to measurements
            measurements.append({
                'from': class1,
                'to': class2,
                'distance': distance,
                'from_center': center1,
                'to_center': center2
            })
            
            # Draw line between centers
            draw.line([center1, center2], fill=color, width=2)
            
            # Calculate midpoint for text
            mid_x = (center1[0] + center2[0]) // 2
            mid_y = (center1[1] + center2[1]) // 2
            
            # Draw distance text
            text = f"{distance:.1f}"
            text_bbox = draw.textbbox((mid_x, mid_y), text, font=font)
            
            # Add a background for the text for better visibility
            padding = 2
            draw.rectangle(
                [
                    text_bbox[0] - padding,
                    text_bbox[1] - padding,
                    text_bbox[2] + padding,
                    text_bbox[3] + padding
                ],
                fill="white"
            )
            draw.text((mid_x, mid_y), text, fill=color, font=font)
    
    # Print the measurements
    print("\nDistance Measurements:")
    for m in measurements:
        print(f"{m['from']} to {m['to']}: {m['distance']:.1f} pixels")
    
    # Save or display the result
    if output_path:
        result_img.save(output_path)
        print(f"Result saved to: {output_path}")
    
    return result_img, measurements

def create_visualization(image, detections, measurements, output_path=None):
    """Create a matplotlib visualization of detections with measurements"""
    plt.figure(figsize=(14, 10))
    plt.imshow(image)
    plt.axis('off')
    
    # Count detections by class
    class_counts = {}
    for det in detections:
        cls = det['class']
        class_counts[cls] = class_counts.get(cls, 0) + 1
    
    # Create title
    title = "Deer Parts Analysis\n"
    for cls, count in class_counts.items():
        title += f"{cls}: {count}, "
    title = title.rstrip(", ")
    
    plt.title(title)
    
    # Add a table of measurements
    if measurements:
        measurement_text = "Distance Measurements:\n"
        # Sort measurements by distance
        sorted_measurements = sorted(measurements, key=lambda x: x['distance'])
        for m in sorted_measurements:
            measurement_text += f"{m['from']} to {m['to']}: {m['distance']:.1f} pixels\n"
        
        plt.figtext(0.02, 0.02, measurement_text, fontsize=10, 
                   bbox={"facecolor":"white", "alpha":0.8, "pad":5})
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Saved visualization to {output_path}")
    else:
        plt.tight_layout()
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Deer Parts Detection with Distance Measurements')
    parser.add_argument('image_path', help='Path to the input image')
    parser.add_argument('--output', '-o', help='Path to save the output image', default=None)
    parser.add_argument('--threshold', '-t', type=float, default=0.3, 
                        help='Detection confidence threshold (default: 0.3)')
    parser.add_argument('--scale', '-s', type=float, default=1.0,
                        help='Scale factor for distance measurements (default: 1.0)')
    parser.add_argument('--model', '-m', help='Path to the model file (optional)')
    
    args = parser.parse_args()
    
    # Check if the input image exists
    if not os.path.exists(args.image_path):
        print(f"Error: Image not found: {args.image_path}")
        return 1
    
    # Load the model
    try:
        model, class_names = load_model(args.model)
    except Exception as e:
        print(f"Error loading model: {e}")
        return 1
    
    # Process the image
    try:
        image = Image.open(args.image_path).convert("RGB")
        result_img, detections = detect_deer_parts(
            model, image, class_names, score_threshold=args.threshold
        )
        
        # Draw distances
        result_img, measurements = draw_distances(
            result_img, detections, output_path=args.output,
            scale_factor=args.scale, min_score=args.threshold
        )
        
        # Create a visualization with matplotlib
        if args.output:
            vis_path = os.path.splitext(args.output)[0] + "_analysis.jpg"
            create_visualization(result_img, detections, measurements, vis_path)
        
        return 0
        
    except Exception as e:
        print(f"Error processing image: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
