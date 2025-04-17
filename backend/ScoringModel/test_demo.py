#!/usr/bin/env python
import os
import torch
from PIL import Image
from ScoringModel.scoringInput import load_model, detect_deer_parts, visualize_detections

def run_test(image_path, output_dir=None, score_threshold=0.3):
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(output_dir, exist_ok=True)
    model, class_names = load_model()
    result_img, detections = detect_deer_parts(
        model,
        image_path,
        class_names,
        score_threshold=score_threshold
        )
        
    base_name = os.path.basename(image_path).split('.')[0]
    output_path = os.path.join(output_dir, f"{base_name}_result.jpg")
    result_img.save(output_path)

    vis_path = os.path.join(output_dir, f"{base_name}_visualization.jpg")
    visualize_detections(result_img, detections, vis_path)

    return detections
if __name__ == "__main__":
    image_path = "dt3.jpg"

    detections = run_test(image_path)

    print(f"Found {len(detections)} deer parts:")
    for i, det in enumerate(detections):
        print(f"  {i+1}. {det['class']}: {det['score']:.2f}")
