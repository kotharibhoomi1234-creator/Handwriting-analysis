"""
AWS Lambda / SageMaker Endpoint Handler for Graphology Analysis
This module handles computer vision feature extraction and graphology trait prediction
on AWS cloud instances (Lambda, ECS, or SageMaker).
"""
import json
import base64
import cv2
import numpy as np
from graphology_db import GRAPHOLOGY_DATABASE

def preprocess_and_crop(image_np):
    if len(image_np.shape) == 3:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_np
        
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 15, 8
    )
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None
        
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    
    pad = 6
    crop_thresh = thresh[max(0, y-pad):min(thresh.shape[0], y+h+pad), 
                         max(0, x-pad):min(thresh.shape[1], x+w+pad)]
    crop_gray = gray[max(0, y-pad):min(gray.shape[0], y+h+pad), 
                     max(0, x-pad):min(gray.shape[1], x+w+pad)]
                     
    return crop_thresh, crop_gray

def extract_detailed_features(crop_thresh, crop_gray):
    h, w = crop_thresh.shape
    aspect_ratio = h / float(w) if w > 0 else 1.0
    
    contours_tree, hierarchy = cv2.findContours(crop_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    internal_loops = 0
    if hierarchy is not None:
        internal_loops = np.sum(hierarchy[0][:, 3] != -1)
        
    upper_third = crop_thresh[0:int(h * 0.35), :]
    middle_third = crop_thresh[int(h * 0.35):int(h * 0.70), :]
    lower_third = crop_thresh[int(h * 0.70):, :]
    
    upper_density = np.sum(upper_third > 0) / (upper_third.size + 1e-5)
    middle_density = np.sum(middle_third > 0) / (middle_third.size + 1e-5)
    lower_density = np.sum(lower_third > 0) / (lower_third.size + 1e-5)
    
    left_half = crop_thresh[:, 0:int(w * 0.5)]
    right_half = crop_thresh[:, int(w * 0.5):]
    left_density = np.sum(left_half > 0) / (left_half.size + 1e-5)
    right_density = np.sum(right_half > 0) / (right_half.size + 1e-5)
    
    # Top and Bottom gap projections
    top_proj = np.sum(upper_third > 0, axis=0)
    top_mid = top_proj[int(w*0.25):int(w*0.75)] if w > 4 else top_proj
    is_top_open = np.any(top_mid == 0) and (np.sum(top_proj > 0) > 0)
    
    bottom_proj = np.sum(lower_third > 0, axis=0)
    bottom_mid = bottom_proj[int(w*0.25):int(w*0.75)] if w > 4 else bottom_proj
    is_bottom_open = np.any(bottom_mid == 0) and (np.sum(bottom_proj > 0) > 0)
    
    stroke_mask = crop_thresh > 0
    mean_intensity = np.mean(crop_gray[stroke_mask]) if np.sum(stroke_mask) > 0 else 255
    stroke_fullness = np.sum(stroke_mask) / float(h * w + 1e-5)
    is_dark_retraced = mean_intensity < 105 or stroke_fullness > 0.40
    
    perimeter = sum([cv2.arcLength(cnt, True) for cnt in contours_tree]) if contours_tree else 0
    complexity_index = perimeter / float(np.sum(stroke_mask) + 1e-5)
    
    return {
        "aspect_ratio": aspect_ratio,
        "internal_loops": internal_loops,
        "upper_density": upper_density,
        "middle_density": middle_density,
        "lower_density": lower_density,
        "left_density": left_density,
        "right_density": right_density,
        "is_top_open": is_top_open,
        "is_bottom_open": is_bottom_open,
        "mean_intensity": mean_intensity,
        "stroke_fullness": stroke_fullness,
        "is_dark_retraced": is_dark_retraced,
        "complexity_index": complexity_index
    }

def predict_trait(image_np, target_char):
    crop_thresh, crop_gray = preprocess_and_crop(image_np)
    char_key = target_char.lower()
    available_traits = GRAPHOLOGY_DATABASE.get(char_key, ["Standard handwriting formation."])
    num_traits = len(available_traits)
    
    if crop_thresh is None or crop_thresh.shape[0] < 5 or crop_thresh.shape[1] < 5:
        return available_traits[0], {}
        
    f = extract_detailed_features(crop_thresh, crop_gray)
    matched_idx = 0
    
    # Precise rules per letter shape
    if char_key == 'a':
        if f["complexity_index"] > 0.45 or f["internal_loops"] >= 2:
            matched_idx = 2  # Complex with curls, twists, and knots
        elif f["is_dark_retraced"]:
            matched_idx = 6  # Dark and straight line / retraced: Determination
        elif f["is_top_open"] and f["upper_density"] > 0.25:
            matched_idx = 0  # Open from up: Talkative
        elif f["is_bottom_open"] or f["lower_density"] > f["upper_density"] * 1.4:
            matched_idx = 1  # Open from down: Cheater
        elif f["upper_density"] > f["lower_density"] * 1.3:
            matched_idx = 7  # Irregular top stroke
        elif f["stroke_fullness"] > 0.45:
            matched_idx = 8  # Stiff structure
        elif f["stroke_fullness"] < 0.12:
            matched_idx = 9  # Sickly / broken line
        elif f["lower_density"] > f["middle_density"]:
            matched_idx = 3  # Cupping 'a'
        else:
            matched_idx = 4  # Double curve

    elif char_key == 'd':
        if f["upper_density"] > 0.25 and f["internal_loops"] >= 1:
            matched_idx = 4  # Tall loop 'd': Sensitive
        elif f["aspect_ratio"] > 2.2:
            matched_idx = 8  # High loop stroke
        elif f["aspect_ratio"] > 1.6:
            matched_idx = 1  # Stretched 'd'
        elif f["aspect_ratio"] < 1.05 and f["stroke_fullness"] > 0.30:
            matched_idx = 5  # Short & broad 'd'
        elif f["aspect_ratio"] < 1.1:
            matched_idx = 0  # Squeezed 'd'
        elif f["mean_intensity"] < 80 or f["stroke_fullness"] > 0.48:
            matched_idx = 9  # Too much dark 'd'
        elif f["is_dark_retraced"]:
            matched_idx = 2  # Dark 'd'
        elif f["internal_loops"] == 0 and f["aspect_ratio"] > 1.4:
            matched_idx = 6  # Printed style 'd'
        elif f["lower_density"] > 0.45:
            matched_idx = 10 # Wide lower extension
        elif f["upper_density"] > 0.45:
            matched_idx = 11 # Ascending upper loop
        else:
            matched_idx = 3  # Light / Open loop 'd'

    elif char_key == 'c':
        if f["aspect_ratio"] > 1.35 or f["stroke_fullness"] < 0.16:
            matched_idx = 1  # Vertically squeezed
        elif f["aspect_ratio"] < 0.85:
            matched_idx = 2  # Broad and squeezed
        elif f["mean_intensity"] > 170 or f["stroke_fullness"] < 0.10:
            matched_idx = 3  # Stroke fading / extra stroke
        else:
            matched_idx = 0  # Normal rounded 'c'

    elif char_key == 'b':
        if f["internal_loops"] >= 2:
            matched_idx = 1  # Looped 'b' (over-extended)
        elif f["upper_density"] > f["lower_density"] * 1.35:
            matched_idx = 5  # Curled top
        elif f["aspect_ratio"] > 1.75 and f["is_dark_retraced"]:
            matched_idx = 4  # Long down stroke
        elif f["aspect_ratio"] < 1.1:
            matched_idx = 3  # Flat base / angular 'b'
        elif f["right_density"] > f["left_density"] * 1.25:
            matched_idx = 6  # Forward slant 'b'
        elif f["lower_density"] > f["upper_density"] * 1.3 and f["internal_loops"] == 1:
            matched_idx = 0  # Larger circle / loop
        else:
            matched_idx = 2  # Closed looped 'b'

    else:
        # Generic multi-feature classifier for rest of alphabets
        if f["complexity_index"] > 0.4:
            matched_idx = min(2, num_traits - 1)
        elif f["internal_loops"] > 0:
            matched_idx = 0
        elif f["is_dark_retraced"]:
            matched_idx = min(1, num_traits - 1)
        else:
            matched_idx = min(3, num_traits - 1)

    matched_idx = min(matched_idx, num_traits - 1)
    return available_traits[matched_idx], f

def lambda_handler(event, context):
    try:
        body = json.loads(event.get("body", "{}"))
        image_bytes = base64.b64decode(body.get("image_base64", ""))
        target_char = body.get("target_char", "A")
        
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        trait, metrics = predict_trait(img_np, target_char)
        
        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({
                "target_char": target_char,
                "predicted_trait": trait,
                "metrics": metrics
            })
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
