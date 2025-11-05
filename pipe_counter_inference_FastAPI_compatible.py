# app/inference.py

import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import io

# --- Model creation ---
def create_model(num_classes=2):
    model = fasterrcnn_resnet50_fpn_v2(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# --- ImageNet normalization ---
def get_transform():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

# --- Load model ---
def load_model_for_inference(model_path, device):
    model = create_model(num_classes=2)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.roi_heads.detections_per_img = 500
    model.to(device)
    model.eval()
    return model

# --- Inference on uploaded image bytes ---
def predict_pipes_bytes(model, image_bytes, device, confidence_threshold=0.5):
    # Load image from bytes
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_np = np.array(image)
    
    transform = get_transform()
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        predictions = model(image_tensor)

    pred = predictions[0]
    keep = pred['scores'] > confidence_threshold
    boxes = pred['boxes'][keep].cpu().numpy()
    scores = pred['scores'][keep].cpu().numpy()
    count = len(boxes)
    
    # Draw boxes
    img = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    for box in boxes:
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
    cv2.putText(img, f"Count: {count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    
    _, img_encoded = cv2.imencode('.jpg', img)
    return img_encoded.tobytes(), count, boxes, scores
