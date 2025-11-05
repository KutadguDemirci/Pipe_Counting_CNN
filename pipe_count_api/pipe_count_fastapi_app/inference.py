import torch
import cv2
import io
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

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
    model.to(device)
    model.eval()
    return model

# --- Prediction with visual title banner ---
def predict(model, image_path, device, confidence_threshold=0.5):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Preprocess
    transform = get_transform()
    image_tensor = transform(Image.fromarray(image)).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        predictions = model(image_tensor)
    pred = predictions[0]
    keep = pred["scores"] > confidence_threshold
    boxes = pred["boxes"][keep].cpu().numpy()
    count = len(boxes)

    # Draw bounding boxes
    annotated = image.copy()
    for box in boxes:
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # --- Create a title banner ---
    title_text = f"Pipes detected: {count}"
    banner_height = 60
    banner = np.ones((banner_height, annotated.shape[1], 3), dtype=np.uint8) * 255  # white banner
    font_scale = 1.2
    thickness = 3
    text_size = cv2.getTextSize(title_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
    text_x = (banner.shape[1] - text_size[0]) // 2
    text_y = (banner.shape[0] + text_size[1]) // 2
    cv2.putText(banner, title_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)

    # --- Stack banner on top of image ---
    final_image = np.vstack((banner, annotated))

    # Convert back to PIL and bytes
    final_pil = Image.fromarray(final_image)
    img_byte_arr = io.BytesIO()
    final_pil.save(img_byte_arr, format="PNG")
    img_byte_arr.seek(0)

    return img_byte_arr, count
