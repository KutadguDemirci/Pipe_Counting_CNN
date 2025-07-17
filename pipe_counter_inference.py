# pipe_counter_inference.py
"""
Minimal script to load a trained pipe detection model and count pipes in an image.
Usage:
    python pipe_counter_inference.py --model_path final_pipe_model.pth --image_path water_pipes_cover1.jpg
"""
import torch
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn, fasterrcnn_resnet50_fpn_v2
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

# --- Pipe counting ---
def predict_pipes(model, image_path, device, confidence_threshold=0.5):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image)
    transform = get_transform()
    image_tensor = transform(image_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        predictions = model(image_tensor)
    pred = predictions[0]
    keep = pred['scores'] > confidence_threshold
    boxes = pred['boxes'][keep].cpu().numpy()
    scores = pred['scores'][keep].cpu().numpy()
    count = len(boxes)
    return count, boxes, scores, image

# --- Optional: Visualize ---
def visualize_predictions(image, boxes, scores, save_path=None):
    img = image.copy()
    for box in boxes:
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    plt.figure(figsize=(12, 8))
    plt.imshow(img)
    plt.title(f'Pipe Detection Results (Total: {len(boxes)})')
    plt.axis('off')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model_for_inference('final_pipe_model_v2.pth', device)
    image_path = 'water_pipes_cover1.jpg'
    count, boxes, scores, image = predict_pipes(model, image_path, device, confidence_threshold=0.5)
    print(f'Number of pipes detected: {count}')
    visualize_predictions(image, boxes, scores)

