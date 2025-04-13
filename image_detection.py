import os
import torch
import torchvision
import requests
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms as T
from PIL import Image, ImageDraw

def get_model(num_classes=10):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def download_model_if_needed(model_path):
    if not os.path.exists(model_path):
        print("📦 Downloading model from Dropbox...")
        url = "https://www.dropbox.com/scl/fi/nalx3tv2tqise73b94bnh/rock_detector_epoch11.pth?rlkey=jh38rburh70lff7mzaqd20vuc&st=ogqhk7gg&dl=1"
        response = requests.get(url)
        with open(model_path, "wb") as f:
            f.write(response.content)
        print("✅ Model downloaded to:", model_path)

# This function is used by your FastAPI server
def predict_on_image(image_path, model_path="/mnt/data/rock_detector_epoch11.pth", threshold=0.6):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Download the model if it doesn't exist yet
    download_model_if_needed(model_path)

    model = get_model()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Load and transform image
    image = Image.open(image_path).convert("RGB")
    transform = T.Compose([T.ToTensor()])
    img_tensor = transform(image).to(device).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)[0]

    # Draw boxes
    draw = ImageDraw.Draw(image)
    for box, score in zip(outputs['boxes'], outputs['scores']):
        if score >= threshold:
            x1, y1, x2, y2 = box.tolist()
            area = (x2 - x1) * (y2 - y1)
            label = f"Bigger - {score:.2f}" if area > 15 else f"{score:.2f}"
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            draw.text((x1, y1), label, fill="red")

    # Save result
    os.makedirs("results", exist_ok=True)
    output_path = os.path.join("results", os.path.basename(image_path))
    image.save(output_path)

    return os.path.basename(output_path)  # returns only the file name, like 'q6.jpg'
