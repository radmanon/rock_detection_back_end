# webcam_detect.py

import cv2
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms as T

def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def transform_frame(frame):
    transform = T.Compose([T.ToTensor()])
    return transform(frame).unsqueeze(0)

def draw_boxes(frame, boxes, scores, threshold=0.85, size_threshold=2500):
    for box, score in zip(boxes, scores):
        if score >= threshold:
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            area = width * height

            label = f"Bigger - {score:.2f}" if area > size_threshold else f"{score:.2f}"

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            cv2.putText(frame, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return frame

# def run_camera(model_path = "/mnt/data/rock_detector_epoch11.pth"):
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = get_model(num_classes=10)
#     model.load_state_dict(torch.load(model_path, map_location=device))
#     model.to(device)
#     model.eval()

#     # doesnt work on render
#     # cap = cv2.VideoCapture(0)  # 0 = default webcam

#     print("🎥 Starting live rock detection... Press 'q' to quit.")
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Convert BGR to RGB
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         input_tensor = transform_frame(rgb_frame).to(device)

#         with torch.no_grad():
#             output = model(input_tensor)[0]

#         boxes = output['boxes'].cpu()
#         scores = output['scores'].cpu()

#         frame = draw_boxes(frame, boxes, scores)

#         # In production: skip GUI display
#         # For Render, do nothing or log
        
#         # cv2.imshow("Rock Detection (Live)", frame)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()
#     print("🛑 Detection stopped.")

# ✅ Add this function at the bottom so /video_feed can use it
def run_frame(frame, model_path='/mnt/data/rock_detector_epoch11.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not hasattr(run_frame, "model"):
        run_frame.model = get_model(num_classes=10)
        run_frame.model.load_state_dict(torch.load(model_path, map_location=device))
        run_frame.model.to(device)
        run_frame.model.eval()

    model = run_frame.model

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_tensor = transform_frame(rgb_frame).to(device)

    with torch.no_grad():
        output = model(input_tensor)[0]

    boxes = output['boxes'].cpu()
    scores = output['scores'].cpu()

    return draw_boxes(frame, boxes, scores)

if __name__ == "__main__":
    run_camera()
