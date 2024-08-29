import cv2
import torch
import numpy as np
from torchvision import models, transforms

# Load the pre-trained model (e.g., Faster R-CNN)
model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Transformation for input images
transform = transforms.Compose([
    transforms.ToTensor(),
])

class TrackedObject:
    def __init__(self, obj_id, box):
        self.id = obj_id
        self.x1, self.y1, self.x2, self.y2 = box
    
    def update(self, box):
        self.x1, self.y1, self.x2, self.y2 = box

class Tracker:
    def __init__(self):
        self.objects = []
        self.next_id = 0
    
    def update(self, box, label, frame):
        # Simplified tracking logic to find the closest tracked object
        obj = self.find_closest_object(box)
        if obj is None:
            obj = TrackedObject(self.next_id, box)
            self.objects.append(obj)
            self.next_id += 1
        
        obj.update(box)
    
    def find_closest_object(self, box):
        # Simplified method to find the closest object to the given box
        if len(self.objects) == 0:
            return None
        # Calculate the centroid of the new box
        new_centroid = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
        min_distance = float("inf")
        closest_obj = None
        for obj in self.objects:
            # Calculate the centroid of the existing object
            obj_centroid = ((obj.x1 + obj.x2) / 2, (obj.y1 + obj.y2) / 2)
            # Calculate Euclidean distance between centroids
            distance = np.sqrt((new_centroid[0] - obj_centroid[0]) ** 2 +
                               (new_centroid[1] - obj_centroid[1]) ** 2)
            if distance < min_distance:
                min_distance = distance
                closest_obj = obj
        return closest_obj if min_distance < 50 else None

def process_video(input_video_path, output_video_path):
    cap = cv2.VideoCapture(input_video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    tracker = Tracker()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Pre-process the frame
        input_tensor = transform(frame).unsqueeze(0)
        
        # Run the model
        with torch.no_grad():
            predictions = model(input_tensor)
        
        # Extract bounding boxes, labels, and scores
        for i in range(len(predictions[0]['boxes'])):
            box = predictions[0]['boxes'][i].cpu().numpy()
            label = predictions[0]['labels'][i].item()
            score = predictions[0]['scores'][i].item()
            
            if score > 0.5:
                # Track the person
                tracker.update(box, label, frame)
        
        # Draw predictions on the frame
        for obj in tracker.objects:
            cv2.rectangle(frame, (int(obj.x1), int(obj.y1)), (int(obj.x2), int(obj.y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {obj.id}", (int(obj.x1), int(obj.y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        out.write(frame)
    
    cap.release()
    out.release()

if __name__ == "__main__":
    input_video = 'input/test_video1.mp4'  # Update with your input video path
    output_video = 'output/results_video1.mp4'  # Update with your output video path
    process_video(input_video, output_video)
