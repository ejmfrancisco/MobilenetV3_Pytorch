import torch
import json
from torchvision import transforms
from PIL import Image
from model.mobilenetv3_pytorch import MobileNetV3  # Make sure this import matches your project's structure

# Load class_to_idx from JSON
with open('class_to_idx.json', 'r') as json_file:
    class_to_idx = json.load(json_file)

# Invert the dictionary to get idx_to_class
idx_to_class = {v: k for k, v in class_to_idx.items()}

# Define the same transformations as used during training (excluding augmentations)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Instantiate and load the trained model
num_classes = len(idx_to_class)  # Number of classes based on idx_to_class
model = MobileNetV3(config_name="large", classes=num_classes)
model_path = './best_model.pth'  # Update this path to where your model is saved

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)

def predict_image(image_path, model, transform, device, idx_to_class):
    image = Image.open(image_path).convert('RGB')  # Ensure image is in RGB format
    image = transform(image).unsqueeze(0)  # Add batch dimension
    image = image.to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    predicted_class = idx_to_class[predicted.item()]
    return predicted_class

# Example usage
image_path = "./8.png"  # Update this path to your image
prediction = predict_image(image_path, model, transform, device, idx_to_class)
print(f"Predicted class: {prediction}")