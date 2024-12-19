from flask import Flask, request, jsonify, render_template
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import torchvision.models as models
import torch.nn as nn

app = Flask(_name_)

# Load the ResNet-18 model and modify the last fully connected layer
model = models.resnet18(weights=None)  # Explicitly set weights=None to avoid the warning
model.fc = nn.Linear(model.fc.in_features, 5)  # Change to output 5 classes
model.load_state_dict(torch.load('C:\\Users\\sumug\\Desktop\\S7 project\\my_model.pth'))
model.eval()  # Set the model to evaluation mode

# Image preprocessing pipeline
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to 224x224 for ResNet
    transforms.Grayscale(num_output_channels=3),  # Convert grayscale to RGB by repeating channels
    transforms.ToTensor(),  # Convert image to Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard ResNet normalization
])

@app.route('/')
def home():
    return render_template('index.htm')  # Render the HTML form

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    img = Image.open(io.BytesIO(file.read()))  # Open the image from the uploaded file

    # Preprocess the image (convert grayscale to RGB and resize)
    img_tensor = preprocess(img).unsqueeze(0)  # Add batch dimension

    # Run the model on the input image
    with torch.no_grad():
        output = model(img_tensor)
        _, predicted_class = torch.max(output, 1)

    return jsonify({"prediction": predicted_class.item()})

if _name_ == '_main_':
    app.run(debug=True)