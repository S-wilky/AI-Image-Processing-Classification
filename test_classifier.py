import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import ToPILImage
import cv2

from torchvision import models, transforms
from PIL import Image

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

# Define the Grad-CAM class
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        # Register hooks
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_image, target_class):
        # Forward pass
        output = self.model(input_image)
        self.model.zero_grad()

        # Backward pass
        target = output[:, target_class]
        target.backward()

        # Get the gradients and activations
        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]

        # Compute weights
        weights = np.mean(gradients, axis=(1, 2))  # Global Average Pooling

        # Compute Grad-CAM
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = np.maximum(cam, 0)  # ReLU
        cam = (cam - cam.min()) / (cam.max() - cam.min())  # Normalize
        cam = np.uint8(cam * 255)  # Scale to 0-255
        return cam

# Example usage
if __name__ == "__main__":

    # Load a pretrained model and an example image
    model = models.resnet18(weights=True) # switch weights and pretrained
    model.eval()
    target_layer = model.layer4[1].conv2  # Select a target layer

    # Preprocess the image
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image_path = "dog.jpg"
    image1 = Image.open(image_path)  # Replace with your image path
    input_tensor = preprocess(image1).unsqueeze(0)

    # Instantiate Grad-CAM
    grad_cam = GradCAM(model, target_layer)

    # Classify the image
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    # Creade, Decode, and Print predictions
    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(predictions, top=3)[0]
        
    print("Top-3 Predictions:")
    for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
        print(f"{i + 1}: {label} ({score:.2f})")

    # Generate Grad-CAM heatmap
    target_class = np.argmax(predictions) # 243  # Replace with your target class index
    heatmap = grad_cam.generate(input_tensor, target_class)
    imagearray = grad_cam.generate(input_tensor, target_class)

    #print array types
    print("Image array shape:", imagearray.shape)
    print("Heatmap array shape:", heatmap.shape)

    # Overlay heatmap on the original image
    #heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET) #original code
    #overlayed_image = cv2.addWeighted(np.array(image), 0.5, heatmap, 0.5, 0) #original code


    # Display the result
    #plt.imshow(overlayed_image)
    plt.axis("off")
    plt.show()
