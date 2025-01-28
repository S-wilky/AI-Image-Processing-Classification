import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import cv2

model = MobileNetV2(weights="imagenet")

def grad_cam(image_path, model, last_conv_layer_name, pred_index=None):
    """Compute Grad-CAM heatmap."""
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    # Get the model's predictions
    preds = model.predict(img_array)
    if pred_index is None:
        pred_index = np.argmax(preds[0])  # Predicted class index

    # Define a gradient tape
    with tf.GradientTape() as tape:
        last_conv_layer = model.get_layer(last_conv_layer_name)  # Get the last conv layer
        tape.watch(last_conv_layer.output)
        preds = model(img_array)  # Forward pass
        class_channel = preds[:, pred_index]  # Extract the predicted class score

    # Compute gradients
    grads = tape.gradient(class_channel, last_conv_layer.output)

    # Pool gradients across spatial dimensions
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Multiply the pooled gradients by the feature map
    last_conv_layer_output = last_conv_layer.output[0]
    heatmap = tf.reduce_mean(pooled_grads * last_conv_layer_output, axis=-1)

    # Normalize the heatmap
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    return heatmap.numpy()

def display_grad_cam(image_path, heatmap, alpha=0.4):
    """Overlay the Grad-CAM heatmap on the original image."""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    overlay = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
    return overlay

def classify_image_with_gradcam(image_path):
    """Classify an image and visualize Grad-CAM."""
    try:
        # Classify the image
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)
        decoded_predictions = decode_predictions(predictions, top=3)[0]
        
        # Print predictions
        print("Top-3 Predictions:")
        for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
            print(f"{i + 1}: {label} ({score:.2f})")

        # Compute Grad-CAM heatmap for the top prediction
        pred_index = np.argmax(predictions[0])
        heatmap = grad_cam(image_path, model, last_conv_layer_name="Conv_1", pred_index=pred_index)

        # Display Grad-CAM heatmap
        overlay = display_grad_cam(image_path, heatmap)
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("Original Image")
        plt.imshow(image.load_img(image_path, target_size=(224, 224)))
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.title("Grad-CAM Overlay")
        plt.imshow(overlay)
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error processing image: {e}")

if __name__ == "__main__":
    image_path = "dog.jpg"  
    classify_image_with_gradcam(image_path)
