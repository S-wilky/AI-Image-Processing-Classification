from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
import matplotlib.pyplot as plt

def apply_spacey_filter(image_path):
    try:
        img = Image.open(image_path)
        img_resized = img.resize((128, 128))
        
        # Enhance the colors
        enhancer = ImageEnhance.Color(img_resized)
        img_colored = enhancer.enhance(2.5)

        # Apply Gaussian blur for a dreamy effect
        #img_blurred = img_colored.filter(ImageFilter.GaussianBlur(radius=1))

        # Add random noise
        img_array = np.array(img_colored)
        noise = np.random.normal(loc=0, scale=30, size=img_array.shape)  # Adjust scale for noise intensity
        img_noisy = np.clip(img_array + noise, 0, 255).astype('uint8')  # Add noise and clip values
        img_noisy = Image.fromarray(img_noisy)

        plt.imshow(img_noisy)
        plt.axis('off')
        plt.savefig("spacey_image.png")
        print("Processed image saved as 'spacey_image.png'.")

    except Exception as e:
        print(f"Error processing image: {e}")

if __name__ == "__main__":
    image_path = "dog.jpg"  # Replace with the path to your image file
    apply_spacey_filter(image_path)