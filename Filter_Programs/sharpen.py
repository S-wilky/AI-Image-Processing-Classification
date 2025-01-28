from PIL import Image, ImageFilter
import matplotlib.pyplot as plt

def apply_blur_filter(image_path):
    try:
        img = Image.open(image_path)
        img_resized = img.resize((128, 128))
        img_sharpened = img_resized.filter(ImageFilter.SHARPEN)

        plt.imshow(img_sharpened)
        plt.axis('off')
        plt.savefig("sharpened_image.png")
        print("Processed image saved as 'sharpened_image.png'.")

    except Exception as e:
        print(f"Error processing image: {e}")

if __name__ == "__main__":
    image_path = "dog.jpg"  # Replace with the path to your image file
    apply_blur_filter(image_path)