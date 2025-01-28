import cv2

# Load the image
image = cv2.imread("dog.jpg")

# Define the region of interest (ROI)
x, y, w, h = 450, 750, 800, 1300  # Top-left corner (x, y) and width/height
roi = image[y:y+h, x:x+w]

# Pixelate the ROI by resizing down and back up
small = cv2.resize(roi, (10, 10), interpolation=cv2.INTER_LINEAR)  # Downsample
pixelated_roi = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)  # Upsample

# Replace the ROI in the original image with the pixelated version
image[y:y+h, x:x+w] = pixelated_roi

# Save or display the result
cv2.imwrite("pixelated_image.jpg", image)