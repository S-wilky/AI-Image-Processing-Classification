import cv2

# Load the image
image = cv2.imread("dog.jpg")

# Define the region of interest (ROI)
x, y, w, h = 450, 750, 800, 1300  # Top-left corner (x, y) and width/height
roi = image[y:y+h, x:x+w]

# Apply Gaussian blur to the ROI
blurred_roi = cv2.GaussianBlur(roi, (15, 15), 0)

# Replace the ROI in the original image with the blurred version
image[y:y+h, x:x+w] = blurred_roi

# Save or display the result
cv2.imwrite("blurred_image.jpg", image)