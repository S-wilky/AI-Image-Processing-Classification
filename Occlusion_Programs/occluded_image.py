import cv2

# Load the image
image = cv2.imread("dog.jpg")

# Define the coordinates and size of the black box
start_point = (450, 750)
end_point = (1250, 2050)

# Add a black rectangle
occluded_image = cv2.rectangle(image.copy(), start_point, end_point, (0, 0, 0), -1)

# Save or display the result
cv2.imwrite("occluded_image.jpg", occluded_image)