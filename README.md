# AI Image Processing and Classification Project

# Part 1: Using the Basic Classifier and Implementing Grad-CAM
### 1. Find an Image:
- dog.jpg from Unsplash
### 2. Run the Basic Classifier:
Top-3 Predictions:
1: timber_wolf (0.27)
2: borzoi (0.12)
3: African_hunting_dog (0.10)

I don't have much previous experience with writing my own Python code, but the explanations given by AI seem pretty easy to understand, especially since the functions used are well named.

### 3. Implement Grad-CAM:
There were some issues with the initial AI implementation of Grad-CAM which took some time to debug(if I don't make it, this is why I'm late). I ended up just looking it up because chatGPT wasn't helping, and I used the solution on https://keras.io/examples/vision/grad_cam/

I ended up saving it in "diy_heatmap.py" because I wanted to save the base_classifier (as "provided_classifier.py").
"base_classifier.py" and "test_classifier.py" were both my attempt at using chatGPT to solve the problem. The back and forth with chatGPT was quite frustrating because it would keep acting like I provided the incorrect code, and it ended up going in a circle.

### 4. Understand Grad-CAM:
It was interesting to learn about Grad-CAM from a high level perspective. I think this step should come before step 3.

### 5. Analyze the Heatmap:
Grad-CAM seems to focus mostly on the neck, back of the face, and parts of the ear. I thought it would focus more on the snout, so that was a bit surprising to me.

# Part 2: Experimenting with Image Occlusion
### 1. Generate Occlusion Ideas:
Based on the AI feedback, I will be using a black box, blurring, and pixelation.

### 2. Implement Occlusions:
I created these in separate programs for ease of use.

### 3. Test the Occlusions:
Black Box (occluded_image):
1: African_hunting_dog (0.12)
2: muzzle (0.06)
3: Scotch_terrier (0.04)

Blur (blurred_image):
1: Saluki (0.53)
2: borzoi (0.12)
3: Afghan_hound (0.09)

Pixelation (pixelated_image):
1: African_hunting_dog (0.06)
2: cairn (0.03)
3: Irish_wolfhound (0.03)

### 4. Analyze the Results:
Did the classifier struggle to classify the occluded images?
   - The classifier slightly struggled to classify the occluded images, but not as much as I thought it would.
Which occlusion had the greatest impact on performance?
   - Surprisingly, the blurred image had the most different predictions from the original, even though it was the easiest for me to see.

# Part 3: Creating and Experimenting with Image Filters
### 1. Explore Filter Ideas:
I don't have much previous experience with writing my own Python code, but the explanations given by AI seem pretty easy to understand, especially since the functions used are well named.

The 3 filters the AI suggested were:
1. Edge Detection
2. Sharpening
3. Embossing

### 2. Implement Filters:
The 3 filters were pretty poor quality after the resizing, but they all worked!

### 3. Design Your Own Artistic Filter:
The prompt I used:
"Modify the program to create a filter that makes the image look 'spacey' with exaggerated colors and noise."

I liked the idea of the first pass, but it felt too blurry. After removing the blur effect I liked it a lot better.
   
### Additional Notes
After completing all the steps I organized the files into folders. I'm not sure if you plan to test each program, so sorry if that makes it harder for you, but it should be easier to view this way!

# Final Report
I've included my analysis and observations at each step along the way. From the heatmap and occlusion examples I learned that the prediction program's confidence scores were significantly impacted by the occlusion, but the guesses were still surprisingly similar. What surprised me even more was how different the results were for the blur, which was the easiest to see with the human eye. It seemed to rely much more on the general shape, becoming very confident in the Saluki instead of the original predictions.
The filter I developed was intended to make it look like a galaxy in the background, as if it was a van gogh painting. It was semi-successful, but the quality of the render was a bit poor.
Overall, the experience of this project was very interesting, but also a little bit frustrating working with the AI at times. It is fascinating to see how easy it is to edit images with python as well!
