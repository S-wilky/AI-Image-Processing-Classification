# AI Image Processing and Classification Project

1. Select a new Image:


# Part 1: Using the Basic Classifier and Implementing Grad-CAM
### 1. Find an Image:
- dog.jpg from Unsplash
### 2. Run the Basic Classifier:
Top-3 Predictions:
1: timber_wolf (0.27)
2: borzoi (0.12)
3: African_hunting_dog (0.10)

I don't have much previous experience with writing my own Python code, but the explanations given by AI seem pretty easy to understand, especially since the functions used are well named.
I used the Grimoire GPT on chatGPT for this assignment.

### 3. Implement Grad-CAM:
Prompt an AI to help you generate code to implement the Grad-CAM heatmap overlay. Example Prompt:
"How can I add Grad-CAM to my image classifier to visualize the areas of the image the model focuses on?"

Add the Grad-CAM functionality to the base_classifier.py program.

# There were some issues with the initial AI implementation of Grad-CAM which took some time to debug.

### 4. Understand Grad-CAM:
It was interesting to learn about Grad-CAM from a high level perspective. I think this step should come before step 3.

### 5. Analyze the Heatmap:
Run the updated classifier with Grad-CAM on your image.
Identify which parts of the image the classifier focuses on most heavily.
Record your observations.
# Part 2: Experimenting with Image Occlusion
Generate Occlusion Ideas:
Prompt the AI to suggest three ways to occlude an image to obscure the areas identified in the Grad-CAM heatmap. Example Prompt:
"What are three ways to occlude an image, such as adding a black box or blurring parts of it?"

Implement Occlusions:
Modify the base_classifier.py program to implement the three occlusions suggested by the AI.
Each occlusion should target the area highlighted by the Grad-CAM heatmap.
Test the Occlusions:
Run the classifier on each occluded image.
Record the top-3 predictions and confidence scores for each occlusion.
Analyze the Results:
Compare the classifier's performance on the original image vs. the occluded versions.
Answer the following questions:
Did the classifier struggle to classify the occluded images?
Which occlusion had the greatest impact on performance?
Part 3: Creating and Experimenting with Image Filters
Explore Filter Ideas:
Start with the provided basic_filter.py program, which applies a simple blur to an image. This version of the program does not include comments or explanations.
Prompt the AI to explain what each line of code in the program does. Example Prompt:
"Explain what each line of this Python program does."

If you do not have previous experience with Python, comment on whether the AI's explanations make sense to you. If you do have Python experience, comment on whether the AI's explanation agrees with your interpretation.
Prompt the AI to suggest three alternative filters to apply. Example Prompt:
"What are three different filters I can apply to an image, such as edge detection or sharpening?"

Start with the provided basic_filter.py program, which applies a simple blur to an image.
Prompt the AI to suggest three alternative filters to apply. Example Prompt:
"What are three different filters I can apply to an image, such as edge detection or sharpening?"

Implement Filters:
Modify the basic_filter.py program to implement the three new filters suggested by the AI.
Design Your Own Artistic Filter:
Use creative prompts to direct the AI to develop an artistic filter. Example Prompt:
"Modify the program to create a filter that makes the image look 'deep fried,' with exaggerated colors and noise."

Experiment with different ideas and iterate on the filter until you're satisfied with the result.
   

