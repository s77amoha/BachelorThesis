import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
import os

# 1. Load a pretrained model (e.g., VGG16) without the classification head
model = VGG16(weights='imagenet', include_top=False, pooling='avg')

# 2. Define a function to extract deep features from a PNG image
def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))  # Resize to input size
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)  # Preprocess for the model
    features = model.predict(img_data)  # Extract features
    return features.flatten()  # Flatten the features into a 1D array

# 3. Load and extract features from multiple PNG images
image_folder = "cat_face/by_5deg"  # Replace with your directory containing PNG images
features = []
image_names = []

# Loop through all PNG files in the specified folder
for filename in os.listdir(image_folder):
    if filename.endswith(".png"):
        img_path = os.path.join(image_folder, filename)
        try:
            feature_vector = extract_features(img_path, model)
            features.append(feature_vector)
            image_names.append(filename)  # Save the image name for reference
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue

# 4. Ensure features are available before calculating similarity
if features:
    # Convert the list of features to a NumPy array
    features = np.array(features)

    # Compute the global cosine similarity matrix
    cosine_sim_matrix = cosine_similarity(features)  # Pairwise cosine similarity

    # 5. Calculate the global cosine similarity score
    global_cosine_similarity_score = np.mean(cosine_sim_matrix[np.triu_indices_from(cosine_sim_matrix, k=1)])

    # Display results
    print(f"Global Cosine Similarity Score: {global_cosine_similarity_score}")

    # Optionally, print the cosine similarity matrix for reference
    #print("Cosine Similarity Matrix:")
    #print(cosine_sim_matrix)

    # Print image names and their corresponding feature vectors
    #for name, feature in zip(image_names, features):
        #print(f"{name}: {feature}")
else:
    print("No valid features were extracted from the images.")