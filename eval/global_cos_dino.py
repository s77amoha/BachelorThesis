import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from torchvision import transforms
from PIL import Image
import os
from torch.hub import load

#  Load DINO model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load('facebookresearch/dino:main', 'dino_vits16', pretrained=True)
model.eval().to(device)
"""
This script computes the global cosine similarity score for a set of images using the DINO (Self-Distillation with No Labels) model,
specifically the dino_vits16 variant. The score measures the average pairwise similarity between the feature representations of all images in a given dataset.
 
 the dataset can be passed by its path in the image_folder variable. 
 
"""


image_folder = '/home/judai/Downloads/clusters/kang_k3'  # Path of the folder containing the images


# Extract the CLS token
def extract_cls_features(img_tensor, model):

    with torch.no_grad():
        # Forward pass
        output = model(img_tensor)

        # Extract CLS token or final layer's embeddings
        return output.squeeze()  # Remove batch dimension


# Preprocess the image and extract features
def extract_features_dino(img_path, model):
    # Define transformations for the image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize
        transforms.ToTensor(),  # Convert image to a tensor
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
    ])

    # Load and preprocess the image
    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)

    # Extract CLS token features
    features = extract_cls_features(img_tensor, model)
    return features.cpu().numpy().flatten()


features = []

# Loop over all PNG files
for filename in os.listdir(image_folder):
    if filename.endswith(".png"):
        img_path = os.path.join(image_folder, filename)

        feature_vector = extract_features_dino(img_path, model)
        features.append(feature_vector)


features = np.array(features)

cosine_sim_matrix = cosine_similarity(features)  # Pairwise cosine similarity

# Global cosine similarity score
global_cosine_similarity_score = np.mean(cosine_sim_matrix[np.triu_indices_from(cosine_sim_matrix, k=1)])

# Print score
print(f"Global Cosine Similarity Score: {global_cosine_similarity_score}")



