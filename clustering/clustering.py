import os
import json

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Allow loading of truncated images
import torch
import torch.nn as nn
from torchvision.models import resnet50
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.cluster import KMeans

"""
This script performs feature extraction using a pre-trained ResNet-50 model and clusters images based on their extracted features using K-means clustering.
the feature extractor can also changed but I used a resnet as it performed better than the DINO-Vit but more testing can show otherwise.

The path of the dataset and number of clusters can be defined in the main function


"""
class Resnet50FeatureExtractor(nn.Module):
    def __init__(
            self,
            device
    ):
        super(Resnet50FeatureExtractor, self).__init__()

        # Load pre-trained ResNet50 model and move it to the specified device (CPU/GPU)
        model = resnet50(pretrained=True).to(device=device)
        model.train(mode=False)  # Set the model to evaluation mode

        # Get the names of the nodes in the model's computation graph
        train_nodes, eval_nodes = get_graph_node_names(model)

        print('train_nodes')
        print(train_nodes)
        print('eval_nodes')
        print(eval_nodes)

        # Define the nodes from which to extract features
        return_nodes = {
            'layer4.2.relu_2': 'layer4',  # Extract features from this layer
        }

        # Create a feature extractor that will return features from the specified nodes
        self.feature_extractor = create_feature_extractor(model, return_nodes=return_nodes).to(device=device)

    def forward(self, x):
        # Forward pass through the feature extractor
        x = self.feature_extractor(x)
        return x


class CustomImageDataset(Dataset):
    def __init__(
            self,
            image_dir,
            transform=None,
    ):
        super(CustomImageDataset, self).__init__()
        self.image_dir = image_dir  # Directory containing the images
        self.transform = transform  # Transformations to apply to the images
        self.images = os.listdir(image_dir)  # List of image filenames in the directory

    def __len__(self):
        return len(self.images)  # Return the number of images in the dataset

    def __getitem__(self, item):
        # Get the path to the image
        image_path = os.path.join(self.image_dir, self.images[item])
        # Open the image and convert it to RGB format
        image = Image.open(image_path).convert("RGB")

        # Apply transformations if any
        if self.transform is not None:
            image = self.transform(image)

        # Return the image and its path
        data = {
            'image': image,
            'path': image_path
        }

        return data


def image_feature_extract(
    image_height,
    image_width,
    batch_size,
    device,
    image_dir
):
    # Define the image transformations: resize, convert to tensor, and normalize
    image_transform = transforms.Compose([
        transforms.Resize((image_height, image_width)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0]
        ),
    ])

    # Create a custom dataset using the specified directory and transformations
    custom_image_dataset = CustomImageDataset(
        image_dir,
        transform=image_transform,
    )

    # Create a DataLoader to handle batching and shuffling of the dataset
    image_loader = DataLoader(
        custom_image_dataset,
        batch_size=batch_size,
        num_workers=0,
        shuffle=True
    )

    # Initialize the feature extractor
    fx = Resnet50FeatureExtractor(device=device)

    # Lists to store image paths and extracted features
    image_path_list = []
    image_feature_list = []

    # Iterate over the DataLoader to process images in batches
    for batch_idx, (data) in enumerate(image_loader):
        image_path = data['path']
        image_path_list.append(image_path)

        img = data['image']
        # Extract features from the image using the feature extractor
        extracted_feature = fx(img.to(device=device))

        # Flatten the extracted features and store them in the list
        for i in extracted_feature.keys():
            img_feature = extracted_feature[i].flatten().cpu().detach().numpy()
            image_feature_list.append(img_feature)

    print(len(image_feature_list))
    print('Done feature extraction.')

    return (image_feature_list, image_path_list)


def cluster_images(image_feature_list, image_path_list, k):
    # Convert the list of features to a numpy array
    image_feature_list = np.array(image_feature_list)
    print(image_feature_list.shape)

    print('STARTED CLUSTERING')
    # Define the number of clusters for KMeans
    number_clusters = k
    # Apply KMeans clustering to the features
    kmeans = KMeans(n_clusters=number_clusters, random_state=0).fit(np.array(image_feature_list))
    print(kmeans.labels_)  # Print the cluster labels
    print(kmeans.inertia_)  # Print the inertia (sum of squared distances to centroids)

    # Create a dictionary to store images by their cluster labels
    image_cluster_dict = {}
    for i, m in enumerate(kmeans.labels_):
        image_cluster_dict[f'{m}'] = []

    print('CLUSTER GROUPS')
    # Populate the dictionary with image paths based on their cluster labels
    for i, m in enumerate(kmeans.labels_):
        image_cluster_dict[f'{m}'].append(image_path_list[i])

    # Print the cluster dictionary in a readable format
    print(json.dumps(image_cluster_dict, indent=4, separators=(',', ':')))

    # Determine the maximum number of images in any cluster for plotting
    biggest_len = 0
    for i in image_cluster_dict.keys():
        biggest_len = max(len(image_cluster_dict[i]), biggest_len)

    # Create a subplot to display images from each cluster
    f, axarr = plt.subplots(number_clusters, biggest_len)
    for i in image_cluster_dict.keys():
        for j in range(len(image_cluster_dict[i])):
            im = Image.open(image_cluster_dict[i][j][0])
            axarr[int(i), j].imshow(np.array(im))
            axarr[int(i), j].set_title(f'Cluster {i}')
    plt.show()


if __name__ == '__main__':
    # Define parameters for image processing
    image_height = 512
    image_width = 512
    batch_size = 1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Use GPU if available
    image_dir = r'/home/judai/Desktop/A.I/DL/NC/data/1image/images_cosegmentation/mask/'  # Directory containing images
    k = 2 # number of clusters

    # Perform feature extraction and clustering
    cluster_images(
        *image_feature_extract(
            image_height=image_height,
            image_width=image_width,
            batch_size=batch_size,
            device=device,
            image_dir=image_dir
        ), k
    )