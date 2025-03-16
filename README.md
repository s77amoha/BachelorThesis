# An Analysis of Image Alignment Methods for Image Collections with Large Pose Variation

This repository contains the codebase and resources used in the thesis titled **"An Analysis of Image Alignment Methods for Image Collections with Large Pose Variation"**. The thesis explores the limitations of the Neural Congealing framework and investigates whether clustering techniques can improve its accuracy and robustness when handling image collections with significant pose variations.

## Repository Overview

The codebase is organized into several directories, each containing scripts, datasets, and results related to specific tasks. Below is a summary of the contents:

### 1. **Neural_Congealing**
   - This directory contains the implementation of the Neural Congealing framework used for joint image alignment.

### 2. **2D_datasets**
   -  The `2D_datasets` directory contains sample datasets that were rotated using a rotater script along the X-axis, along with the corresponding results.
   
### 2. **3D_datasets**
   - Blender was used to perform rotations along both Y-axis and Z-axis. The results are also included.

### 4. **Clustering**
   - This directory contains the implementation of the k-means clustering algorithm used to cluster datasets into `k` clusters.
   - **Datasets**: The `clustering` directory includes all datasets used in the Atlas initialization section of the thesis, along with clustering results.

### 5. **Eval**
   - An evaluation script is provided to measure the similarity between unlabeled images using cosine similarity.
   - **Results**: The `eval` directory contains example outputs and quantitative results presented in the thesis.
