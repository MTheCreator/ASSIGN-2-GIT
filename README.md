# Assignment 2: Graphics and Interactive Techniques

This repository contains the implementation of Assignment 2, which consists of three main parts and questionnaire responses.

## Group Members
- BADDOU Mounia
- FRI Zyad
- KABLY Malak

## Repository Structure

### Part 1: Questionnaire
- The responses to the questionnaire can be found in `part_1.pdf`

### Challenge 1: Tomato Segmentation using U-Net
- Located in `/challenge1/`
- Main implementation: `Tomato_UNet.ipynb`
- Dataset: `Tomato_dataset.zip`
- Results and model outputs can be found in `/challenge1/results/`
  - Trained model: `tomato_unet_model.pth`
  - Training visualizations: `training_history.png`, `iou_history.png`
  - Prediction examples: `prediction_*.png`
  - Sample data visualization: `sample_data.png`

### Part 2: Image Processing and Filtering
- Located in `/part2/`
- Implementation notebook: `part2.ipynb`
- Contains implementations of:
  - Linear transformations
  - Image filtering
  - Global thresholding
  - Edge detection
  - Histogram equalization
  - 2D convolution
- Sample image: `tulips.png`

### Part 3: Machine Learning-based Image Segmentation
- Located in `/part3/`
- Contains two implementation notebooks:
  1. `part3.1.ipynb`: K-means clustering based segmentation
  2. `part3.2.ipynb`: SVM-based classification segmentation
- Output example: `segmented_tulips.png`

## Running the Code
Each notebook is self-contained and includes all necessary imports and function definitions. The notebooks should be run in order of their cell arrangement.
