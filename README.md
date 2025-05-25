<h1 align="center" style="color: #4682B4;">Assignment 2: Graphics and Interactive Techniques</h1>

This repository contains the complete implementation of Assignment 2, which includes three main parts, two challenges, and questionnaire responses.

---

## Group Members

- BADDOU Mounia  
- FRI Zyad  
- KABLY Malak  

---

## Repository Structure

### Part 1: Questionnaire

The responses to the questionnaire can be found in the root directory:  
`part_1.pdf`

---

### Challenge 1: Tomato Segmentation using U-Net

- Directory: `/challenge1/`  
- Main implementation: `Tomato_UNet.ipynb`  
- Dataset: `Tomato_dataset.zip`  
- Results in `/challenge1/results/`:  
  - Trained model: `tomato_unet_model.pth`  
  - Training visualizations: `training_history.png`, `iou_history.png`  
  - Prediction examples: `prediction_0.png` through `prediction_4.png`  
  - Sample data visualization: `sample_data.png`  

---

### Challenge 2: China Map Segmentation with Custom Heat Map

- Directory: `/challenge2/`  
- Main implementation: `Challenge2.ipynb`  
- Required files:  
  - Input map: `china_map.png`  
  - City values data: `china_map_values.csv`  
- Implemented features:  
  - Advanced image segmentation with K-Means clustering  
  - Adaptive thresholding and morphological operations  
  - Custom heat map generation with a 6-color gradient  
  - Transparent overlay techniques  
  - Comprehensive statistical analysis and visualization  
- Results include:  
  - Segmented map with heat map overlay  
  - Processing pipeline visualization  
  - Quality metrics and performance analysis  
  - Publication-ready visualizations  

---

### Part 2: Image Processing and Filtering

- Directory: `/part2/`  
- Implementation notebook: `part2.ipynb`  
- Includes:  
  - Linear transformations  
  - Image filtering  
  - Global thresholding  
  - Edge detection  
  - Histogram equalization  
  - 2D convolution  
- Sample image: `tulips.png`  

---

### Part 3, 4, and 5:

- Directory: `/part3/`  
- Implementation notebooks:  
  1. `part3.1.ipynb`: K-means clustering-based segmentation (Part 3.1)  
  2. `part3.2.ipynb`: SVM-based classification segmentation (Part 3.2.1-3.2.5)  
  3. `From3.2.6To5.4.ipynb`: Contains:
     - Remaining Part 3 sections (3.2.6 onwards)
     - Part 4: Image classification
     - Part 5: Rendering
- Output: `segmented_tulips.png`   

---

## Running the Code

Each notebook is self-contained and includes all required imports and function definitions. To run the notebooks:

1. Ensure required files are in the same directory as the notebook (e.g., `china_map.png` and `china_map_values.csv` for Challenge 2).  
2. Run all cells in order from top to bottom.    

