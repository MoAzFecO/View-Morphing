# View Morphing

## Overview

This project implements view morphing in Python using Dlib for facial landmark detection and OpenCV for image processing and triangulation. The goal is to morph one face into another through a series of intermediate frames, creating a smooth transition.

## Requirements

- Python
- OpenCV (cv2)
- Dlib
- NumPy
- os
- PIL

## Setup

1. **Install dependencies:**
   ```bash
   pip install numpy
   ```
   
2. **Download a Dlib shape predictor model or train one:**
   - For example download the `shape_predictor_68_face_landmarks.dat` file from [here](https://github.com/codeniko/shape_predictor_81_face_landmarks/blob/master/).

3. **Prepare your images:**
   - Place your two unput images into the directory `input_images` 
   - Update the `img1_path` and `img2_path` variables in the `main()` function of `view_morphing.py` with your image filenames.

## Usage

1. **Run the script:**
   ```bash
   python view_morphing.py
   ```
   
2. **Output:**
   - The script will generate a series of images and a gif in the `./output/` directory, showing the morphing process from the first image to the second.

## Key Functions

- `detect_keypoints_face(img)`: Detects facial landmarks using Dlib's pre-trained model.
- `draw_triangulation(img, subdiv, delaunay_color)`: Draws the Delaunay triangulation on the image.
- `get_triangulation(img, points)`: Computes the Delaunay triangulation for a set of points.
- `apply_affine_transform(warp_image, src, srcTri, dstTri)`: Applies an affine transformation to warp the image.
- `morph_triangle(img1, img2, img, t1, t2, t, alpha)`: Morphs a triangle from img1 and img2 into an intermediate triangle.

## Customization

- **Feature Points Detection:**
  - You can customize the feature points by modifying the `detect_keypoints_face` function or adding additional points manually.
  - In the directory `features_points`, the python file `features_points.py` allow to add manually features points.
  
- **Number of Frames:**
  - Adjust the `frames` variable in the `main()` function to change the number of intermediate frames generated.

## Results

The results are organized into four directories, each demonstrating different inputs and outcomes of the view morphing algorithm:

- **output_Joconde**: This directory contains results generated using images of the Joconde. The algorithm detected facial features, and the output includes a GIF along with individual images showcasing various alpha values for morphing.

- **output_Joconde_manual_points**: In addition to automatically detected facial features, manual key points were added to enhance the accuracy of the morphing. The resulting images show improved transitions and are saved both as a GIF and as separate frames.

- **output_Jules**: This directory contains results using images of my face. Due to size constraints, only five images were saved. 

- **output_einstein**: Using images of Einstein as input, this directory showcases the results of the morphing process, including a GIF and individual images representing different alpha values.
