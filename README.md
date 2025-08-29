# Image Processing Assignments

This repository contains implementations of various **image processing techniques** using **Python, OpenCV, NumPy, and Matplotlib**. The scripts cover interpolation, thresholding, contrast enhancement, histogram equalization, CLAHE, and more.  

## 📂 Repository Structure

```
├── Q1b_nearest_neighbour_interpolation.py   # Nearest-neighbour interpolation
├── Q1d_Bicubic_interpolation.py             # Bicubic interpolation
├── Q1f_imagedownandupsampling.py            # Downsampling and upsampling
├── Q2a_manual_thresholding.py               # Manual thresholding
├── Q2b_Otsu_Threshold.py                    # Otsu's thresholding
├── Q2c_Local_Adaptive.py                    # Adaptive thresholding
├── Q3a_Linear_contrast_streching.py         # Linear contrast stretching
├── Q3b_Histogram_equalization.py            # Histogram equalization
├── Q3c_CLAHE.py                             # Contrast Limited Adaptive Histogram Equalization
├── Q3d_histogram_matching.py                # Histogram matching
├── Q3d_bin_tunning.py                       # Histogram bin tuning
├── assignment_1_InterpThreshHist.pdf        # Assignment report
├── Q1input.png                              # Sample input image
├── Q1output.png                             # Sample output image
└── README.md                                # Project documentation
```

## 🚀 Features

- **Interpolation**  
  - Nearest neighbour  
  - Bicubic interpolation  
  - Downsampling & upsampling  

- **Thresholding**  
  - Manual thresholding  
  - Otsu’s method  
  - Local adaptive thresholding  

- **Contrast Enhancement**  
  - Linear contrast stretching  
  - Histogram equalization  
  - CLAHE (Contrast Limited Adaptive Histogram Equalization)  
  - Histogram matching & bin tuning  

## 🛠️ Dependencies

Make sure the following libraries are installed:

```bash
pip install numpy opencv-python matplotlib imageio pillow
```

## ▶️ Usage

Run any script directly with Python:

```bash
python Q2b_Otsu_Threshold.py
```

Each script loads an image, applies the respective image processing technique, and displays both the **original** and **processed images**, along with their histograms (where applicable).  

## 📖 Assignment Notes

- This repository was created as part of coursework on **Digital Image Processing**.  
- The included `.pdf` provides explanations, mathematical derivations, and results.  

---

✨ This repo is a handy reference for beginners in **image processing with Python and OpenCV**.  
