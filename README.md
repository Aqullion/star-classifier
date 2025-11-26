# star-classifier
This project implements and compares two distinct Machine Learning approaches—Feature Engineering (Random Forest) and Deep Learning (1D-CNN)—to classify TESS (Transiting Exoplanet Survey Satellite) light curves into 'variable' or 'constant' stellar categories.

The primary goal was to test the efficiency of hand-crafted features versus raw sequence data on a large, real-world astronomical dataset.

Highlights from the projects are:

Scale: Developed a robust, GPU-accelerated pipeline to process and model $\mathbf{10,000}$ TESS light curves, successfully handling a $\mathbf{3 \text{ GB}}$ high-dimensional sequence array.

The top 3 predictors determined by the RF Model:

flux_range (Highest Importance): Directly measures the $\mathbf{amplitude}$ of the star's brightness change.

kurtosis: Measures the $\mathbf{sharpness}$ of the light curve features (e.g., sharp dips in eclipsing binaries).

std_flux: Measures the average deviation or variability around the mean.

For the visual check the results folder to see rf_feature_importance.png

Requirement libraries in the project are written in the file check to see which one are you missing.

Download the processed data since the file size is large here is the download link
Link:https://drive.google.com/drive/folders/1VECKUuQCDIwNlZ2-lTLdJDKEqnVmYe73?usp=drive_link
