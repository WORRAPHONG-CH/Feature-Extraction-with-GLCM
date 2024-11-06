# Feature Extraction with GLCM

### Steps
    * Post-processing frame from video using erosion, canny edge and get corner point from contour detection.
    * Press "q" to capture save images 
    * Press "e" to extract feature GLCM of each images 
    * Summerize along with graph and save in csv file

###  Grey Level Co-occurrence Matrices (GLCM)
    #### There are 5 features from characterize the texture of images.
    * Contrast: Measures the local variations in the image. High contrast values indicate large differences between neighboring pixel intensities.
    * Dissimilarity: Measures the average difference in intensity between neighboring pixels. High dissimilarity values indicate greater heterogeneity in texture.
    * Homogeneity: Reflects the closeness of the distribution of elements in the GLCM to the GLCM diagonal. High homogeneity values indicate that elements are concentrated along the diagonal, suggesting a more uniform texture.
    * Energy (or Angular Second Moment): Represents the orderliness or homogeneity of the image. High energy values indicate more uniform texture.
    *Correlation: Measures the linear dependency between pixel pairs. High correlation values indicate a more predictable texture.