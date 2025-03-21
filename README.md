This project implements unsupervised clustering and evaluation techniques for 4D-STEM (Scanning Transmission Electron Microscopy) Orientation Mapping, as described in the article: 'Unsupervised Multi-Clustering and Decision-Making Strategies for 4D-STEM Orientation Mapping'.

The project focuses on clustering diffraction patterns acquired from 4D-STEM experiments. It includes four main scripts:

1. '00_clustering_nmf_kConponentLoss/00_clustering_NMF_kConponentLoss.py' Purpose: Determines the optimal number of clusters by applying Non-negative Matrix Factorization (NMF) to diffraction patterns.
    Description:
     1. Loads a dataset of diffraction patterns and reshapes them (4d-24).
     2. Performs NMF on the data using GPU computation.
     3. Calculates the reconstruction error (mean absolute difference) for different numbers of components .
     4. Identifies the optimal number of components using the knee point detection method.
     5. Saves the result as .npy and .png files.

2. '01_clustering_NMF_iqa/00_IQA_final_version.py' Purpose: Evaluates the quality of clustering results by comparing typical diffraction patterns of each cluster using various IQA (Image Quality Assessment) metrics.
    Description:
     1. Computes IQA metrics: PSNR, SSIM, MDSI, and GMSD.
     2. Generates an IQA matrix comparing each pair of clusters.
     3. Highlights the best clustering results and saves them as .npy files and .png heatmaps.
   
3. '02_clusteringwithcolor_overlapping_returnRawdata/00_final_test_firstSecond_overlapping.py' Purpose: analyze and visualize diffraction patterns (DPs) by clustering them based on their features, assigning specific colors to each cluster, and studying the evolution of overlapping regions with varying thresholds. Additionally, it visualizes the Non-Negative Matrix Factorization (NMF) results alongside their corresponding raw diffraction patterns
    Description:
     1. Clustering with Color Assignment: 
       Each diffraction pattern is assigned to a cluster, and a specific color is assigned to each cluster for visualization.
       The first and second most dominant clusters for each diffraction pattern are identified, and overlapping regions are highlighted.

     2. Overlapping Evolution :
      The transparency of overlapping regions is studied by varying the threshold from 75% to 95%.
      This helps in understanding how the overlapping regions evolve with different thresholding levels.

     3. Visualization of NMF and Raw Diffraction Patterns:
      The NMF results are visualized alongside their corresponding raw diffraction patterns.
       Image Quality Assessment (IQA) metrics such as PSNR, SSIM, MDSI, and GMSD are computed to compare the NMF results with the raw diffraction patterns.

4. 'clustering_NMF_dp_clustering' Purpose : selecting the brightest diffraction patterns and creating mapping of the results, given the selectec clustering number
    Description:
     1. 'clustering_NMF_dp_clustering/01_clustering_NMF.py': 
       Performs NMF-based clustering on diffraction patterns.
       Assigns each diffraction pattern to a cluster and saves the results.
       Generates the W and H matrices from NMF decomposition.

     2. 'clustering_NMF_dp_clustering/01_montage.py':
       Creates a montage of the clustered diffraction patterns and their corresponding raw patterns.
       Visualizes the results in a grid format for easy comparison.

     3. 'clustering_NMF_dp_clustering/04_choose_Maxlight.py':
       Selects the brightest diffraction pattern from each cluster based on the average pixel intensity.
       Saves the selected patterns for further analysis.
