


SpatialDiffusion- Predicting Spatial Transcriptomics with Denoising Diffusion Probabilistic Models
---------------------------------------------------------------------------------------------------------
This repository contains the  SpatialDiffusion code and online data for in silico generation and interpolation. It also contains the code to evaluate and visualize the  results. 
* [Summary](#Summary)
* [Installation Requisites](#Installation-Requisites )
* [Datasets](#Datasets)
* [Usage](#Usage)
* [Running Example](#Running-Example)



 Summary
 -------
SpatialDiffusion is a python based deep learning model that is designed for spatial transcriptomic in silico generation and interpolation. SpatialDiffusion performs this by learning via a forward noise process and reverse diffusion process via a neural network.

SpatialDiffusion Workflow
-----------------


<img width="1224" alt="Screenshot 2024-03-13 at 1 46 44 PM" src="https://github.com/sumeer1/stDiff/assets/70262340/c74278cb-bbbb-4061-ad95-8d64a001a08c">









Installation Requisites 
-----------------------

The required libraries are included in [environment file](https://github.com/sumeer1/scAEGAN/blob/main/environment.yml). In order to install these libraries, follow the following steps:

* Creating the conda environment with the following command. This will create and install the libraries included in the environment.yml file for training the SpatialDiffusion.
```
conda env create --prefix ./env --file environment.yml --force
 ```

* The second step is to activate the conda envirnoment. 
```
conda activate ./env      
```



	



* SpatialDiffusion is simply installed by cloning the repository.
```
git clone https://github.com/sumeer1/stDiff.git

cd stDiff/
```

Datasets
---------

The preprocessed datasets can be downloaded from [(SpatialDiffusion-data)](https://drive.google.com/drive/folders/1gjwjor6MBrUm4yAiOcCUMkfXYZobojvN?usp=share_link). The folder contains the output data for analysis as well.

Usage
------
*  Training the SpatialDiffusion with the given parameters to get the latent representation by running. 
```bash
python train_interpolate_dlpfc.py
             --data_file <Specifies the input to the stDiff model in spot by gene format> \
             --epochs <Specifies  the number of epochs for which stDiff is trained, default=300> \
              --batch_size <Specifies the batch size to train stDiff, default=4>  \
             --learning_rate <Specifies the learning rate, default = 1e-3> \
             --output_file <Specifies the interpolated layer/slice output from the stDiff > \
```




 Running Example
 ---------------
*   In this tutorial we show how to run SpatialDiffusion  on the DLPFC Data. We have 
prepared the required input dataset which you can find in the stData folder. 
*   We created a command-line interface for SpatialDiffusion that allows it to be run in a high-performance computing environment. Because SpatialDiffusion is built with Pytorch, we recommend running it on GPUs to significantly reduce run-time. It has been tested on Linux and OS X platforms.
*   The experiments were performed on a Linux server using an Intel (R) Xeon (R) CPU Gold 6230R @ 2.10GHz processor with 256 GB RAM and an Quadro RTX 8000 GPU.
 * For model training and evaluation, a [vignette](https://github.com/sumeer1/stDiff/blob/main/notebooks/stDiff_dlpfc_analysis.ipynb) presents an example how to run the SpatialDiffusion and carry out the benchmarking.
 
 


