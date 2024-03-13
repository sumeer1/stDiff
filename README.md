# stDiff
This repository contains the stDiff code and online data for spatial transcriptomics. It also contains the code to evaluate and visualize the integration results. Metrics are available for quantifying outputs quality.


stDiff- Predicting Spatial Transcriptomics Slices with Denoising Diffusion Probabilistic Models 
---------------------------------------------------------------------------------------------------------
This repository contains the  scAEGAN code and online data for the single-omics and multi-omics integration. It also contains the code to evaluate and visualize the integration results. Metrics are available for quantifying outputs quality.

* [Summary](#Summary)
* [Installation Requisites](#Installation-Requisites )
* [Datasets](#Datasets)
* [Usage](#Usage)
* [Running Example](#Running-Example)



 Summary
 -------
stDiff is a python based deep learning model that is designed for spatial transcriptomic in silico generation and interpolation. stDiff performs this by learning via a forward noise process and neural network learning the reverse diffusion process.

stDiff Workflow


<img width="1230" alt="Screenshot 2024-03-12 at 5 09 31 PM" src="https://github.com/sumeer1/stDiff/assets/70262340/760c1514-f355-4230-a009-6574d02e78ed">








Installation Requisites 
-----------------------

The required libraries are included in [environment file](https://github.com/sumeer1/scAEGAN/blob/main/environment.yml). In order to install these libraries, follow the following steps:

* Creating the conda environment with the following command. This will create and install the libraries included in the environment.yml file for training the scAEGAN.
```
conda env create --prefix ./env --file environment.yml --force
 ```

* The second step is to activate the conda envirnoment. 
```
conda activate ./env      
```



	



* stDiff is simply installed by cloning the repository.
```
git clone https://github.com/sumeer1/stDiff.git

cd stDiff/
```

Datasets
---------


Real data: The pre-processed dorsolateral prefrontal cortex can be found in the real data folder.

Usage
------
*  Training the stDiff with the given parameters to get the latent representation by running. 
```bash
python interpolate_dlfpc.py --input_file <Specifies the input to the stDiff model in spot by gene format> \
             
             --output_file1 <Specifies the interpolated layer/slice output from the stDiff > \
             
             --batch_size <Specifies the batch size to train stDiff, default=8>  \
             --epochs <Specifies  the number of epochs for which autoencoder is trained, default=300> \
             
             --learning_rate <Specifies the learning rate, default=1e-3>
```




 Running Example
 ---------------
*   In this tutorial we show how to run stDiff  on the DLPFC Data. We have 
prepared the required input dataset which you can find in the Real_Data folder. 
*   We created a command-line interface for scAEGAN that allows it to be run in a high-performance computing environment. Because stDiff is built with Pytorch, we recommend running it on GPUs to significantly reduce run-time. It has been tested on Linux and OS X platforms.
*   The experiments were performed on a Linux server using an Intel Xeon CPU E5-2680 v4 @ 2.40GHz processor with 128 GB RAM and an NVIDIA Tesla V100 GPU.
 * For model training and evaluation, a [vignette](https://github.com/sumeer1/scAEGAN/blob/main/Example/scAEGAN_Analysis.ipynb) presents an example how to run the scAEGAN and carry out the benchmarking using the Evaluation folder scripts. 
 
 


