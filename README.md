# Likelihood-Based Diverse Sampling for Trajectory Forecasting

This is the official repository for ICCV 2021 paper [Likelihood-Based Diverse Sampling for Trajectory Forecasting](https://arxiv.org/abs/2011.15084). 

LDS is a simple training objective to diversify the predictions of a pre-trained Normalizing Flow or VAE-based forecasting models. In this repository,
we provide a lightweight example implementing LDS on a toy forecasting task (similar to Figure 1 in the paper). 

## Usage 
Run ```LDS.ipynb```. This notebook is self-contained and should walk you through an example of how to use LDS. Models are declared in ```/models```, and the pre-generated dataset are in ```/data```.  

## Citations
If you find this repository useful for your research, please cite:
```
@article{ma2020diverse,
      title={Likelihood-Based Diverse Sampling for Trajectory Forecasting}, 
      author={Yecheng Jason Ma and Jeevana Priya Inala and Dinesh Jayaraman and Osbert Bastani},
      year={2020},
      url={https://arxiv.org/abs/2011.15084}
}
```

## Contact
If you have any questions regarding the code, feel free to contact me at jasonyma@seas.upenn.edu or open an issue on this repository.