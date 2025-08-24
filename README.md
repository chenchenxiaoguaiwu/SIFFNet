## A Semantic Information Feature Fusion Network for Suppressing Diverse DAS-VSP Noise


## Description
In SIFFNet, a lightweight self-similar attention module is designed to efficiently extract non-local self-similarity through grid attention.  These  denoising network that leverages both hybrid attention mechanisms and multi-scale processing to recofeatures are fused with learned semantic information of DAS-VSP signals to guide the subsequent dual-branchnstruct local details and global morphology simultaneously. 



## Features
• We propose a semantic information feature fusion network (SIFFNet)  by incorporating semantic information and non-local self-similarity of DAS-VSP data.

• We exploit the non-local self-similarity prior by improved lightweight attention module to effectively and efficiently recover weak seismic signal as well as global structure preservation.

• Guided by semantic and self-similarity, SIFFNet thoroughly suppresses multi-type DAS-VSP noise and recover


## Requirements

- python
- matplotlib
- pytorch
- einops
- numpy

## Project Structure

text


SIFFNet.py              # Main network architecture  

CAMixing.py             # Channel Attention Mixing module

SIE.py                  # Segmentation network (UUNet)

segmentation_network.py # Alternative segmentation network

train.py                # Training script

denoise_test.py         # Denoising testing script

denoise_test_real.py    # Real data denoising script

losses.py               # Custom loss functions

Cut_combine.py          # Data patching and recombination utilities

requirements.txt        # Python dependencies

README.md               # This file

## Instruction

**Training**

Prepare your dataset in the required structure:

Feature data: ../SIFFNet/data/feature_VSP/

Label data: ../SIFFNet/data/label_VSP/

Mask data: ../SIFFNet/data/mask_VSP/

Run the training script:


```bash
python train.py
```


**Training Output:**

-   Model checkpoints saved at each epoch (e.g., `model_epoch1.pth`, `model_epoch2.pth`, etc.)
    
-   Training and validation loss values printed to console
    
-   Loss plot saved as `loss_plot_VSP.png`
    
-   Loss values saved to `loss_sets_VSP.txt`


### Testing


**Synthetic data testing**
Synthetic data testing requires two input files:

-   **Clean signal**: `clean_normal.npy` 
    
-   **Real noise data**: `noise.npy`
    

For synthetic data testing:
```bash

python denoise_test.py
```

**Synthetic Test Output:**

-   Denoised data saved as `denoise.npy` and `denoise.mat`
    
-   Predicted noise component saved as `predict.npy` and `predict.mat`
    
-   Difference map (denoised vs clean) saved as `chatu.npy` and `chatu.mat`
    
-   Original clean data saved as `seismic.npy` and `seismic.mat`
    
-   Noisy input data saved as `seismic_noise.npy` and `seismic_noise.mat`



**real data testing**
For real data testing:

```bash
python denoise_test_real.py
```
**Real Test Input:**

-   MATLAB format seismic data file (.mat)
    
-   Data should be contained in the 'data' variable of the mat file
    
    

**Real Test Output:**

-   Segmentation results saved as `chatu.npy` and `chatu.mat`
    
-   Processed seismic data with noise removed
    
-   Binary segmentation mask 
