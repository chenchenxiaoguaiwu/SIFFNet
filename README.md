# SIFFNet 

SIFFNet is a network model based on Python 3.9, designed to remove the complex noise of DAS-VSP.

## Simple installation from pypi.org

Install the Python libraries required for training the SIFFNet and testing the DAS-VSP denoising process

`pip install -`

  `pip install python3.9`

  `pip install pytorch1.31.1`

  `pip install matplotlib`

  `pip install einops`

  `pip install numpy`

The above command should directly install all the dependencies required for the fully functional version of SIFFNet. You don't need to manually download anything.


## A simple example

To illustrate how to denoise DAS-VSP data using SIFFNet, let's start with a simple example.


### 1.Training(train.py)

First of all, the data loading module required for training must be imported.

`from SIFFNet_main.utils.dataset_feature_label_segmentation import MyDataset`

Next, load the SIFFNet denoising network.

`from SIFFNet_main.model.SIFFNet import SIFFNet.`

The loaded data is divided into batch-x noisy data, batch-y clean labels, and batch-z semantic labels, and is provided to SIFFNet for supervised denoising training.

### 2. Testing(denoise_test_real.py)

First, import the DAS-VSP data.

`seismic_noise = np.load('../noisy.npy')`

Load `from Cut_combine import cut`to trim the DAS-VSP seismic data into test blocks of size patch-size by patch-size.

`patch-size=160`

Next, call the trained SIFFNet denoising model. The model path is:

`model = torch.load('../model_epoch30.pth')`

Then, load the denoised test block with `from Cut_combine import combine` to restore it to the entire DAS-VSP seismic data.

`seismic_predict = combine(predict_datas, patch_size, strides_x, strides_y, seismic_block_h, seismic_block_w)`

The noise removal of the entire DAS-VSP seismic data is saved at:

`np.save('../data/denoise.npy', denoise_data)`

`denoise_data = seismic_noise-seismic_predict`

## Download trained SIFFNet and DAS-VSP noisy data

The trained model and Data are available for Google Drive. To ensure proper access and usage, please follow these steps:
Click on the Google Drive link.[Google Drive](https://drive.google.com/drive/folders/1JfO6M9vVnOCb0VUHeephwKo0UPqxsuwy?usp=sharing).

## Demo
import numpy as np
import matplotlib.pyplot as plt
from Cut_combine import cut, combine
import torch
from scipy.io import savemat

### Load test DAS-VSP data
seismic_noise = np.load('..data/noisy.npy')
seismic_block_h, seismic_block_w = seismic_noise.shape


### Crop the DAS-VSP test data
patch_size = 160
patches, strides_x, strides_y, fill_arr_h, fill_arr_w = cut(seismic_noise, patch_size, patch_size, patch_size)

### Check if there is a GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

### Load the trained SSIF denoising model
model = torch.load('..model/model_epoch30.pth')
model.to(device=device)  

### Store test data patches
predict_datas = []  

### Perform denoising processing on the test data patches
for patch in patches:
    patch = np.array(patch)
    print(patch.shape)
    patch = patch.reshape(1, 1, patch.shape[0], patch.shape[1])
    print(patch.shape)
    patch = torch.from_numpy(patch)
    patch = patch.to(device=device, dtype=torch.float32) 
    predict_data = model(patch) 
    predict_data = predict_data[1].detach().cpu().numpy() 
    print(predict_data.shape)
    predict_data = predict_data.squeeze() 
    print(predict_data.shape)
    predict_datas.append(predict_data)

### Restore the denoised test data patches to the entire DAS-VSP
seismic_noise_predict = combine(predict_datas, patch_size, strides_x, strides_y, seismic_block_h, seismic_block_w)

seismic_denoised=seismic_noise-seismic_noise_predict

### DAS-VSP denoised result display

plt.figure(figsize=(12, 5))  

### Left subgraph: Noisy DAS-VSP data
plt.subplot(1, 2, 1)
plt.imshow(seismic_noise, cmap='seismic', aspect='auto', 
           vmin=-np.max(np.abs(seismic_noise))*0.5, 
           vmax=np.max(np.abs(seismic_noise))*0.5)
plt.title('Noisy DAS-VSP Data')
plt.xlabel('Trace Number')
plt.ylabel('Time Sample')
plt.colorbar(label='Amplitude')

### Right subimage: Denoised result
plt.subplot(1, 2, 2)
plt.imshow(seismic_denoised, cmap='seismic', aspect='auto', 
           vmin=-np.max(np.abs(seismic_denoised))*0.5, 
           vmax=np.max(np.abs(seismic_denoised))*0.5)
plt.title('Denoised DAS-VSP Data')
plt.xlabel('Trace Number')
plt.ylabel('Time Sample')
plt.colorbar(label='Amplitude')

plt.tight_layout()  
plt.show()

![Comparison of noisy data and denoised results](https://github.com/yangqingchen2024/SIFFNet/blob/main/SIFFNet/Noisy%20and%20denoised%20images.png)

**Figure 1: Comparison of Denoising Effects of DAS-VSP Data**
*Left side: Original noisy data | Right side: SIFFNet denoising result*

## Dataset

The noise in this experiment comes from the real DAS-VSP seismic records. It is confidential data and cannot be made public.


