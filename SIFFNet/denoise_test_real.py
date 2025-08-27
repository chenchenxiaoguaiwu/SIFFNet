import numpy as np
import matplotlib.pyplot as plt
from Cut_combine import cut, combine  # Custom module for cutting and combining image patches
import torch
from scipy.io import savemat
import scipy.io

# Load data from MATLAB file
mat_data = scipy.io.loadmat("../real_noisy.mat")
# Extract the data array from the loaded MATLAB structure
data = mat_data['data']
# Assign to seismic_noise variable and crop to specific dimensions
seismic_noise = data
seismic_noise = seismic_noise[0:6001, 0:1601]
# Get dimensions of the seismic data
seismic_block_h, seismic_block_w = seismic_noise.shape

# Data normalization
seismic_noise_max = abs(seismic_noise).max()  # Get maximum amplitude of data
seismic_noise = seismic_noise / seismic_noise_max  # Normalize data to range (-1, 1)

# Expand and pad missing shot gather data, then split into patches for processing
patch_size = 160
patches, strides_x, strides_y, fill_arr_h, fill_arr_w = cut(seismic_noise, patch_size, patch_size, patch_size)

# Check if GPU is available for accelerated processing
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load pre-trained denoising model
model = torch.load('../model_epoch30.pth')
model.to(device=device)  # Transfer model to appropriate device (GPU/CPU)
model.eval()  # Set model to evaluation mode
predict_datas = []  # Empty list to store network-predicted patch data

# Process each patch through the neural network
for patch in patches:
    patch = np.array(patch)  # Ensure data is in numpy array format
    print(patch.shape)
    patch = patch.reshape(1, 1, patch.shape[0], patch.shape[1])
    print(patch.shape)
    patch = torch.from_numpy(patch)
    patch = patch.to(device=device, dtype=torch.float32)
    with torch.no_grad():
        predict_data = model(patch)
    predict_data = predict_data[1].detach().cpu().numpy()
    print(predict_data.shape)
    predict_data = predict_data.squeeze()
    print(predict_data.shape)
    predict_datas.append(predict_data)

# Reconstruct full image from processed patches
seismic_predict = combine(predict_datas, patch_size, strides_x, strides_y, seismic_block_h, seismic_block_w)

# Save intermediate results for analysis and debugging
np.save('..\\chatu.npy', seismic_predict)
loaded_data3 = np.load('..\\chatu.npy')
savemat('..\\chatu.mat', data_dict3)  # Save as MATLAB format for compatibility with other tools

# Calculate denoised result by subtracting predicted noise from original
seismic_noise = torch.from_numpy(seismic_noise)
denoise_data = seismic_noise - seismic_predict

# Save various outputs for analysis
np.save('..\\denoise.npy', denoise_data)  # Final denoised result
np.save('..\\predict.npy', predict_data)  # noise
np.save('..\\seismic_noise.npy', seismic_noise)  # Original input data

# Load saved files to verify they were saved correctly
loaded_data1 = np.load('..\\denoise.npy')
loaded_data2 = np.load('..\\predict.npy')
loaded_data5 = np.load('..\\seismic_noise.npy')

# Prepare data dictionaries for MATLAB export
data_dict1 = {'im_denoise': loaded_data1}  # Denoised image
data_dict2 = {'im_predict': loaded_data2}  # noise
data_dict5 = {'im_noisy': loaded_data5}    # Original noisy input

# Save all data as MATLAB .mat files for further analysis in MATLAB
savemat('..\\denoise.mat', data_dict1)
savemat('..\\predict.mat', data_dict2)
savemat('..\\seismic_noise.mat', data_dict5)

print("Processing complete. Results saved in multiple formats.")