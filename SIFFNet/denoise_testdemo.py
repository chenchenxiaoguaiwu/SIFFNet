import numpy as np
import matplotlib.pyplot as plt
from Cut_combine import cut, combine
import torch
from scipy.io import savemat
import os

# Load test DAS-VSP data
seismic_noise = np.load('../noisy.npy')
seismic_block_h, seismic_block_w = seismic_noise.shape

# Crop the DAS-VSP test data
patch_size = 160
patches, strides_x, strides_y, fill_arr_h, fill_arr_w = cut(seismic_noise, patch_size, patch_size, patch_size)

# Check if there is a GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the trained SSIF denoising model
model = torch.load('../model_epoch30.pth')
model.to(device=device)

# Store test data patches
predict_datas = []

# Perform denoising processing on the test data patches
for patch in patches:
    patch = np.array(patch)
    patch = patch.reshape(1, 1, patch.shape[0], patch.shape[1])
    patch = torch.from_numpy(patch)
    patch = patch.to(device=device, dtype=torch.float32)
    predict_data = model(patch)
    predict_data = predict_data[1].detach().cpu().numpy()
    predict_data = predict_data.squeeze()
    predict_datas.append(predict_data)

# Restore the denoised test data patches to the entire DAS-VSP
seismic_predict = combine(predict_datas, patch_size, strides_x, strides_y, seismic_block_h, seismic_block_w)

seismic_denoise = seismic_noise - seismic_predict

# Create directory to save results (if it doesn't exist)
output_dir = 'E:/jiaojie/SIFFNet/results'
os.makedirs(output_dir, exist_ok=True)

# Save comparison plot
plt.figure(figsize=(12, 5))

# Left subplot: Noisy DAS-VSP data
plt.subplot(1, 2, 1)
plt.imshow(seismic_noise, cmap='seismic', aspect='auto',
           vmin=-np.max(np.abs(seismic_noise))*0.1,
           vmax=np.max(np.abs(seismic_noise))*0.1)
plt.title('Noisy DAS-VSP Data')
plt.xlabel('Trace Number')
plt.ylabel('Time Sample')
plt.colorbar(label='Amplitude')

# Right subplot: Denoised result
plt.subplot(1, 2, 2)
plt.imshow(seismic_denoise, cmap='seismic', aspect='auto',
           vmin=-np.max(np.abs(seismic_denoise))*0.1,
           vmax=np.max(np.abs(seismic_denoise))*0.1)
plt.title('Denoised DAS-VSP Data')
plt.xlabel('Trace Number')
plt.ylabel('Time Sample')
plt.colorbar(label='Amplitude')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'denoising_comparison.png'), dpi=300, bbox_inches='tight')
plt.show()

# Save denoised result separately
plt.figure(figsize=(10, 8))
plt.imshow(seismic_denoise, cmap='seismic', aspect='auto',
           vmin=-np.max(np.abs(seismic_denoise))*0.1,
           vmax=np.max(np.abs(seismic_denoise))*0.1)
plt.title('Denoised DAS-VSP Data')
plt.xlabel('Trace Number')
plt.ylabel('Time Sample')
plt.colorbar(label='Amplitude')
plt.savefig(os.path.join(output_dir, 'denoised_result.png'), dpi=300, bbox_inches='tight')
plt.close()  # Close the figure to prevent memory leaks

# Optional: Save denoised data as npy file
np.save(os.path.join(output_dir, 'denoised_data.npy'), seismic_denoise)

# Optional: Save denoised data as mat file
savemat(os.path.join(output_dir, 'denoised_data.mat'), {'denoised_data': seismic_denoise})

print(f"Denoising results saved to: {output_dir}")