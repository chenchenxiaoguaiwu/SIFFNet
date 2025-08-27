import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


class MyDataset(Dataset):
    # Constructor
    def __init__(self, feature_path, label_path, segmentation_label_path):
        super(MyDataset, self).__init__()
        # Get list of all .npy files in feature directory
        self.feature_paths = glob.glob(os.path.join(feature_path, '*.npy'))
        # Get list of all .npy files in label directory
        self.label_paths = glob.glob(os.path.join(label_path, '*.npy'))
        # Get list of all .npy files in segmentation label directory
        self.segmentation_label_paths = glob.glob(os.path.join(segmentation_label_path, '*.npy'))

    # Return the size of the dataset
    def __len__(self):
        return len(self.feature_paths)

    # Return data and label for the given index
    def __getitem__(self, index):
        # Load feature data from .npy file
        feature_data = np.load(self.feature_paths[index])
        # Load label data from .npy file
        label_data = np.load(self.label_paths[index])
        # Load segmentation label data from .npy file
        segmentation_label_path = np.load(self.segmentation_label_paths[index])

        # Convert numpy arrays to PyTorch tensors
        feature_data = torch.from_numpy(feature_data)
        # feature_data = feature_data.permute(2, 0, 1)  # Commented out dimension permutation
        label_data = torch.from_numpy(label_data)
        segmentation_label_path = torch.from_numpy(segmentation_label_path)
        # label_data = label_data.permute(2, 0, 1)  # Commented out dimension permutation

        # Add a channel dimension: 128*128 => 1*128*128
        feature_data.unsqueeze_(0)
        label_data.unsqueeze_(0)
        segmentation_label_path.unsqueeze_(0)

        return feature_data, label_data, segmentation_label_path


if __name__ == "__main__":
    # Define paths to data directories
    feature_path = "..\\data\\feature\\"
    label_path = "..\\data\\label\\"
    segmentation_label_path = "..\\data\\label\\"

    # Create dataset instance
    seismic_dataset = MyDataset(feature_path, label_path, segmentation_label_path)

    # Create data loader for training
    train_loader = torch.utils.data.DataLoader(dataset=seismic_dataset,
                                               batch_size=32,
                                               shuffle=True)

    # Print dataset and loader information
    print('Dataset size:', len(seismic_dataset))
    print('train_loader:', len(train_loader))
