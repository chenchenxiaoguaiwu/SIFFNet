# Data patching and recombination utilities

# '''
#     1. Pad the right and bottom of the original data block (arr1) so that both horizontal and vertical dimensions are divisible by the patch size (L*L).
#     2. After feeding the cut patches into the network for training, only take the center part (L*L) of the data and reassemble in order to obtain data of the same size as arr1.
# '''
import numpy as np


def cut(seismic_block, patch_size, stride_x, stride_y):
    """
    Cut seismic data into patches with specified size and stride

    :param seismic_block: Seismic data block
    :param patch_size: Size of each patch
    :param stride_x: Horizontal stride (equal to patch_size)
    :param stride_y: Vertical stride (equal to patch_size)
    :return: List of patches, number of patches in height direction, number of patches in width direction,
             height after padding, width after padding
    """
    [seismic_h, seismic_w] = seismic_block.shape  # Get height and width of seismic data block

    # Pad the data to ensure complete coverage when cutting
    # Determine width after padding
    n1 = 1
    while (patch_size + (n1 - 1) * stride_x) <= seismic_w:
        # Calculate how many steps of size stride_x are needed to cover seismic_w with patch_size
        n1 = n1 + 1
    # After loop: patch_size + (n1-1)*stride_x > seismic_w, ensuring complete coverage with integer steps
    arr_w = patch_size + (n1 - 1) * stride_x

    # Determine height after padding
    n2 = 1
    while (patch_size + (n2 - 1) * stride_y) <= seismic_h:
        n2 = n2 + 1
    arr_h = patch_size + (n2 - 1) * stride_y

    # Pad the right and bottom of seismic_block with zeros
    fill_arr = np.zeros((arr_h, arr_w), dtype=np.float32)
    fill_arr[0:seismic_h, 0:seismic_w] = seismic_block

    # Calculate sliding positions in width direction
    # Python indexing starts at 0 and uses half-open intervals [start, end)
    # patch_size + (n-1)*stride_x represents the end position (exclusive) of each slice
    # Note: n is calculated as 1 more than the actual number of steps
    path_w = []  # Store x-direction sliding positions
    x = np.arange(n1)  # Generate sequence [0 to n1-1]
    x = x + 1  # Convert to [1 to n1]
    for i in x:
        s_x = patch_size + (i - 1) * stride_x  # Calculate end position for each step
        path_w.append(s_x)  # Add to list
    number_w = len(path_w)

    # Calculate sliding positions in height direction
    path_h = []
    y = np.arange(n2)
    y = y + 1
    for k in y:
        s_y = patch_size + (k - 1) * stride_y
        path_h.append(s_y)
    number_h = len(path_h)

    # Extract patches from the data using slice indices
    cut_patches = []
    for index_x in path_h:  # path_h indices represent patch rows
        for index_y in path_w:  # path_w indices represent patch columns
            patch = fill_arr[index_x - patch_size: index_x, index_y - patch_size: index_y]
            cut_patches.append(patch)

    return cut_patches, number_h, number_w, arr_h, arr_w


def combine(patches, patch_size, number_h, number_w, block_h, block_w):
    """
    Reconstruct the original data block from patches after processing

    :param patches: List of patches from get_patches
    :param patch_size: Size of each patch
    :param number_h: Number of patches in height direction
    :param number_w: Number of patches in width direction
    :param block_h: Height of original seismic data block
    :param block_w: Width of original seismic data block
    :return: Reconstructed seismic data block
    """
    # Convert list of patches to 2D matrix and concatenate in order
    temp = np.zeros((int(patch_size), 1), dtype=np.float32)  # Temporary matrix for concatenation (will be deleted)
    print(temp.size)

    # Concatenate each patch from patches list along axis=1 (columns)
    for i in range(len(patches)):
        temp = np.append(temp, patches[i], axis=1)

    # Delete the initial temporary column
    # Now temp1 has dimensions: patch_size × (patch_size*number_h*number_w)
    temp1 = np.delete(temp, 0, axis=1)

    # Reshape data to (patch_size*number_h) × (patch_size*number_w)
    test = np.zeros((1, int(patch_size * number_w)),
                    dtype=np.float32)  # Temporary matrix for concatenation (will be deleted)

    # Reorganize data by adding line breaks every patch_size*number_w columns
    for j in range(0, int(patch_size * number_h * number_w), int(patch_size * number_w)):
        test = np.append(test, temp1[:, j:j + int(patch_size * number_w)], axis=0)

    # Delete the initial temporary row
    test1 = np.delete(test, 0, axis=0)

    # Extract the original block dimensions
    block_data = test1[0:block_h, 0:block_w]

    return block_data