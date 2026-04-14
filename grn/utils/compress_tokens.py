import torch
import numpy as np

def save_packed_tensor(filename, tensor):
    """use np.savez_compressed to save compressed tensor and its shape"""
    if tensor.dtype != torch.bool:
        raise TypeError("Input tensor must be of dtype torch.bool")
    shape_array = np.array(tensor.shape)
    packed_data = np.packbits(tensor.numpy())
    np.savez_compressed(filename, shape=shape_array, data=packed_data)

def load_packed_tensor(filename):
    """read .npz file and decompress tensor"""
    with np.load(filename) as loader:
        shape = loader['shape']
        packed_data = loader['data']
    numel = np.prod(shape)
    unpacked_data = np.unpackbits(packed_data, count=numel)
    restored_tensor = torch.from_numpy(unpacked_data.reshape(shape))
    return restored_tensor
