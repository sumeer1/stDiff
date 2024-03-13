import numpy as np
import torch
from torch.utils.data import Dataset

class SpatialTranscriptomicDataset(Dataset):
    def __init__(self, dataframe, exclude_bregma=None, transform=None):
        self.features = dataframe.iloc[:, 1:-3].values
        self.spatial_coords = np.array(dataframe.iloc[:, -3:-1].values, dtype=np.float32)
        self.slice_index = dataframe['Bregma'].values

        if exclude_bregma is not None:
            indices_to_keep = [i for i, bregma in enumerate(self.slice_index) if bregma != exclude_bregma]
            self.features = self.features[indices_to_keep]
            self.spatial_coords = self.spatial_coords[indices_to_keep]
            self.slice_index = self.slice_index[indices_to_keep]

    def __getitem__(self, index):
        x = torch.from_numpy(self.features[index]).float()
        spatial_coord_data = self.spatial_coords[index]
        spatial_coord = torch.from_numpy(spatial_coord_data).float()
        slice_index = torch.tensor(self.slice_index[index], dtype=torch.float32)
        return x, spatial_coord, slice_index

    def __len__(self):
        return len(self.features)

    def select_slices(self, bregma_value, tolerance=1e-3):
        indices = [i for i, bregma in enumerate(self.slice_index) if np.isclose(bregma, bregma_value, atol=tolerance)]
        slice_data = [self.__getitem__(i) for i in indices]
        return slice_data

    def get_coords_for_bregma(self, bregma_value, tolerance=1e-3):
        indices = [i for i, bregma in enumerate(self.slice_index) if np.isclose(bregma, bregma_value, atol=tolerance)]
        coords = [self.spatial_coords[i] for i in indices]
        return torch.tensor(coords).float()
