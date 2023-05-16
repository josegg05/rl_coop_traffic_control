from os import path
import pickle 
import numpy as np
from scipy.sparse import coo_matrix
import dgl
import torch
from torch.utils.data import Dataset, DataLoader


class StandardScaler():
    """
    Standard the input
    """

    def __init__(self, mean, std, norm_type):
        self.mean = mean
        self.std = std
        self.norm_type = norm_type

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform_old(self, data):
        return (data * self.std) + self.mean

    def inverse_transform(self, data):
        if self.norm_type == 'std':
            result = (data * self.std_torch) + self.mean_torch
        elif self.norm_type == 'node':
            mean = self.mean_torch.repeat(data.shape[1], int(data.shape[0]/(self.mean_torch.shape[0]))).transpose(0, 1)
            std = self.std_torch.repeat(data.shape[1], int(data.shape[0]/(self.mean_torch.shape[0]))).transpose(0, 1)
            result = (data * std) + mean
        return result

    def to(self, device):
        if self.norm_type == 'std':
            self.mean_torch = torch.Tensor([self.mean])
            self.std_torch = torch.Tensor([self.std])
        elif self.norm_type == 'node':
            self.mean_torch = torch.Tensor(self.mean)
            self.std_torch = torch.Tensor(self.std)
        self.mean_torch = self.mean_torch.to(device)
        self.std_torch = self.std_torch.to(device)


def load_graph_data(pkl_filename):
    sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(pkl_filename)
    return sensor_ids, sensor_id_to_ind, adj_mx


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


def load_graph(pkl_filename):
    _, _, adj_matrix = load_graph_data(pkl_filename)
    adj_matrix_sparse = coo_matrix(adj_matrix)
    graph = dgl.from_scipy(adj_matrix_sparse, eweight_name='edge_weights')
    return graph

def load_node_data(path_to_data, normalize, norm_type, scaler) -> np.ndarray:
    """ load test, train or val *.npz file containing the sensor data and 
    convert it to a numpy array to make it efficient"""
    data = np.load(path_to_data)
    x, y = np.array(data['x'], np.float32), np.array(data['y'], np.float32)
    if normalize:
        if not scaler:
            if norm_type == 'std':
                scaler = StandardScaler(mean=x[..., 0].mean(), std=x[..., 0].std(), norm_type=norm_type)
            elif norm_type == 'node':
                scaler = StandardScaler(mean=x[..., 0].mean(axis=(0, 1)), std=x[..., 0].std(axis=(0, 1)), norm_type=norm_type)
        x[..., 0] = scaler.transform(x[..., 0])
    return x, y, scaler
    

class GraphDataset(Dataset):
    """ GraphDataset used to load train test and val datasets independently """
    def __init__(self, path_to_data, path_to_graph, normalize, norm_type, scaler) -> None:
        self.x, self.y, self.scaler = load_node_data(path_to_data, normalize, norm_type, scaler)
        self.g = load_graph(path_to_graph)

    def __len__(self): 
        return self.x.shape[0]

    def __getitem__(self, index) -> np.ndarray:
        return self.x[index, ...], self.y[index, ..., 0], self.g

    def getscaler(self):
        return self.scaler

def collate(samples): 
    data_x, data_y, graphs = map(list, zip(*samples))
    x = torch.stack([torch.from_numpy(x) for x in data_x], dim = 0)
    y = torch.stack([torch.from_numpy(y) for y in data_y], dim = 0)
    g = dgl.batch(graphs)
    return x, y, g   

def loader(data_path, graph_path, batch_size=32, num_workers=0, normalize=False, norm_type='std'):

    dataset_train = GraphDataset(path.join(data_path, 'train.npz'), graph_path, normalize, norm_type, None)
    scaler = dataset_train.getscaler()
    dataset_test  = GraphDataset(path.join(data_path, 'test.npz'), graph_path, normalize, norm_type, scaler)
    dataset_val   = GraphDataset(path.join(data_path, 'val.npz'), graph_path, normalize, norm_type, scaler)

    train_loader = DataLoader(
        dataset_train, 
        batch_size=batch_size,
        shuffle=True, 
        prefetch_factor=2,
        num_workers=num_workers,
        collate_fn=collate
    )
    test_loader = DataLoader(
        dataset_test, 
        batch_size=batch_size, 
        prefetch_factor=2, 
        num_workers=num_workers,
        collate_fn=collate
    )
    val_loader = DataLoader(
        dataset_val, 
        batch_size=batch_size, 
        prefetch_factor=2, 
        num_workers=num_workers,
        collate_fn=collate
    )
    loaders = {
        'train': train_loader, 
        'val': val_loader, 
        'test': test_loader, 
        'scaler': scaler,
        'y_test':dataset_test.y
    }

    return loaders
