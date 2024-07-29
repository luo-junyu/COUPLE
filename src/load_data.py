import torch
import scipy.io as sio
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

class CustomDataSet(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __getitem__(self, index):
        return self.images[index], self.labels[index], index

    def __len__(self):
        count = len(self.images)
        assert len(self.images) == len(self.labels)
        return count

def get_loader_source(batch_size, base_path, domain_name):
    path = base_path + domain_name + '_feature_mat.mat'
    data = sio.loadmat(path)
    data_tensor = torch.from_numpy(data['deepfea'])
    label_tensor = torch.from_numpy(data['label'])

    source_dataset = CustomDataSet(images=data_tensor, labels=label_tensor)
    source_loader = torch.utils.data.DataLoader(source_dataset, batch_size=batch_size, pin_memory=True, shuffle=True, num_workers=4)
    classes = torch.unique(label_tensor)
    n_class = classes.size(0)
    dim_fea = data_tensor.size(1)

    return source_loader, n_class, dim_fea, source_dataset

def get_loader_target(batch_size, base_path, domain_name):
    path = base_path + domain_name + '_feature_mat.mat'
    data = sio.loadmat(path)
    data_tensor = torch.from_numpy(data['deepfea'])
    label_tensor = torch.from_numpy(data['label'])
    train_data, test_data, train_label, test_label = train_test_split(data_tensor, label_tensor, test_size=0.1, random_state=313)
    imgs = {'train': train_data, 'query': test_data}
    labels = {'train': train_label, 'query': test_label}

    dataset = {x: CustomDataSet(images=imgs[x], labels=labels[x]) for x in ['train', 'query']}
    shuffle = {'train': True, 'query': False}
    dataloader = {x: DataLoader(dataset[x], batch_size=batch_size, pin_memory=True, shuffle=shuffle[x], num_workers=4) for x in ['train', 'query']}

    return dataloader, dataset



