import json
import numpy as np
import os

import torch

from torch.utils.data import Dataset, DataLoader

'''
old_range = old_max - old_min
new_range = new_max - new_min
new_val = (((old_val - old_min) * new_range) / old_range) + new_min
'''

classes = [
    'melanoma', 
    'nevus', 
    'seborrheic_keratosis'
]

means = [ 123.68, 116.779, 103.939 ]
stds = [ 0.5, 0.5, 0.5 ]

def find_stats_of_images(root):
    json_data = None
    with open('rgb_means.json', 'r') as fh:
        json_data = json.load(fh)

        means = []
        stds = []
        for root, dirs, files in os.walk(root):
            for dirname in dirs:
                images = []
                for filename in os.listdir(dirname):
                    images.append(cv2.cvtColor(cv2.imread(), cv2.COLOR_BGR2RGB))
                images_np = np.array(images)
                means.append(images_np.mean())
                stds.append(images_np.std())

        json_data['means'] = json_data['means'] + np.array(means).mean() / 2.0
        json_data['stds'] = json_data['stds'] + np.array(stds).std() / 2.0

    if json_data is not None:
        with open('rgb_means.json', 'w+') as fh:
            json.dump(json_data, fh)

# Augmentations
def Normalize(data_pair):
    data_pair[0] /= 255

    data_pair[0][0] -= means[0]
    data_pair[0][1] -= means[1]
    data_pair[0][2] -= means[2]

    data_pair[0][0] /= stds[0]
    data_pair[0][1] /= stds[1]
    data_pair[0][2] /= stds[2]

    return data_pair

def RandomBrightness(data_pair, range_vals):
    random_number = np.random.random() * range_vals[1] / range_vales[0]  # [0, max]
    data_pair[0] *= np.clip(random_number, a_min=0, a_max=255)

def ToTensor(data_pair):
    data_pair[0] = data_pair[0].transpose(2, 0, 1)
    data_pair[0] = torch.from_numpy(data_pair[0]).float()
    data_pair[1] = torch.from_numpy(data_pair[1]).float()
    return data_pair

class DermatologistDataset(Dataset):
    def __init__(self, root, transform):
        self.transform = transform
        self.image_paths = []

        for root, dirs, files in os.walk(root):
            for dir_name in dirs:
                for file_name in os.listdir(os.path.join(root, dir_name)):
                    self.image_paths.append([dir_name, os.path.join(root, dir_name, file_name)])

    def __getitem__(self, idx):
        lbl = classes.index(self.image_paths[idx][0])
        rgb = cv2.cvtColor(
            cv2.imread(self.image_paths[idx][1]), 
            cv2.COLOR_BGR2RGB).astype(np.uint8)

        data_pair = [rgb, lbl]

        if self.transform:
            data_pair = self.transform(data_pair)

        if data_pair is None or not torch.is_tensor(data_pair[0]):
            raise('Invalid data')

        return data_pair

    def __len__(self):
        return len(self.image_paths)

