import cv2
import json
import multiprocessing as mp
import numpy as np
import os

import torch
import torchvision.transforms as transforms

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

means = [ 0.5, 0.5, 0.5 ]
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
class Normalize:
    def __init__():
        pass

    def __call__(self, data_pair):
        data_pair[0] /= 255

        data_pair[0][0] -= means[0]
        data_pair[0][1] -= means[1]
        data_pair[0][2] -= means[2]

        data_pair[0][0] /= stds[0]
        data_pair[0][1] /= stds[1]
        data_pair[0][2] /= stds[2]

        return data_pair

class RandomBrightness:
    def __init__(self, range_vals):
        self.range_vals = range_vals

    def __call__(self, data_pair):
        random_number = np.random.random() * self.range_vals[1] / self.range_vales[0]  # [0, max]
        data_pair[0] *= np.clip(random_number, a_min=0, a_max=255)

class ToTensor:
    def __init__(self):
        pass

    def __call__(self, data_pair):
        data_pair[0] = data_pair[0].transpose(2, 0, 1)
        data_pair[0] = torch.from_numpy(data_pair[0]).float()
        data_pair[1] = data_pair[1]
        return data_pair

class DermatologistDataset(Dataset):
    def __init__(self, root, transforms):
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

if __name__ == '__main__':
    transform = transforms.Compose([
        ToTensor()
    ])

    dataset = DermatologistDataset(root='data/train/', transforms=transform)
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=True, num_workers=int(mp.cpu_count() * 0.8))

    print('DataLoader type: ', type(dataloader))
    data = next(iter(dataloader))

    img_tensor = data[0]
    lbl = data[1]

    #Move back to numpy (HWC) from (BCHW)
    rgb = img_tensor.squeeze(0).numpy().transpose(1, 2, 0).astype(np.uint8)

    cv2.imshow(classes[lbl], cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), (648, 480)))

    while cv2.waitKey(1) & 0xff != ord('q'):
        continue
    cv2.destroyAllWindows()
