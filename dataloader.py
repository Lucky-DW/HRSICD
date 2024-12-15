import os
from PIL import Image
import numpy as np
import torch
import torch.utils.data as data
from torchvision import transforms
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

x_transformer = Compose([
    ToTensor(),
])

y_transformer = Compose([
    ToTensor(),
])

class Dataset_self(data.Dataset):
    def __init__(self, data_dir):
        super(Dataset_self, self).__init__()

        self.root_path = data_dir
        self.li = ['t1','t2','gt']
        self.filenames = [x for x in sorted(os.listdir(os.path.join(data_dir,self.li[0])))]
        self.img_transform = x_transformer
        self.label_transform = y_transformer


    def __getitem__(self, index):
        gt_filepath = os.path.join(self.root_path, self.li[2], self.filenames[index])
        t1_filepath = gt_filepath.replace('gt', 't1')
        t2_filepath = gt_filepath.replace('gt', 't2')

        img1 = np.array(self.img_transform(Image.open(t1_filepath).convert('RGB')))
        img2 = np.array(self.img_transform(Image.open(t2_filepath).convert('RGB')))
        label = np.array(self.label_transform(Image.open(gt_filepath).convert('L')))


        return img1, img2, label

    def __len__(self):
        return len(self.filenames)


if __name__ == "__main__":
    data_A_dir = '/home/september/code/layman/DATA/LEVIR-CD/train'
    img_data = Dataset_self(data_A_dir)
    data_loader_train = torch.utils.data.DataLoader(dataset=img_data,
                                                    batch_size=16,
                                                    shuffle=True)
    for x in data_loader_train:
        image1,image2,gt = x
        print(image1.shape)
        print(image2.shape)
        print(gt.shape)