import os.path
import numpy as np
import torch.utils.data as data
import scipy.io
from utils import data_augmentation
import torch

def is_mat_file(filename):
    return any(filename.endswith(extension) for extension in [".mat"])


class loadingData(data.Dataset):
    """
    Read Hyper-spectral images and RGB images pairs,
    The pair is ensured by 'sorted' function, so please check name convention.
    """
    def __init__(self, image_dir, augment=None, total_num=int):
        super(loadingData, self).__init__()
        self.image_folders = os.listdir(image_dir)
        self.image_files = []
        for i in self.image_folders:
            if is_mat_file(i) and len(self.image_files) <= total_num:
                full_path = os.path.join(image_dir, i)
                self.image_files.append(full_path)
        self.augment = augment
        if self.augment:
            self.factor = 8
        else:
            self.factor = 1

    def __getitem__(self, index):
        file_index = index
        aug_num = 0
        if self.augment:
            file_index = index // self.factor
            aug_num = int(index % self.factor)
        load_dir = self.image_files[file_index]
        data = scipy.io.loadmat(load_dir)
        ms = np.array(data['ms'], dtype=np.float32)
        lms = np.array(data['ms_bicubic'], dtype=np.float32)
        gt = np.array(data['gt'], dtype=np.float32)

        ms, lms, gt = data_augmentation(ms, mode=aug_num), data_augmentation(lms, mode=aug_num), data_augmentation(gt, mode=aug_num)
        ms = torch.from_numpy(ms.copy()).permute(2, 0, 1)
        lms = torch.from_numpy(lms.copy()).permute(2, 0, 1)
        gt = torch.from_numpy(gt.copy()).permute(2, 0, 1)

        # print("sp_ms",ms.min())
        # print("sp_lms",lms.min())
        ms = torch.where(torch.isnan(ms), torch.full_like(ms, 0), ms)

        lms = torch.where(torch.isnan(lms), torch.full_like(lms, 0), lms)


        return ms, lms, gt

    def __len__(self):
        return len(self.image_files)*self.factor
    # def __len__(self):
    #     return len(self.image_files)

class loadingRGBData(data.Dataset):
    """
    Read Hyper-spectral images and RGB images pairs,
    The pair is ensured by 'sorted' function, so please check name convention.
    """
    """
    Read Hyper-spectral images and RGB images pairs,
    The pair is ensured by 'sorted' function, so please check name convention.
    """
    def __init__(self, image_dir, augment=None, total_num=int):
        super(loadingRGBData, self).__init__()
        self.image_folders = os.listdir(image_dir)
        self.image_files = []
        for i in self.image_folders:
            if is_mat_file(i) and len(self.image_files) <= total_num:
                full_path = os.path.join(image_dir, i)
                self.image_files.append(full_path)
        self.augment = augment
        if self.augment:
            self.factor = 8
        else:
            self.factor = 1

    def __getitem__(self, index):
        file_index = index
        aug_num = 0
        if self.augment:
            file_index = index // self.factor
            aug_num = int(index % self.factor)
        load_dir = self.image_files[file_index]
        data = scipy.io.loadmat(load_dir)
        ms = np.array(data['ms'], dtype=np.float32)
        lms = np.array(data['ms_bicubic'], dtype=np.float32)
        gt = np.array(data['gt'], dtype=np.float32)

        ms, lms, gt = data_augmentation(ms, mode=aug_num), data_augmentation(lms, mode=aug_num), data_augmentation(gt, mode=aug_num)
        ms = torch.from_numpy(ms.copy()).permute(2, 0, 1)
        lms = torch.from_numpy(lms.copy()).permute(2, 0, 1)
        gt = torch.from_numpy(gt.copy()).permute(2, 0, 1)

        ms = torch.where(torch.isnan(ms), torch.full_like(ms, 0.99), ms)

        lms = torch.where(torch.isnan(lms), torch.full_like(lms, 0.99), lms)


        # print("ms",ms.min())
        # print("mss",lms.min())


        return ms, lms, gt

    def __len__(self):
        return len(self.image_files)*self.factor
    # def __len__(self):
    #     return len(self.image_files)




# def data_normal(ori_data):
#     d_min=ori_data.min()
#     d_max=ori_data.max()
#     if d_max > 1:
#         ori_data = 0.9999

#     # if d_min< 0:
#     #     ori_data += torch.abs(d_min)
#     #     d_min = ori_data.min()
#     # d_max= ori_data.max()
#     # dst =  d_max - d_min
#     # din=ori_data-d_min
#     # norm_data =torch.div(din,(dst+0.001))
#     # # norm_data = max(norm_data,0.001)
#     return ori_data

