from torch.utils.data import Dataset
import cv2
import numpy as np
from typing import List
import os
import sys
import project_paths as pp

class NoiseDataloader(Dataset):

    IMAGE_EXTENSIONS = ['.jpg', '.png', '.jpeg']

    TRAIN = 0
    TEST = 1
    VALIDATION = 2

    GAUSSIAN = 0
    POISSON = 1

    PATCH_STRIDE = 256

    @staticmethod
    def crop(image):
        return image[0:-1, 0:-1]

    @staticmethod
    def convert_image_to_model_input(image):
        return image.transpose(2, 0, 1)

    @staticmethod
    def convert_model_output_image(torch_tensor):
        return torch_tensor.to('cpu').detach().numpy().transpose(1, 2, 0)


    def __init__(self, dataset_type=TEST, noisy_per_image=500, noise_type=GAUSSIAN, mean=0, std=1.0):

        self.dataset_type = dataset_type

        if dataset_type == NoiseDataloader.TRAIN:
            self.images_folder_path = os.path.join(pp.bsd_500_dataset_folder_path, 'train')
        if dataset_type == NoiseDataloader.TEST:
            self.images_folder_path = os.path.join(pp.bsd_500_dataset_folder_path, 'test')
        if dataset_type == NoiseDataloader.VALIDATION:
            self.images_folder_path = os.path.join(pp.bsd_500_dataset_folder_path, 'val')

        self.clean_patches: List[np.ndarray] = []

        if os.path.isdir(self.images_folder_path):
            for image_file_name in os.listdir(self.images_folder_path):
                image_file_path = os.path.join(self.images_folder_path, image_file_name)
                if os.path.isfile(image_file_path) and os.path.splitext(image_file_name)[1].lower() in NoiseDataloader.IMAGE_EXTENSIONS:
                    image = np.asarray(cv2.cvtColor(cv2.imread(image_file_path), cv2.COLOR_BGR2RGB) / 255, dtype=np.float32)
                    for i in range(0, image.shape[0], NoiseDataloader.PATCH_STRIDE):
                        if i + NoiseDataloader.PATCH_STRIDE >= image.shape[0]:
                            i = image.shape[0] - NoiseDataloader.PATCH_STRIDE
                        for j in range(0, image.shape[1], NoiseDataloader.PATCH_STRIDE):
                            if j + NoiseDataloader.PATCH_STRIDE >= image.shape[1]:
                                j = image.shape[1] - NoiseDataloader.PATCH_STRIDE

                            patch = image[i:i + NoiseDataloader.PATCH_STRIDE, j:j + NoiseDataloader.PATCH_STRIDE]
                            patch = NoiseDataloader.convert_image_to_model_input(patch)

                            self.clean_patches.append(patch)
        else:
            print("Dataset Path Doesn't Exist!")
            sys.exit(0)
        self.number_of_clean_images = len(self.clean_patches)
        self.noisy_per_image = noisy_per_image

        self.noise_type = noise_type
        if self.noise_type == NoiseDataloader.GAUSSIAN:
            self.mean = mean
            self.std = std


    def __len__(self):
        return self.noisy_per_image * self.number_of_clean_images


    def __getitem__(self, idx):
        clean_patch = self.clean_patches[int(idx) % self.number_of_clean_images]

        if self.noise_type == NoiseDataloader.GAUSSIAN:
            # Adding Zero Mean Gaussian Noise
            noisy_image_1 = clean_patch + np.random.normal(self.mean, self.std, size=clean_patch.shape)
            if self.dataset_type == NoiseDataloader.TRAIN:
                noisy_image_2 = clean_patch + np.random.normal(self.mean, self.std, size=clean_patch.shape)

        elif self.noise_type == NoiseDataloader.POISSON:
            # Adding Poisson Noise
            noisy_image_1 = np.random.poisson(clean_patch)
            if self.dataset_type == NoiseDataloader.TRAIN:
                noisy_image_2 = np.random.poisson(clean_patch)


        if self.dataset_type == NoiseDataloader.TRAIN:
            return np.asarray(noisy_image_1, dtype=np.float32), np.asarray(noisy_image_2, dtype=np.float32)
        else:
            return np.asarray(noisy_image_1, dtype=np.float32), np.asarray(clean_patch, dtype=np.float32)
