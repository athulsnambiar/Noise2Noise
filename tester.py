import torch
from torch import nn
from model import FullCNN
import os
import project_paths as pp
import cv2
from matplotlib import pyplot as plt
import numpy as np
from dataloader import NoiseDataloader

# Reading Random Image
image_file_names = os.listdir(pp.bsd_500_test_dataset_folder_path)
image = np.asarray(
    cv2.cvtColor(
        cv2.imread(
            os.path.join(pp.bsd_500_test_dataset_folder_path, image_file_names[np.random.randint(0, len(image_file_names))])
        ),
        cv2.COLOR_BGR2RGB
    ) / 255, dtype=np.float32)
image = NoiseDataloader.crop(image)

# # Custom Image
# image = np.asarray(cv2.cvtColor(cv2.imread(r'C:\Users\adity\Downloads\image_2.jpg'), cv2.COLOR_BGR2RGB) / 255, dtype=np.float32)
# noisy_image = image.copy()

# Initializing network
network = FullCNN()
network = nn.DataParallel(network)
instance = '000'
pretrained_model_folder_path = os.path.join(pp.trained_models_folder_path, 'Instance_' + instance)
for pretrained_model_file_name in os.listdir(pretrained_model_folder_path):
    try:
        if pretrained_model_file_name.endswith('.pt'):
            network.load_state_dict(torch.load(os.path.join(pretrained_model_folder_path, pretrained_model_file_name)))
            print('Network weights initialized using file from:', pretrained_model_file_name)
        else:
            continue
    except:
        print('Unable to load network with weights from:', pretrained_model_file_name)
        continue

    # noisy_image = image + np.random.normal(loc=0, scale=0.5, size=image.shape)
    noisy_image = image + np.random.uniform(low=-0.1, high=0.1, size=image.shape)
    noisy_image = NoiseDataloader.convert_image_to_model_input(noisy_image)
    predicted_image = network(torch.unsqueeze(torch.as_tensor(noisy_image, dtype=torch.float32), dim=0))

    predicted_image = NoiseDataloader.convert_model_output_image(predicted_image[0])

    plt.figure(num='Network Performance using weights at {}'.format(pretrained_model_file_name), figsize=(20, 20))

    plt.subplot(2, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.colorbar()
    plt.title('Original Image')

    plt.subplot(2, 2, 2)
    plt.imshow(NoiseDataloader.convert_model_output_image(torch.as_tensor(noisy_image)), cmap='gray')
    plt.colorbar()
    plt.title('Noisy Image')

    plt.subplot(2, 2, 3)
    plt.imshow(predicted_image, cmap='gray')
    plt.colorbar()
    plt.title('Predicted Image')

    plt.subplot(2, 2, 4)
    plt.imshow(np.sqrt(np.sum((image - predicted_image)**2, axis=2)), cmap='gray')
    plt.title('Euclidean Distance')
    plt.colorbar()
    plt.show()