from dataloader import NoiseDataloader
import matplotlib.pyplot as plot

train_dataloader = NoiseDataloader(dataset_type=NoiseDataloader.TRAIN,
                                   noisy_per_image=1,
                                   noise_type=NoiseDataloader.SALT_PEPPER)


for i in range(5):
    x, y = train_dataloader[i]
    plot.imshow(x)
    plot.show()
    plot.imshow(y)
    plot.show()
