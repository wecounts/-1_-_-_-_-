from skimage.feature import local_binary_pattern
from skimage import filters
import matplotlib.pyplot as plt
from torchvision.transforms import Compose, ToTensor
from torchvision.datasets import FashionMNIST
import pandas as pd
import numpy as np
from skimage.feature import hog

if __name__ == '__main__':
    transform_funcs = Compose([
        ToTensor()
    ])
    train_data = FashionMNIST(root='../data/train_data', train=True, download=False, transform=transform_funcs)
    test_data = FashionMNIST(root='../data/test_data', train=False, download=False, transform=transform_funcs)

    whole_data = {'train': train_data, 'test': test_data}

    # settings for LBP
    radius = 3  # LBP算法中范围半径的取值
    n_points = 8 * radius  # 领域像素点数

    for train_or_test in whole_data:
        hog_features = pd.DataFrame()
        for pic_index, pic_data in enumerate(whole_data[train_or_test]):
            image, label = pic_data
            image = image.view(28, 28)
            # 原图像，又称像素值特征
            plt.subplot(131)  # 把画布分成2*2的格子,放在第1格
            plt.imshow(image, cmap='gray')

            # LBP特征
            lbp = local_binary_pattern(np.array(image), n_points, radius)
            plt.subplot(132)  # 把画布分成2*2的格子,放在第1格
            plt.imshow(lbp, cmap='gray')

            # 使用scikit-image库提取HOG特征
            hog_feature, hog_image = hog(
                image,
                orientations=9,
                pixels_per_cell=(8, 8),
                cells_per_block=(2, 2),
                block_norm='L2-Hys',
                visualize=True,
                transform_sqrt=False,
                feature_vector=True
            )
            plt.subplot(133)  # 把画布分成2*2的格子,放在第2格
            plt.imshow(hog_image, cmap='gray')

            plt.show()
