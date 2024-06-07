from skimage.feature import local_binary_pattern
from torchvision.transforms import Compose, ToTensor
from torchvision.datasets import FashionMNIST
import pandas as pd
import numpy as np
import os

def get_LBP_histogram(LBP_image):
    # print(LBP_image)
    histogram_features, _ = np.histogram(LBP_image, bins=20)
    return histogram_features

# 定义LBP特征提取函数
def extract_lbp_features(LBP_image):
    # max_bins = int(LBP_image.max() + 1)
    # 将LBP图像转换为直方图
    # hist, _ = np.histogram(LBP_image, bins=max_bins, range=(0, max_bins))
    hist, _ = np.histogram(LBP_image.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    return hist

if __name__ == '__main__':
    transform_funcs = Compose([
        ToTensor()
    ])
    train_data = FashionMNIST(root='../data', train=True, download=True, transform=transform_funcs)
    test_data = FashionMNIST(root='../data', train=False, download=True, transform=transform_funcs)

    whole_data = {'train': train_data, 'test': test_data}

    # settings for LBP
    radius = 3  # LBP算法中范围半径的取值
    n_points = 8 * radius  # 领域像素点数

    for train_or_test in whole_data:
        LBP_histogram_features = pd.DataFrame()
        for pic_index, pic_data in enumerate(whole_data[train_or_test]):
            image, label = pic_data
            image = image.view(28, 28)
            # LBP特征
            lbp = local_binary_pattern(np.array(image), n_points, radius)

            LBP_histogram_features_temp = {pic_index: lbp.ravel()}
            # print(hog_features_temp)
            LBP_histogram_features_temp = pd.DataFrame.from_dict(LBP_histogram_features_temp, orient='index')
            # print(hog_features_temp)
            LBP_histogram_features = LBP_histogram_features._append(LBP_histogram_features_temp, ignore_index=True)
            if (pic_index + 1) % 2000 == 0:
                print(f"当前提取了{pic_index + 1}/{len(whole_data[train_or_test])}的数据.")

        print(LBP_histogram_features)
        try:
            save_path = f"../features/{train_or_test}"
            os.makedirs(save_path)
        except:
            pass
        LBP_histogram_features.to_csv(f"../features/{train_or_test}/lbp_features.csv")