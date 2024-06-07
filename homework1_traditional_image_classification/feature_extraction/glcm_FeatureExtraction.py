import os
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops
from torchvision.datasets import FashionMNIST
from torchvision.transforms import Compose, ToTensor

if __name__ == '__main__':
    transform_funcs = Compose([
        ToTensor()
    ])
    train_data = FashionMNIST(root='../data', train=True, download=True, transform=transform_funcs)
    test_data = FashionMNIST(root='../data', train=False, download=True, transform=transform_funcs)
    print(len(test_data))

    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    levels = 8
    # glcm_features = pd.DataFrame(columns=['contrast', 'dissimilarity', 'homogeneity', 'ASM', 'energy', 'correlation'])
    # whole_data = {'train': train_data, 'test': test_data}
    whole_data = {'train': train_data, 'test': test_data}
    for train_or_test in whole_data:
        glcm_features = pd.DataFrame()
        for pic_index, pic_data in enumerate(whole_data[train_or_test]):
            image, label = pic_data
            # image:输入图像(灰度级为4的4X4的二维矩阵)
            # [1]:距离为1
            # [0, np.pi/4, np.pi/2, 3*np.pi/4]：四个方向，0度，45度，90度，135度
            # level =4:灰度级为4，这里的    灰度级要和输入的image的灰度级相对应

            result = graycomatrix(image.view([28, 28]).int(), [1], angles, levels=levels)

            # 以下输出的结果为4个方向上contrast值
            contrast = graycoprops(result, 'contrast')

            # 以下输出的结果为4个方向上dissimilarity值
            dissimilarity = graycoprops(result, 'dissimilarity')
            # 以下输出的结果为4个方向上homogeneity值
            homogeneity = graycoprops(result, 'homogeneity')
            # 以下输出的结果为4个方向上ASM值
            ASM = graycoprops(result, 'ASM')
            # 以下输出的结果为4个方向上energy值
            energy = graycoprops(result, 'energy')
            # 以下输出的结果为4个方向上correlation值
            correlation = graycoprops(result, 'correlation')
            glcm_features_temp = {}
            for index, angle in enumerate(angles):
                glcm_features_temp.update({f"contrast_{index}pi/4": contrast[:,index],
                                      f"dissimilarity_{index}pi/4": dissimilarity[:,index],
                                      f"homogeneity_{index}pi/4": homogeneity[:,index],
                                      f"ASM_{index}pi/4": ASM[:,index],
                                      f"energy_{index}pi/4": energy[:,index],
                                      f"correlation_{index}pi/4": correlation[:,index]})
            glcm_features_temp = pd.DataFrame.from_dict(glcm_features_temp)
            glcm_features = glcm_features._append(glcm_features_temp, ignore_index=True)
            if (pic_index+1) % 2000 == 0:
                print(f"当前提取了{pic_index+1}/{len(whole_data[train_or_test])}的数据.")
                # print(glcm_features)

        # glcm_features = pd.DataFrame.from_dict(glcm_features, orient='index')
        print(glcm_features)
        try:
            save_path = f"../features/{train_or_test}"
            os.makedirs(save_path)
        except:
            pass
        glcm_features.to_csv(f"../features/{train_or_test}/glcm_features.csv")

