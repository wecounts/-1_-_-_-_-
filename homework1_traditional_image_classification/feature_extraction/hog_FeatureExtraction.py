import pandas as pd
from skimage.feature import hog
import matplotlib.pyplot as plt
from torchvision.transforms import Compose, ToTensor
from torchvision.datasets import FashionMNIST
import os

def show_pic(original_image, hog_image):
    # 使用matplotlib显示原始图像和HOG图像
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)

    ax1.axis('off')
    ax1.imshow(original_image, cmap=plt.cm.gray)
    ax1.set_title('Input image')

    # Rescale histogram for better display
    # hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

    ax2.axis('off')
    ax2.imshow(hog_image, cmap=plt.cm.gray)
    ax2.set_title('HOG image')

    plt.show()


if __name__ == '__main__':
    transform_funcs = Compose([
        ToTensor()
    ])
    train_data = FashionMNIST(root='../data', train=True, download=True, transform=transform_funcs)
    test_data = FashionMNIST(root='../data', train=False, download=True, transform=transform_funcs)

    whole_data = {'train': train_data, 'test': test_data}
    for train_or_test in whole_data:
        hog_features = pd.DataFrame()
        for pic_index, pic_data in enumerate(whole_data[train_or_test]):
            image, label = pic_data
            image = image.view(28, 28)
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
            # show_pic(image, hog_image)
            # print(hog_feature)
            # print(f'HOG feature vector shape: {hog_feature.shape}')

            hog_features_temp = {pic_index: hog_feature}
            # print(hog_features_temp)
            hog_features_temp = pd.DataFrame.from_dict(hog_features_temp, orient='index')
            # print(hog_features_temp)
            hog_features = hog_features._append(hog_features_temp, ignore_index=True)
            if (pic_index + 1) % 2000 == 0:
                print(f"当前提取了{pic_index + 1}/{len(whole_data[train_or_test])}的数据.")

        print(hog_features)
        try:
            save_path = f"../features/{train_or_test}"
            os.makedirs(save_path)
        except:
            pass
        hog_features.to_csv(f"../features/{train_or_test}/hog_features.csv")
