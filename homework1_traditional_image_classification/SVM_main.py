from torchvision.datasets import FashionMNIST
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 定义一个显示图像的函数
def show_image(img):
    plt.imshow(img.permute(1, 2, 0))  # 将图像的通道维度提到最外层
    plt.axis('off')  # 不显示坐标轴
    plt.show()

def get_PixelFeature(data):
    images = []
    for i in data:
        images.append(i[0].view(28*28))

    return images

def get_labels(data):
    labels = []
    for i in data:
        labels.append(i[1])

    return labels


# 导入FashionMNIST数据集
transform_funcs = transforms.Compose([
    transforms.ToTensor()
])

train_data = FashionMNIST(root='./data', train=True, download=True, transform=transform_funcs)
test_data = FashionMNIST(root='./data', train=False, download=True, transform=transform_funcs)


# 使用像素值作为特征
train_PixelFeature = get_PixelFeature(train_data)
test_PixelFeature = get_PixelFeature(test_data)


feature = 'Pixel'# 可以选择'Pixel', 'LBP', 'GLCM', 'HOG'

if feature == 'Pixel':
    train_used_features = train_PixelFeature
    test_used_features = test_PixelFeature
else:
    feature_names = {'LBP': 'lbp_features.csv', 'GLCM': 'glcm_features.csv', 'HOG': 'hog_features.csv'}
    try:
        train_used_features = pd.DataFrame(pd.read_csv(fr"./features/train/{feature_names[feature]}"))
    except:
        print("请先调用feature_extraction中的代码提取特征并保存到表格")
    train_used_features.set_index(train_used_features.columns[0], inplace=True)
    test_used_features = pd.DataFrame(pd.read_csv(fr"./features/test/{feature_names[feature]}"))
    test_used_features.set_index(test_used_features.columns[0], inplace=True)

train_data_labels = get_labels(train_data)
test_data_labels = get_labels(test_data)


model = SVC(verbose=True, kernel="poly", random_state=42)
model.fit(train_used_features, train_data_labels)
train_predict = model.predict(train_used_features)
test_predict = model.predict(test_used_features)
train_accuracy = accuracy_score(train_data_labels, train_predict)
test_accuracy = accuracy_score(test_data_labels, test_predict)
print(train_accuracy)
print(test_accuracy)




