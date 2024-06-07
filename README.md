西电杭研院·2024年春季机器学习课程期末作业
=

作业说明
-

* 数据集：FashionMNIST
* 基础要求（占比50分）：利用「手工特征+SVM」实现分类任务，按照标准划分，进行训练和测试   

   -手工特征：可以选用像素值、HOG、SIFT、LBP等，利用VLFeat等现有代码库提取即可
* 扩展要求（占比40分）

  -核函数：分析SVM核函数类型的影响，SVM可才用LibSVM库实现；

  -对比SVM与前馈神经网络
* 分组：2-4人/组

  -每组人数为N，则需要对比 2*N 种 设置下的效果
  
  -例如，2个人一组，可以对比 （2种特征 + 2种核函数）
* 提交：将技术报告PDF以附件形式发送至gaofeihifly@qq.com

  -邮件标题及PDF文档命名：作业1 -机器学习-姓名

  -代码及结果：自建Git项目

  -技术报告：使用LaTex模板，提交PDF文档；文档中包含项目Git链接；写明分工/贡献度比例。
* 技术报告模板：https://www.overleaf.com/latex/templates/zhong-wen-ji-zhu-bao-gao-latexmo-ban-dan-lan-cjc-xelatex/tcnttxfsqykx
  技术报告规范程度（占比1 0分）

数据集说明
-
代码可以自行下载数据集。
若想使用已下载好的数据集，只需将FashionMNIST数据集文件夹移动到/homework1_traditional_image_classification/data文件夹下，这时工程文件中的代码便可正常运行。

依赖包说明
-
在命令行窗口中运行以下指令即可安装所需的依赖包。
     
      pip install -r requirements.txt.

其他说明
-
在运行SVM_main.py前，需要先运行feature_extraction文件夹中的代码以提取特征。
