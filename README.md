# Pytorch-Common-Template
Storing functions commonly used in training pytorch models

## Training Template
```
# 验证集分割 数据增强 记录损失曲线 AdaBound优化器 动态显示训练过程(类似caffe) 自动设置线程数 图片异常跳过
Solution(model, # the model need training 训练用的模型
         criterion, # loss function 损失函数
         optimizer, # default: AdaBound 优化器：默认为AdaBound
         image_path, # all image folder 所有图片的文件夹
         scheduler=None, # 调度器
         epochs=100, # 训练轮次
         batch_size=64, # 训练批次
         show_interval=20, # show loss and accuracy per training show_interval batches 每训练show_interval个批次显示一次损失与准确率
         valid_interval=100, # test the model per valid_interval batches 每训练valid_interval个批次测试一次模型
         record_loss=True, # Whether to use tensorboardX to record the training process 是否使用tensorboardX记录训练过程
         image_size=224, # image input size 图片输入尺寸
         thread_size=8, # default: number of cpu 线程数：默认为cpu核心数
         seed=66, # random state 随机数种子
         ten_crops=False, # Whether to use TenCrops data augmentation 是否使用TenCrops增强训练数据
         test_ratio=0.25, # default: 1/4 images in image folder will be used to test 图片文件夹中用于验证集的比率
         )
         
# images should be placed in folders like:
    # --root
    # ----root\dogs
    # ----root\dogs\image1.png
    # ----root\dogs\image2.png
    # ----root\cats
    # ----root\cats\image1.png
    # ----root\cats\image2.png
    # path: the root of the image folder
```
## Tools
```
# Split the entire picture folder into a training set and a validation set, with a directory structure lik: root directory {the directory for each category {the corresponding pictures}}
# 分割整个图片文件夹为训练集和验证集，目录结构为根目录{每个类别的目录{对应类别的图片}}
ImageFolderSplitter(path, # 所有图片文件夹
                    train_size=0.8, # 图片文件夹中用于训练集的比率
                    seed=66,
                    drop_type=('txt', 'csv'), # skip file with these suffix 遍历图片目录时需要跳过的文件类型
                    )
```
## References :blush:
[文件夹分割](https://blog.csdn.net/xgbm_k/article/details/84325347)

[AdaBound](https://github.com/Luolc/AdaBound)

[Pytorch预训练模型](https://github.com/Cadene/pretrained-models.pytorch)

[AdamW](https://github.com/egg-west/AdamW-pytorch)
