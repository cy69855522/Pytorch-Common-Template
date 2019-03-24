# Pytorch-Common-Template
Storing functions commonly used in training pytorch models

Solution(model, # the model need training 训练用的模型
         criterion, # loss function 损失函数
         optimizer, # default: AdaBound 优化器：默认为AdaBound
         image_path, # all image folder 所有图片的文件夹（默认按8:2分离训练集和验证集）
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
         )
