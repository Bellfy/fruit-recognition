
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

train_dir = 'C:\\Users\\21164\Desktop\\MY_data\\train'
test_dir = 'C:\\Users\\21164\\Desktop\\MY_data\\test'
# Load training dataset without preprocessing
#函数：从文件系统中目录加载图像数据
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    validation_split=0.2,#从train_dir中划分训练集与验证集
    subset='training',
    batch_size=32,
    image_size=(100, 100),
    seed=123,
    shuffle=True,#创建数据集时对数据进行洗牌
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    validation_split=0.2,#划分数据集
    subset='validation',
    batch_size=32,#每个批次样本数量
    image_size=(100, 100),#设置输入图像的尺寸
    seed=42,#设置随机种子，确保分割的可重复性  
)
#在训练集中获取子目录类别名
class_names = sorted(os.listdir(train_dir))
#打印类别名和类别数量
print("Class Names:", class_names)
num_classes = len(class_names)
print("Number of Classes:", num_classes)
# Function to display images from a dataset
#展示数据集中的一批图像
def show_images(dataset, class_names):
    plt.figure(figsize=(100, 100))#设置图像窗口大小
    for i, (images, labels) in enumerate(dataset.take(25)):  # 取出一个批次的数据
            ax = plt.subplot(5, 5, i+ 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")
show_images(train_ds, class_names)
plt.show()
#根据系统硬件和当前工作负载自动选择最优的数据预处理配置，以提高性能
AUTOTUNE = tf.data.experimental.AUTOTUNE
#cach将训练数据集缓存到内存中，加速数据加载，shuffle随机打乱，提高模型泛化能力，
#确保了每次迭代数据顺序不同，预先加载数据，减少等待时间
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
#数据增强
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),#随机水平翻转层，增加数据多样性
    tf.keras.layers.RandomRotation(0.5)#随机旋转层，旋转最多50%的角度
    #其他数据增强方式
    #tf.keras.layers.experimental.preprocessing.RandomColor(factor=0.1)
    #tf.keras.layers.RandomFlip('vertical'),
    #tf.keras.layers.RandomZoom(height_factor=(0.8,1),width_factor=(0.8,1)),#随机缩放层
    #tf.keras.layers.RandomCrop(height=100,width=100,seed=123),#随机裁剪层
    #tf.keras.layers.Rescaling(1.0/255)#归一化层

])
#迁移学习的模型构建
#不包括顶部全连接层
base_model = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_shape=(100, 100, 3))
preprocess_input = tf.keras.applications.resnet.preprocess_input#预处理输入图像，使其符合resnet50模型输入要求
base_model.trainable = False#冻结基础模型的权重
#对特征图平均池化，减少特征维度，得到维度（batch_size,1,1,featrue_dim)的输出
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer = tf.keras.layers.Dense(num_classes)#输出传递给dense层，用于分类，输出维度（类别数）为num_class
#构建完整模型
inputs = tf.keras.Input(shape=(100, 100, 3))#定义模型输入
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)#通过基础模型传递数据，设置False以确保在推理模式下运行
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)#减少过拟合
outputs = prediction_layer(x)#通过全连接层进行分类
#创建并定义整个模型，包括输入，输出
model = tf.keras.Model(inputs=inputs, outputs=outputs)
#配置和训练基于迁移学习的图像分类模型
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)#实例化Adam优化器
#编译模型
model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),#稀疏多分类交叉熵损失函数，直接使用logits计算损失函数
    metrics=['accuracy']#设置列表，包含了用于评估模型性能的指标
)
base_model.summary()#查看基础模型架构，调用summay方法打印架构信息，包括层数，每层神经元数量和输入/输出形状
#训练模型，用fit方法
history = model.fit(
    train_ds,
    epochs=10,#训练轮数
    validation_data=val_ds,#指定了验证数据集，用于在训练过程评估模型性能 
)
#可视化
plt.plot(history.history['accuracy'], label='Training Accuracy', marker='o')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy',marker='o')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()
plt.plot(history.history['loss'], label='Training Loss',marker='o')
plt.plot(history.history['val_loss'], label='Validation Loss',marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
def predict(model, img):
    #将图像转换为数组
    img_array = tf.keras.utils.img_to_array(images[i].numpy())
    #在数组第0维增加一个维度，因为模型期望输入是一个批次的图像
    img_array = tf.expand_dims(img_array, 0)
    #对图像进行预测，返回一个包含预测概率的数组
    predictions = model.predict(img_array)
    #np.argmax找到预测概率最高的类别索引，根据索引获取预测的类别名称
    predicted_class = class_names[np.argmax(predictions[0])]
    #计算最高概率的百分比，并保留两位小数
    confidence = round((10*np.max(predictions[0])), 2)
    return predicted_class, confidence
plt.figure(figsize=(15, 15))#设置图像显示大小
for images, labels in val_ds.take(1):#从验证集取出一个批次
    for i in range(25):
        ax = plt.subplot(5, 5, i+1) 
        plt.imshow(images[i].numpy().astype("uint8"))
       #对每张图像进行预测
        predicted_class, confidence = predict(model, images[i].numpy())
        actual_class = class_names[labels[i]]
        plt.title(f"Actual: {actual_class}, \n Predicted: {predicted_class}.\n Confidence: {confidence}%")
        plt.axis('off')
plt.tight_layout()
plt.show()
#测试集
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir, 
    batch_size=32, 
    image_size=(100, 100), 
    seed=123,
    shuffle=False
)
# 获取测试集中的类别名
class_names = sorted(os.listdir(test_dir))
test_loss, test_accuracy = model.evaluate(test_ds, verbose=2)
print(f'Test accuracy: {test_accuracy}')
plt.figure(figsize=(15, 15))#设置图像显示大小
for i,(images, labels) in enumerate(test_ds.take(1)):
    for i in range(25):
        ax = plt.subplot(5, 5, i+1) 
        plt.imshow(images[i].numpy().astype("uint8"))
        predicted_class, confidence = predict(model, images[i].numpy())
        actual_class = class_names[labels[i]]

        plt.title(f"Actual: {actual_class}, \n Predicted: {predicted_class}.\n Confidence: {confidence}%")

        plt.axis('off')
plt.tight_layout()
plt.show()

