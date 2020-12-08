########################################################################
###---------------应用自动编码器重建mnist图像
########################################################################
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

import numpy as np
import tensorflow as tf 
from tensorflow import keras
from keras.datasets import mnist
from keras.models import Model 
from keras.layers import Dense, Input
import matplotlib.pyplot as plt 


np.random.seed(42)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print("x_train.shape: ", x_train.shape)
print("x_test.shape: ", x_test.shape)
print("------------------------------------------------")
x_train = x_train.astype('float')/255. - 0.5
x_test = x_test.astype('float')/255. - 0.5
x_train = x_train.reshape((x_train.shape[0], -1))
x_test = x_test.reshape((x_test.shape[0], -1))
print("x_train.shape: ", x_train.shape)
print("x_test.shape: ", x_test.shape)
print("------------------------------------------------")

#encoding_dim 要压缩的维度
encoding_dim = 32         #压缩到2维
input_img = Input(shape=(784, )) #返回一个张量

encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(10, activation='relu')(encoded)
encoder_output = Dense(encoding_dim)(encoded)

decoded = Dense(10, activation='relu')(encoder_output)
decoded = Dense(64, activation='relu')(decoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(784, activation='tanh')(decoded)

#实例化并激活模型
autoencoder = Model(input_img, decoded)
encoder = Model(input_img, encoder_output)

autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.summary()
print("------------------------------------------------")
encoder.summary()

autoencoder.fit(x_train, x_train, 
                epochs=20,
                batch_size=256,
                shuffle=True)


num_images = 10
np.random.seed(42)
random_test_images = np.random.randint(x_test.shape[0], size=num_images)

#可视化特征压缩后的结果
encoded_imgs = encoder.predict(x_test)
decoded_imgs = autoencoder.predict(x_test)

plt.scatter(encoded_imgs[:,0],encoded_imgs[:,1],c=y_test)
plt.colorbar()
plt.show()

#绘图显示重建后的效果
plt.figure(figsize=(18, 4))

for i, image_idx in enumerate(random_test_images):
    # plot original image
    ax = plt.subplot(3, num_images, i + 1)
    print("x_test[image_idx].shape: ",x_test[image_idx].shape)
    plt.imshow(x_test[image_idx].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    # plot encoded image
    ax = plt.subplot(3, num_images, num_images + i + 1)
    print("encoded_imgs[image_idx].shape: ",encoded_imgs[image_idx].shape)
    plt.imshow(encoded_imgs[image_idx].reshape(8, 4))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # plot reconstructed image
    ax = plt.subplot(3, num_images, 2*num_images + i + 1)
    print("decoded_imgs[image_idx].shape: ",decoded_imgs[image_idx].shape)
    plt.imshow(decoded_imgs[image_idx].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
