import os
import keras
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image

#---------------------------------------(DenseNet-121)--（网络部分）--------------------------------------



#搭建DenseNetBlock模块
class DenseNetBlock(tf.keras.Model):
    def __init__(self,input_features,growth_rate):
        super(DenseNetBlock, self).__init__()
        #BN-ReLU-Conv
        self.batch1 = layers.BatchNormalization()
        self.relu1=layers.Activation('relu')
        self.conv1=layers.Conv2D(4*input_features,kernel_size=[1,1],strides=[1,1],padding='same')

        self.batch2=layers.BatchNormalization()
        self.relu2=layers.Activation('relu')
        self.conv2=layers.Conv2D(growth_rate,kernel_size=[3,3],strides=[1,1],padding='same')
    def call(self,inputs,training=None):
        x=self.batch1(inputs)
        x=self.relu1(x)
        x=self.conv1(x)

        x=self.batch2(x)
        x=self.relu2(x)
        x=self.conv2(x)

        x=tf.concat([
            x,inputs
        ],axis=3)

        return x

#搭建Transition 模块
class TransitionLayer(tf.keras.Model):
    def __init__(self,input_features):
        super(TransitionLayer, self).__init__()
        self.batch=layers.BatchNormalization()
        self.relu=layers.Activation('relu')
        self.conv=layers.Conv2D(input_features,kernel_size=[1,1],strides=[1,1],padding='same')
        self.avgpool=layers.AveragePooling2D(pool_size=[2,2],strides=[2,2],padding='same')
    def call(self,inputs,training=None):
        x=self.batch(inputs)
        x=self.relu(x)
        x=self.conv(x)
        x=self.avgpool(x)

        return x

#DenseNet-121
class DenseNet121(tf.keras.Model):
    def __init__(self,growth_rate,input_features,num_layers,num_classes):  #num_classes 分类类别数
        super(DenseNet121, self).__init__()
        self.growth_rate=growth_rate
        self.num_layers=num_layers

        #输入部分  input
        self.Inputs=keras.Sequential([
            layers.Conv2D(input_features, kernel_size=[7, 7], strides=[2, 2], padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPool2D(pool_size=[3,3],strides=[2,2],padding='same')
        ],name='input')


        self.densenetblock1=self.DenseNetBlocks(num_layers[0],input_features,0)  #64+32*6=256  64为input的通道数，32为x的通道数
        input_features=input_features+num_layers[0]*self.growth_rate  # 256=64+6*32
        input_features = input_features // 2  # 128=256/2
        self.transition1=self.transitionlayer(input_features,0)  #128

        self.densenetblock2 = self.DenseNetBlocks(num_layers[1], input_features,1) #128+32*12=512
        input_features = input_features + num_layers[1] * self.growth_rate  #512=128+12*32
        input_features = input_features // 2  # 256 = 512/2
        self.transition2 = self.transitionlayer(input_features,1)  #256

        self.densenetblock3 = self.DenseNetBlocks(num_layers[2], input_features,2)  #256+32*24=1024
        input_features = input_features + num_layers[2] * self.growth_rate  #1024=256+24*32
        input_features = input_features // 2      #512=1024/2
        self.transition3 = self.transitionlayer(input_features,2)  #512

        self.densenetblock4 = self.DenseNetBlocks(num_layers[3], input_features,3)  #512+32*16=1024
        input_features = input_features + num_layers[3] * self.growth_rate   #1024=512+16*32

        #输出部分
        self.avgpool=layers.GlobalAveragePooling2D()
        self.dense=layers.Dense(num_classes)
        self.softmax=layers.Activation('softmax')


    def DenseNetBlocks(self,blocks,input_features,k):     # k用来给每个模块命名的
        densenetblocks=keras.Sequential([],name='block'+str(k))
        for i in range(blocks):
            densenetblocks.add(
                DenseNetBlock(input_features+i*self.growth_rate,self.growth_rate)
            )
        return densenetblocks
    def transitionlayer(self,input_features,k):
        tranlayer=keras.Sequential([],name='tranlayer'+str(k))
        tranlayer.add(TransitionLayer(input_features))
        return tranlayer

    def call(self,inputs,training=None):
        x=self.Inputs(inputs)
        x=self.densenetblock1(x)
        x=self.transition1(x)

        x=self.densenetblock2(x)
        x=self.transition2(x)

        x=self.densenetblock3(x)
        x=self.transition3(x)

        x=self.densenetblock4(x)

        x=self.avgpool(x)
        x=self.dense(x)
        x=self.softmax(x)

        return x

model_denseNet=DenseNet121(growth_rate=32,input_features=64,num_layers=[6,12,24,16],num_classes=10)
model_denseNet.build(input_shape=(None,32,32,3))
model_denseNet.summary()


#  6,12,24,16


if __name__ == '__main__':
    print('pycharm')









