import numpy as np
import struct
import os
import matplotlib.pyplot as plt

def load_mnist_train(path, kind='train'):
    labels_path = os.path.join(path,'%s-labels-idx1-ubyte'% kind)
    images_path = os.path.join(path,'%s-images-idx3-ubyte'% kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',lbpath.read(8))
        labels = np.fromfile(lbpath,dtype=np.uint8)
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',imgpath.read(16))
        images = np.fromfile(imgpath,dtype=np.uint8).reshape(len(labels), 784)
    return images, labels

def load_mnist_test(path, kind='t10k'):
    labels_path = os.path.join(path,'%s-labels-idx1-ubyte'% kind)
    images_path = os.path.join(path,'%s-images-idx3-ubyte'% kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',lbpath.read(8))
        labels = np.fromfile(lbpath,dtype=np.uint8)
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',imgpath.read(16))
        images = np.fromfile(imgpath,dtype=np.uint8).reshape(len(labels), 784)
    return images, labels
 
 
path='data'
train_images,train_labels=load_mnist_train(path)
test_images,test_labels=load_mnist_test(path)
 
# fig=plt.figure(figsize=(8,8))
# fig.subplots_adjust(left=0,right=1,bottom=0,top=1,hspace=0.05,wspace=0.05)
# for i in range(30):
#     images = np.reshape(train_images[i], [28,28])
#     ax=fig.add_subplot(6,5,i+1,xticks=[],yticks=[])
#     ax.imshow(images,cmap=plt.cm.binary,interpolation='nearest')
#     ax.text(0,7,str(train_labels[i]))
# plt.show()