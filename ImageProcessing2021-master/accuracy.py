import numpy as np
import math
import mnist
X = mnist.download_and_parse_mnist_file("t10k-images-idx3-ubyte.gz")
Y = mnist.download_and_parse_mnist_file("t10k-labels-idx1-ubyte.gz")

ImageSize = 28
D = ImageSize ** 2
ImageNum = 10000
ClassNum = 10
MiddleNodeNum = 100

def inputFileName():
    return np.load("temp50_A1.npy")

def preprocess():
    file = inputFileName()
    image = X
    return image, file

def inputLayer(image):
    image784 = image.reshape(ImageNum, D).T
    return image784

def combine(input, weight, b):
    return np.dot(weight, input) + b

def sigmoid(t):
    return 1/(1+np.exp(-t))

def softmax(x):
    y = np.exp(x - np.max(x, axis=0))
    f_x = y / np.sum(y, axis=0)
    return f_x

def propagation(img, w1, b1, w2, b2):
    x = inputLayer(img)
    y1 = sigmoid(combine(x, w1, b1))
    a = combine(y1, w2, b2)
    y2 = softmax(a)
    max = np.argmax(y2, axis=0)
    return max

def accuracy():
    (img, (w1, b1, w2, b2)) = preprocess()
    max = propagation(img, w1, b1, w2, b2)
    print(str((np.count_nonzero(Y == max)/ImageNum)*100) + "%")

accuracy()

