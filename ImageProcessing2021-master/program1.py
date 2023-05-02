import numpy as np
import math
import mnist
import matplotlib.pyplot as plt
from pylab import cm
X = mnist.download_and_parse_mnist_file("t10k-images-idx3-ubyte.gz")
Y = mnist.download_and_parse_mnist_file("t10k-labels-idx1-ubyte.gz")

ImageSize = 28
D = ImageSize ** 2
ImageNum = 10000
ClassNum = 10
MiddleNodeNum = 100

def inputNumber():
    num = input("Please enter the number as you like. (0 ~ " + str(ImageNum - 1) + ")\n")
    try:
        num = int(num)
    except:
        print("Please enter the 'number'.")
        return inputNumber()
    if num < 0 or num > (ImageNum - 1):
        print("Please enter '0 ~ " + str(ImageNum - 1) + "'.")
        return inputNumber()
    else:
        return num

def random(row, column, preNodeNum):
    array = np.random.normal(
                loc = 0,
                scale = math.sqrt(1/preNodeNum),
                size = (row, column),
            )
    return array

def preWB():
    w1 = random(MiddleNodeNum, D, D)
    b1 = random(MiddleNodeNum, 1, D)
    w2 = random(ClassNum, MiddleNodeNum, MiddleNodeNum)
    b2 = random(ClassNum, 1, MiddleNodeNum)
    wb = w1, b1, w2, b2
    return wb

def preprocess():
    i = inputNumber()
    np.random.seed(i)
    image28 = X[i]
    ans = Y[i]
    return image28, ans

def inputLayer(image):
    image784 = image.reshape(D, 1)
    return image784

def combine(input, weight, b):
    return np.dot(weight, input) + b

def sigmoid(t):
    return 1/(1+np.exp(-t))

def softmax(x):
    y = np.exp(x - np.max(x))
    f_x = y / np.sum(y)
    return f_x

def main1():
    (img, ans) = preprocess()
    (w1, b1, w2, b2) = preWB()
    x = inputLayer(img)
    y1 = sigmoid(combine(x, w1, b1))
    a = combine(y1, w2, b2)
    y2 = softmax(a)
    max = np.argmax(y2)
    print("result: " + str(max))
    print("answer: " + str(ans))
    plt.imshow(img, cmap=cm.gray)
    plt.show()

main1()


