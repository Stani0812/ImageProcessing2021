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

def inputFileName():
    name = input("What's your file's name? ('.npy' isn't needed!) "+ "\n")
    try:
        file = np.load(name + ".npy")
    except:
        print("No file!")
        return inputFileName()
    return file

def preprocess():
    i = inputNumber()
    image28 = X[i]
    ans = Y[i]
    return i, image28, ans

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

def propagation(img, w1, b1, w2, b2):
    x = inputLayer(img)
    y1 = sigmoid(combine(x, w1, b1))
    a = combine(y1, w2, b2)
    y2 = softmax(a)
    max = np.argmax(y2, axis=0)
    return max

def main4(w1, b1, w2, b2):
    (i, img, ans) = preprocess()
    max = propagation(img, w1, b1, w2, b2)
    print("result: " + str(max))
    print("answer: " + str(ans))
    plt.imshow(X[i], cmap=cm.gray)
    plt.show()
    main4(w1, b1, w2, b2)

(w1, b1, w2, b2) = inputFileName()
main4(w1, b1, w2, b2)

