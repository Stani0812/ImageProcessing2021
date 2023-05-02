import numpy as np
import mnist
X = mnist.download_and_parse_mnist_file("train-images-idx3-ubyte.gz")
Y = mnist.download_and_parse_mnist_file("train-labels-idx1-ubyte.gz")

ImageSize = 28
D = ImageSize ** 2
ImageNum = 60000
ClassNum = 10
MiddleNodeNum = 100
BatchSize = 100

np.random.seed(0)

def random(row, column, preNodeNum):
    array = np.random.normal(
                loc = 0.,
                scale = np.sqrt(1/preNodeNum),
                size = (row, column)
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
    i = np.random.choice(ImageNum, BatchSize, replace=False)
    image = X[i]
    answer = Y[i]
    return image, answer

def inputLayer(image):
    image784 = image.reshape(BatchSize, D).T
    return image784

def combine(input, weight, b):
    return np.dot(weight, input) + b

def sigmoid(t):
    return 1./(1.+np.exp(-t))

def softmax(x):
    y = np.exp(x - np.max(x, axis=0))
    f_x = y / np.sum(y, axis=0)
    return f_x

def crossEntropy(answer, out):
    onehot = np.identity(10)[answer]
    e = np.dot(-onehot, np.log2(out))
    return (np.sum(np.diag(e)))/BatchSize

def propagation(img, ans, w1, b1, w2, b2):
    x = inputLayer(img)
    y1 = sigmoid(combine(x, w1, b1))
    a = combine(y1, w2, b2)
    y2 = softmax(a)
    max = np.argmax(y2, axis=0)
    e = crossEntropy(ans, y2)
    return max, e

def main2():
    (img, ans) = preprocess()
    (w1, b1, w2, b2) = preWB()
    (max, e) = propagation(img, ans, w1, b1, w2, b2)
    #print(max)
    print(e)

main2()
