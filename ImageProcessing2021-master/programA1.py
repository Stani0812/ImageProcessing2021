import numpy as np
import mnist
X = mnist.download_and_parse_mnist_file("train-images-idx3-ubyte.gz")/255
Y = mnist.download_and_parse_mnist_file("train-labels-idx1-ubyte.gz")

ImageSize = 28
D = ImageSize ** 2
ImageNum = 60000
ClassNum = 10
MiddleNodeNum = 100
BatchSize = 100
LearnPara = 0.01
Epoc = 50
ActiveFanction = 1
# 0:sigmoid, 1:ReLU

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
    return 1/(1+np.exp(-t))

def relu(t):
    return np.maximum(0.0, t)

def softmax(x):
    y = np.exp(x - np.max(x, axis=0))
    f_x = y / np.sum(y, axis=0)
    return f_x

def crossEntropy(answer, out):
    onehot = np.identity(10)[answer]
    e1 = np.dot(-onehot, np.log2(out))
    e2 = (np.sum(np.diag(e1)))/BatchSize
    e = onehot, e2
    return e

def propagation(img, ans, w1, b1, w2, b2):
    x = inputLayer(img)
    if ActiveFanction == 0:
        y1 = sigmoid(combine(x, w1, b1))
    else:
        y1 = relu(combine(x, w1, b1))
    a = combine(y1, w2, b2)
    y2 = softmax(a)
    max = np.argmax(y2, axis=0)
    (onehot, e) = crossEntropy(ans, y2)
    prop = (x, y1, y2, onehot, e, max)
    return prop

def revEntropy(y2, onehot):
    return (y2 - onehot.T) / BatchSize

def revCombine(w, x, enY):
    enX = np.dot(w.T, enY)
    enW = np.dot(enY, x.T)
    enB = np.sum(enY, axis=1).reshape(-1,1)
    en = enX, enW, enB
    return en

def revSigmoid(y, enY):
    return enY*(1-y)*y

def revRelu(y, enY):
    return enY*np.sign(relu(y))

def renew(w1, b1, w2, b2, enW1, enW2, enB1, enB2):
    N = LearnPara
    newW1 = w1 - N*enW1
    newB1 = b1 - N*enB1
    newW2 = w2 - N*enW2
    newB2 = b2 - N*enB2
    new = newW1, newB1, newW2, newB2
    return new

def backPropagation(w1, b1, w2, b2, prop):
    (x, y1, y2, onehot, e, max) = prop
    enY2 = revEntropy(y2, onehot)
    (enX2, enW2, enB2) = revCombine(w2, y1, enY2)
    if ActiveFanction == 0:
        enY1 = revSigmoid(y1, enX2)
    else:
        enY1 = revRelu(y1, enX2)
    (enX1, enW1, enB1) = revCombine(w1, x, enY1)
    new = renew(w1, b1, w2, b2, enW1, enW2, enB1, enB2)
    return new, max, e

def loadOrTemplate():
    ynL = input("Do you have a file? y/n ")
    if ynL == "y":
        name = input("What's your file's name? ('.npy' isn't needed!) ")
        try:
            return np.load(name + ".npy")
        except:
            print("Not found!")
            return loadOrTemplate()
    elif ynL == "n":
        return preWB()
    else:
        print("Please 'y' or 'n'.")
        return loadOrTemplate()

def save(data):
    ynS = input("Do you save file? y/n ")
    if ynS == "y":
        name = input("Please enter the file's name. ('.npy' isn't needed!) ")
        print("Saved.")
        return np.save(name, data)
    elif ynS == "n":
        print("Done.")
    else:
        print("Please enter 'y' or 'n'.")
        return save(data)

def mainA1():
    (w1, b1, w2, b2) = loadOrTemplate()
    total = 0.
    accuracy = 0
    for i in range(Epoc):
        for j in range(int(ImageNum/BatchSize)):
            (img, ans) = preprocess()
            prop = propagation(img, ans, w1, b1, w2, b2)
            ((w1, b1, w2, b2), max, e) = backPropagation(w1, b1, w2, b2, prop)
            total = total + e
            accuracy = accuracy + np.count_nonzero(ans == max)/BatchSize
        print(str(i+1) + "_epoc  : " + str(total/(ImageNum/BatchSize)))
        total = 0.
        print(str(i+1) + "_result: " + str(int((accuracy/(ImageNum/BatchSize))*100))+ "%")
        accuracy = 0
    save((w1, b1, w2, b2))

mainA1()
