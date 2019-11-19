import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 14})


data = np.loadtxt("train/Training Labels.txt")

src = np.zeros(shape=(2400, 784))

for c in range(2400):
    img = plt.imread('train/{}.jpg'.format(c+1))
    ser = np.array([])
    
    for i in range(28):
        for j in range(28):
            ser = np.append(ser, img[i][j] / 255)
    
    src[c] = ser

mean = np.zeros(shape=(10, 784))
variance = np.zeros(shape=(10, 784))

for c in range(10):
    for attribute in range(784):
        summation = 0
        for row in range(240):
            summation += src[row + c*240][attribute]
        mean[c][attribute] = summation / 240
        
for c in range(10):
    for attribute in range(784):
        partialVariance = 0
        for row in range(240):
            partialVariance += (src[row + c*240][attribute] - mean[c][attribute])**2
        
        
        partialVariance /= 240
        partialVariance = max(partialVariance, 0.01)
        variance[c][attribute] = partialVariance


confusionMatrix = np.zeros(shape=(11, 11))
confusionIdx = -1

for test in range(200):
    if test % 20 == 0:
        confusionIdx += 1
    
    testImg = plt.imread('test/{}.jpg'.format(test+1))
    serTestImg = np.array([])
    
    probabilityArray = np.ones(shape=(10))
    
    pi = 3.141592653589793
    exp = 2.718281828459045
    
    
    for i in range(28):
        for j in range(28):
            serTestImg = np.append(serTestImg, testImg[i][j] / 255)
            
    for c in range(10):
        for attribute in range(784):
            x = serTestImg[attribute]
            probabilityArray[c] *= (1/np.sqrt(2*pi*variance[c][attribute])) * (exp**(-(x-mean[c][attribute])**2/(2*variance[c][attribute])))
        
    maximumIdx = 0
    maximumNo = probabilityArray[0]
    
    for i in range(1, 10):
        if probabilityArray[i] > maximumNo:
            maximumIdx = i
            maximumNo = probabilityArray[i]

    confusionMatrix[confusionIdx][maximumIdx] += 1
    
print(confusionMatrix)

fig, ax = plt.subplots(figsize=(150, 10))

ax.matshow(confusionMatrix)

accuracy = 0

for i in range(11):
    for j in range(11):
        if i < 10 and j < 10:
            val = confusionMatrix[i][j]
            ax.text(j, i, val, ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
        else:
            val = confusionMatrix[i][i] / 20
            accuracy += val
            ax.text(j, i, '{:.0%}'.format(val), ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
                      
for i in range(10):
    summation = 0
    for j in range(10):
        summation += confusionMatrix[j][i]
    val = confusionMatrix[i][i] / summation
    accuracy += val
    ax.text(i, 10, '{:.0%}'.format(val), ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))

ax.text(10, 10, '{:.0%}'.format(accuracy / 20), ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))

plt.savefig('Confusion-Gauss.jpg', bbox_inches="tight")
plt.show()