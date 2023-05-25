from nn import NeuralNetwork
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

net = NeuralNetwork(2, 4, 1)

x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

net.train(x, y, 0.1, 10000)

predictions = net.predict(x)
print(predictions)

def f1(X):
  return str(x)

arr1 = np.array([])
arr2 = np.array([])
for xi in x:
  arr1 = np.append(arr1, str(xi))
for xi in predictions:
  arr2 = np.append(arr2, xi[0])
  
plt.plot(arr1, arr2)
plt.show()
