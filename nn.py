import numpy as np

class NeuralNetwork:
  def __init__(self, input_size, hidden_size, output_size):
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.output_size = output_size
    
    self.weights1 = np.random.randn(self.input_size, self.hidden_size)
    self.weights2 = np.random.randn(self.hidden_size, self.output_size)
    
    self.bias1 = np.zeros((1, self.hidden_size))
    self.bias2 = np.zeros((1, self.output_size))
    
  def forward(self, x):
    self.hidden_layer = sigmoid(np.dot(x, self.weights1) + self.bias1)
    self.output_layer = sigmoid(np.dot(self.hidden_layer, self.weights2) + self.bias2)
    return self.output_layer
  
  def train(self, x, y, learning_rate, epochs):
    for epoch in range(epochs):
      output = self.forward(x)
      
      error = y - output
      delta_output = error * sigmoid_derivative(output)
      
      error_hidden = delta_output.dot(self.weights2.T)
      delta_hidden = error_hidden * sigmoid_derivative(self.hidden_layer)
      
      self.weights2 += self.hidden_layer.T.dot(delta_output) * learning_rate
      self.weights1 += x.T.dot(delta_hidden) * learning_rate
      
      self.bias2 += np.sum(delta_output, axis=0) * learning_rate
      self.bias1 += np.sum(delta_hidden, axis=0) * learning_rate
  
  def predict(self, x):
    return self.forward(x)
  
def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
  return x * (1 - x)