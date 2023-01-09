import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 100)

relu = lambda x: np.maximum(x, 0)
leaky_relu = lambda x: np.maximum(0.1*x, x)
sigmoid = lambda x: 1 / (1 + np.exp(-x))
tanh = lambda x: np.tanh(x)
softmax = lambda x: np.exp(x) / np.exp(x).sum()


y1 = relu(x)
y2 = leaky_relu(x)
y3 = sigmoid(x)
y4 = tanh(x)
y5 = softmax(x)


plt.figure(figsize=(10, 10))
plt.plot(x, x, 'r-', label="main")
plt.plot(x, y1, 'b--', label="relu")
plt.plot(x, y2, 'g-', label="leaky relu")
plt.plot(x, y3, 'y', label="sigmiod")
plt.plot(x, y4, 'orange', label="tanh")
plt.plot(x, y5, 'black', label="softmax")
plt.legend()
plt.grid()
plt.show()
