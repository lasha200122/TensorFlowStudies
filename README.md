# TensorFlowStudies
MIT Introduction to Deep Learning 

## example of creating a dense layer like g(w_0 + X^T W)

```python

class DenseLayer(tf.keras.layers.Layer):
  def __init__(self, input_dim, output_dim):
    super(DenseLayer, self).__init__()

    self.W = self.add_weight([input_dim, output_dim])
    self.b = self.add_weight([1, output_dim])

  
  def call(self, inputs):
    z = tf.matmul(inputs, self.W) + self.b

    output = tf.math.sigmoid(z)

    return output

```

Tensorflow has already implemented this class to library and we can call it easily using this code:

```python
kayer =tf.keras.layers.Dense(units=2)
```

We can also stack dense laers using tensorflow, where n is number of inputs
```python
model = tf.keras.Sequential([tf.keras.layers.Dense(n), tf.keras.layers.Dense(2)])
```

# Mean Squared Error Loss

Mathematical visualization
```math

J \left( W \right) = \frac{1}{n} \sum_{i = 1}^{n} \left( y^{(i)} - f \left( x^{(i)} ; W \right)\right)^{2}

```

Python Tensorflow Code:
```python

loss = tf.reduce_mean(tf.square(tf.subtract(y, predicted)))
loss = tf.keras.losses.MSE(y , predicted)

```

# Loss Optimization
We want to find the network weights that achive the lowest loss

```math
W^{*} = \text{argmin}_{W} \frac{1}{n} \sum_{i=1}^{n} L \left( f \left( x^{(i)} ; W \right), y^{(i)} \right)
```
```math
W^{*} = \textbf{argmin}_{W} J \left(W\right)

```

# Gradient Descent
Algorithm
1) Initialize weights randomly 
2) Loop until Convergence
3) Compute gradient
4) Update weights
5) Return weights

```python

weights = tf.Variable([tf.random.normal()])

while True:
  with tf.GradientTape() as g:
    loss = compute_loss(weights)
    gradient = g.gradient(loss, weights)
  
  weights = weights - lr * gradient


```
