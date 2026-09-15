🧠 Neural Networks — From Neuron to Adam

A practical, step-by-step understanding of how a neural network works — from a single neuron to hidden layers, ReLU, loss, backpropagation, gradients, and the Adam optimizer.

📚 Learning Progress

Neuron

Dense Layer

Multiple Neurons

Hidden Layer

ReLU

Sigmoid

Loss Function

Backpropagation

Gradients

Optimizer

Adam

XOR intuition

Traditional ML vs Neural Networks

1. What is a Neuron?

A neuron takes inputs, gives each input a weight, adds a bias, and produces an output.

The basic equation is:

$$
z = w_1x_1 + w_2x_2 + \dots + b
$$

Then an activation function can be applied:

$$
y = f(Wx+b)
$$

Example

Suppose:

x₁ = 2
x₂ = 3

w₁ = 0.5
w₂ = 0.2
b  = 1

Then:

z = (0.5 × 2) + (0.2 × 3) + 1
z = 2.6

So the neuron produces:

2.6

2. Why Do We Need Multiple Neurons?

One neuron learns one weighted combination of the input features:

$$
y = w_1x_1 + w_2x_2 + b
$$

But real problems can contain many different patterns.

Instead of:

Input
  ↓
Neuron
  ↓
Output

we can use:

                 Input
                   ↓
       ┌───────────┼───────────┐
       ↓           ↓           ↓
    Neuron 1    Neuron 2    Neuron 3
       ↓           ↓           ↓
       └───────────┼───────────┘
                   ↓
                 Output

Every neuron has its own:

weights

bias

activation

Therefore, multiple neurons can learn different transformations from the same input.

Multiple neurons give the network multiple ways to transform the same input.

3. What Does Dense(10) Mean?

keras.layers.Dense(10)

means:

Create a layer containing 10 neurons.

It does not mean:

10 data points

10 features

10 epochs

It means exactly 10 neurons.

If the input contains 3 features:

x₁
x₂
x₃

every neuron receives all 3 features, but each neuron has different weights.

Conceptually:

x₁ ─────────┬────────┬────────┐
x₂ ─────────┼────────┼────────┤
x₃ ─────────┼────────┼────────┤
            ↓        ↓        ↓
         Neuron 1 Neuron 2 Neuron 3

4. What is a Hidden Layer?

A hidden layer is a layer between the input and output layers.

Input
  ↓
Hidden Layer
  ↓
Output

Example:

model = keras.Sequential([
    keras.Input(shape=(3,)),
    keras.layers.Dense(4, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid")
])

Architecture:

3 Input Features
      ↓
4 Hidden Neurons
      ↓
1 Output Neuron

The hidden layer transforms the original input into a new internal representation.

Instead of directly learning:

Input → Output

the network learns:

Input
  ↓
Intermediate representations
  ↓
Output

5. What Does a Hidden Neuron Actually Do?

Suppose our inputs are:

hours_studied
attendance
previous_score

One hidden neuron calculates:

$$
z = w_1(hours) + w_2(attendance) + w_3(score) + b
$$

Then:

$$
h = ReLU(z)
$$

Another neuron has different weights:

$$
z = w'_1(hours) + w'_2(attendance) + w'_3(score) + b'
$$

Therefore, the same input can produce different signals in different neurons.

Important

We should not assume that every neuron learns one clean, human-interpretable feature.

For example, saying:

Neuron 1 = attendance detector
Neuron 2 = score detector

is only a simplified mental model.

In reality, neurons often learn distributed internal representations.

6. What Does ReLU Do?

ReLU stands for Rectified Linear Unit.

$$
ReLU(x) = max(0,x)
$$

Examples:

-5 → 0
-2 → 0
 0 → 0
 2 → 2
 7 → 7

So ReLU acts like a gate:

Negative signal → OFF → 0

Positive signal → ON → positive value

Important clarification

ReLU itself does not detect the pattern.

The neuron's:

Weights + Bias

determine what kind of input relationship it responds to.

Then ReLU filters the result.

Weights + Bias
      ↓
Neuron calculation
      ↓
     ReLU
      ↓
Filtered activation

7. Why is ReLU Important in Hidden Layers?

Consider:

Input
  ↓
Linear Layer
  ↓
Linear Layer
  ↓
Output

Stacking linear transformations still results in a linear transformation.

So simply adding more linear layers does not give the network the nonlinear power we want.

Now consider:

Input
  ↓
Linear + ReLU
  ↓
Linear + ReLU
  ↓
Output

The nonlinear activation allows the network to represent much more complex relationships.

This is one of the main reasons neural networks can solve problems such as XOR.

8. XOR — A Simple Demonstration

XOR behaves like this:

X₁

X₂

Output

0

0

0

0

1

1

1

0

1

1

1

0

A single linear decision boundary cannot represent this pattern directly from the original features.

A hidden layer can transform the inputs first.

Consider two hand-designed hidden neurons:

Neuron 1

$$
h_1 = ReLU(x_1-x_2)
$$

Weights:

[1, -1]

Bias:

0

Neuron 2

$$
h_2 = ReLU(x_2-x_1)
$$

Weights:

[-1, 1]

Bias:

0

Their outputs are:

Input

h₁

h₂

[0,0]

0

0

[0,1]

0

1

[1,0]

1

0

[1,1]

0

0

Then the output neuron can calculate:

$$
y=h_1+h_2
$$

Result:

0
1
1
0

Exactly XOR.

The key idea

The hidden layer changed the representation:

Original input
[0,1]

became:

Hidden representation
[0,1]

and:

[1,1]

became:

[0,0]

The output layer can now easily distinguish the cases.

Hidden layers make the problem easier by transforming the representation.

The exact weights learned by a real trained network may be different; this is a hand-built demonstration of the idea.

9. Why Not Just Use Traditional Machine Learning?

Traditional ML models are extremely useful.

Examples:

Linear Regression

Logistic Regression

Decision Trees

Random Forest

SVM

The point is not:

Neural networks are always better.

Instead:

Different models are good at different types of problems.

For small structured/tabular datasets, traditional ML is often simpler and can work extremely well.

Neural networks become especially powerful when we need to learn complex representations from data.

Examples include:

Images
Audio
Video
Natural Language
Complex nonlinear relationships
Large-scale unstructured data

Traditional ML often relies more heavily on manual feature engineering.

Neural networks can learn useful intermediate representations automatically.

10. What is Loss?

After making a prediction, the model needs to know:

How wrong was the prediction?

That is the job of the loss function.

For regression, one common loss is Mean Squared Error:

$$
MSE = \frac{1}{n}\sum(y_{actual}-y_{predicted})^2
$$

Example:

Actual      = 100
Prediction  = 90

Error       = 10
Squared     = 100

A larger prediction error generally produces a larger loss.

For binary classification, a common loss is:

Binary Cross-Entropy

11. Loss → Backpropagation → Gradients

The training process can be visualized as:

Input
  ↓
Neural Network
  ↓
Prediction
  ↓
Loss
  ↓
Backpropagation
  ↓
Gradients
  ↓
Optimizer
  ↓
Updated Weights

The goal is to reduce the loss.

12. What is a Gradient?

A gradient tells us how the loss changes with respect to a parameter.

For a weight:

$$
\frac{\partial Loss}{\partial w}
$$

Conceptually:

Gradient
   ↓
How does changing this weight affect the loss?

For example:

Gradient = +5

means the loss has a positive slope with respect to that parameter.

Gradient = -5

means the slope is negative.

The optimizer uses this information to decide how to update the parameter.

13. What is Backpropagation?

Backpropagation calculates the gradients of the loss with respect to the network's parameters.

Conceptually:

Prediction
    ↓
   Loss
    ↓
Backpropagation
    ↓
  Gradients

Backpropagation uses the chain rule to propagate gradient information backward through the network.

Important distinction:

Backpropagation calculates gradients.

It does not define the complete parameter-update strategy.

14. What is an Optimizer?

An optimizer uses the gradients to update the model's parameters.

A simple gradient-descent style update looks like:

$$
w_{new}=w_{old}-learning_rate\times gradient
$$

The optimizer's job is:

Use gradient information to improve the parameters and reduce the loss.

Common optimizers:

SGD

Adam

RMSprop

15. What is Adam?

Adam stands for:

Adaptive Moment Estimation

Adam is an optimization algorithm widely used for training neural networks.

Instead of only using the current gradient, Adam keeps moving estimates based on:

The average of gradients

The average of squared gradients

Conceptually:

Current gradient
       +
Previous gradient information
       ↓
      Adam
       ↓
Parameter update

This allows Adam to adapt the update for different parameters.

16. Adam vs Backpropagation

These are not the same thing.

Backpropagation asks:

"What is the gradient of the loss with respect to each parameter?"

Adam asks:

"Given those gradients, how should I update the parameters?"

So the complete flow is:

Prediction
    ↓
   Loss
    ↓
Backpropagation
    ↓
  Gradients
    ↓
    Adam
    ↓
Weight / Bias Update

This distinction is extremely important.

17. Complete Neural Network Training Loop

One training step can be understood as:

             INPUT
                ↓
          Forward Pass
                ↓
           Prediction
                ↓
          Loss Function
                ↓
        "How wrong are we?"
                ↓
         Backpropagation
                ↓
            Gradients
                ↓
              Adam
                ↓
         Update Parameters
                ↓
             Repeat 🔄

This process repeats for many batches and epochs.

18. Complete TensorFlow Example

import tensorflow as tf
from tensorflow import keras
import numpy as np

# -------------------------
# 1. Data
# -------------------------

X = np.array([
    [1, 50, 35],
    [2, 60, 40],
    [3, 65, 45],
    [4, 75, 55],
    [5, 80, 60],
    [6, 85, 70],
    [8, 90, 80],
    [2, 90, 80]
], dtype=np.float32)

y = np.array([
    [0],
    [0],
    [0],
    [1],
    [1],
    [1],
    [1],
    [1]
], dtype=np.float32)


# -------------------------
# 2. Neural Network
# -------------------------

model = keras.Sequential([
    keras.Input(shape=(3,)),

    # Hidden layer
    keras.layers.Dense(
        4,
        activation="relu"
    ),

    # Output layer
    keras.layers.Dense(
        1,
        activation="sigmoid"
    )
])


# -------------------------
# 3. Compile
# -------------------------

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# -------------------------
# 4. Train
# -------------------------

model.fit(
    X,
    y,
    epochs=500,
    verbose=0
)


# -------------------------
# 5. Test
# -------------------------

test_data = np.array([
    [6, 85, 70],
    [2, 50, 35],
    [5, 80, 60]
], dtype=np.float32)

predictions = model.predict(test_data)

print(predictions)

Architecture:

3 Input Features
       ↓
4 Hidden Neurons + ReLU
       ↓
1 Output Neuron + Sigmoid
       ↓
Pass / Fail

19. The Complete Mental Model

Remember the whole system like this:

NEURON
↓
Weighted combination of inputs

MULTIPLE NEURONS
↓
Different transformations of the same inputs

HIDDEN LAYER
↓
Creates intermediate representations

RELU
↓
Introduces nonlinearity and filters negative activations

OUTPUT LAYER
↓
Combines learned representations

LOSS
↓
Measures prediction error

BACKPROPAGATION
↓
Calculates gradients

GRADIENT
↓
Tells how loss changes with respect to parameters

ADAM
↓
Uses gradient information to update parameters

TRAINING
↓
Repeat until useful parameters are learned

🚀 The Big Picture

The real power of a neural network is not simply:

"It has many neurons."

The deeper idea is:

Multiple neurons + nonlinear activations allow a network to transform raw inputs into increasingly useful representations.

Then the output layer uses those representations to make the final prediction.

Raw Input
   ↓
Neuron Transformations
   ↓
Hidden Representations
   ↓
More Transformations
   ↓
Useful Representation
   ↓
Prediction
   ↓
Loss
   ↓
Gradients
   ↓
Adam
   ↓
Better Parameters
   ↓
Repeat 🔄

🧠 One-Line Summary

Neurons transform inputs, hidden layers build representations, activations add nonlinearity, loss measures the mistake, backpropagation finds gradients, and Adam updates the parameters to reduce that mistake.

🔜 Next Concepts

After understanding this foundation, the next useful topics are:

Forward Propagation in detail

Weight initialization

Learning Rate

Epoch vs Batch

Gradient Descent

SGD vs Adam

Overfitting

Dropout

Normalization

Validation Data

Training vs Testing

CNNs

Embeddings
