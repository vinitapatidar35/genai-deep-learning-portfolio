# Chapter 1 - PyTorch Workflow

---

## Topic 33 - Introduction to the PyTorch Workflow

Building a model in PyTorch follows the same 6 steps every time.

| Step | Workflow step                 | What happens                                            |
|------|-------------------------------|---------------------------------------------------------|
| 1    | Get data ready                | Turn the data into tensors                              |
| 2    | Build a model                 | Create the structure that will learn                    |
| 3    | Fit the model to the data     | Training: the model adjusts its numbers to fit the data |
| 4    | Make predictions and evaluate | Check how well the model works                          |
| 5    | Save and load the model       | Keep what the model learned for later use               |
| 6    | Put it all together           | Run every step from start to finish                     |

### What Chapter 1 builds

A straight-line model based on a single neuron:

```text
output = input x weight + bias
```

We create data from a line with a known weight and bias. The model starts with random numbers and has to discover the weight and bias by itself, from the data alone.

**Remember**

- Training (step 3) is where the model learns
- Saving a model keeps its learned weights and biases, so it doesn't need training again
- Saved models can be reused later, shared, used in apps, or trained further

---

## Topic 34 - Getting Set Up

```python
import os                                    # load Python's operating-system tools
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # avoid the OpenMP crash on Windows (torch + matplotlib)
import torch                                 # load PyTorch
from torch import nn                         # load PyTorch's model-building parts
import matplotlib.pyplot as plt              # load the chart-drawing part of matplotlib
```

### Importing

| Code                   | What it means                  | Use it as                  |
|------------------------|--------------------------------|----------------------------|
| `import torch`         | Bring in the whole library     | `torch.rand(...)`          |
| `import numpy as np`   | Bring it in with a nickname    | `np.array(...)`            |
| `from torch import nn` | Take one part out of a library | `nn` instead of `torch.nn` |

`nn` (neural network) is PyTorch's toolbox of ready-made model parts: the base for every model (`nn.Module`), learnable numbers (`nn.Parameter`), loss functions, and layers (`nn.Linear`).

### matplotlib

| Function                              | What it does                         |
|---------------------------------------|--------------------------------------|
| `plt.scatter(x, y)`                   | Draws separate dots                  |
| `plt.plot(x, y)`                      | Draws a connected line               |
| `plt.xlabel(...)` / `plt.ylabel(...)` | Labels the x-axis / y-axis           |
| `plt.title(...)`                      | Adds a title                         |
| `plt.legend()`                        | Shows the names given with `label=`  |
| `plt.show()`                          | Displays the chart (and finishes it) |

**Watch out**

- `import matplotlib` alone does not load the drawing functions; use `import matplotlib.pyplot as plt`
- After a kernel restart, run the import cell again or you get `NameError`
- On Windows, torch + matplotlib can crash the kernel; the `KMP_DUPLICATE_LIB_OK` line fixes it

---

## Basics - Straight Lines and How a Model Learns

A straight line:

```text
y = weight x X + bias
```

| Part   | Meaning                           | Effect on the line         |
|--------|-----------------------------------|----------------------------|
| X      | The input                         | Position along the x-axis  |
| weight | Number the input is multiplied by | Tilts the line (how steep) |
| bias   | Number added at the end           | Lifts the line up or down  |
| y      | The output                        | Position along the y-axis  |

**Linear regression** = finding the weight and bias of the straight line that best fits the data (linear = straight line, regression = predicting a number).

| Term         | Meaning                                                     |
|--------------|-------------------------------------------------------------|
| X (features) | The inputs                                                  |
| y (labels)   | The correct answers                                         |
| Prediction   | The model's guess                                           |
| Loss         | One number showing how wrong the predictions are            |
| Training     | Repeatedly adjusting the weight and bias to reduce the loss |

### How a model learns

1. Start with a random weight and bias
2. Predict outputs
3. Measure the loss (how wrong the predictions are)
4. Adjust the weight and bias to reduce the loss
5. Repeat until the predictions are close to the labels

**Remember**

- Without a bias, every line must pass through (0, 0)
- Weights hold most of what a model "knows"
- Stop training when the loss stops going down, or when the test loss starts going up

---

## Topic 35 - Creating a Dataset

We create our own data from a known straight line, so we know the right answer and can check later whether the model learned it.

```python
weight = 0.7                                         # the weight we choose
bias = 0.3                                           # the bias we choose

X = torch.arange(0, 1, 0.02).unsqueeze(dim=1)        # 50 inputs from 0 to 0.98, as a column
y = weight * X + bias                                # the correct answer (label) for every input
```

| Step | Code                           | What it does                                      |
|------|--------------------------------|---------------------------------------------------|
| 1    | `weight = 0.7`, `bias = 0.3`   | Choose the rule (the answer key)                  |
| 2    | `X = torch.arange(0, 1, 0.02)` | Make the inputs (questions)                       |
| 3    | `y = weight * X + bias`        | Calculate the correct answers (labels)            |
| 4    | `.unsqueeze(dim=1)`            | Arrange the data as a column: [samples, features] |

First few data points:

| X (input) | 0.7 x X + 0.3   | y (label) |
|-----------|-----------------|-----------|
| 0.0       | 0.7 x 0.0 + 0.3 | 0.30      |
| 0.2       | 0.7 x 0.2 + 0.3 | 0.44      |
| 0.4       | 0.7 x 0.4 + 0.3 | 0.58      |

**Remember**

- X = inputs (features), y = correct answers (labels)
- Number of points = (end - start) / step, e.g. 1 / 0.02 = 50
- Models expect data shaped as [samples, features]; X has shape [50, 1]
- The model will see X and y, but not the weight and bias

**Watch out**

- `arange` does not include the end value
- Without `.unsqueeze(dim=1)`, X is a flat row with shape [50]

---

## Topic 36 - Training and Test Sets

We split the data so we can check whether the model really learned, or just memorised.

| Set          | School version | Purpose                                      | Usual size |
|--------------|----------------|----------------------------------------------|------------|
| Training set | Textbook       | The model learns from it                     | About 80%  |
| Test set     | Final exam     | Checks the model on data it has never seen   | About 20%  |

**Generalization** = doing well on new, unseen data (the real goal).

### Splitting in code

```python
train_split = int(0.8 * len(X))                             # 80% of the examples
train_data, train_label = X[:train_split], y[:train_split]  # first 80%: inputs and labels
test_data, test_label = X[train_split:], y[train_split:]    # last 20%: inputs and labels
```

### Seeing the split and the predictions

```python
def plot_predictions(train_data=train_data, train_labels=train_label,
                     test_data=test_data, test_labels=test_label,
                     predictions=None):
    plt.scatter(train_data, train_labels, label="Training data")  # draw the training examples
    plt.scatter(test_data, test_labels, label="Testing data")     # draw the test examples
    if predictions is not None:                                    # only if predictions were given...
        plt.scatter(test_data, predictions, label="Predictions")  # ...draw them too
    plt.legend()                                                   # show which colour is which
    plt.show()                                                     # display the chart

plot_predictions()  # draw the training and test data (no predictions yet)
```

- Test labels and predictions are drawn at the same inputs (`test_data`), so they can be compared
- Prediction dots on top of the test dots = good model; a big gap = large error

**Remember**

- Testing on training data can't show if the model really learned
- Cut X and y at the same position, so each input keeps its own label
- `None` means "nothing"; `if predictions is not None:` draws predictions only when given
- Real projects usually shuffle the data before splitting; a validation set is also common

**Watch out**

- `int(...)` is needed because positions must be whole numbers
- Run the data cells before the function cell, and re-run it if the data changes

---

## Concept - How a Model Learns (Backpropagation Intuition)

The learning loop: predict, measure the loss, find the direction, take a small step, repeat.

Example: `prediction = weight x x`, with one example x = 3 and correct answer y = 6. The model starts with weight = 1 and uses learning rate 0.05.

| Round | Weight | Prediction (weight x 3) | Error | Loss (error squared) |
|-------|--------|-------------------------|-------|----------------------|
| Start | 1.00   | 3.00                    | -3.00 | 9.00                 |
| 1     | 1.90   | 5.70                    | -0.30 | 0.09                 |
| 2     | 1.99   | 5.97                    | -0.03 | 0.0009               |

Update rule:

```text
new weight = old weight - learning rate x gradient
```

| Term             | Meaning                                                                |
|------------------|------------------------------------------------------------------------|
| Loss             | How wrong the prediction is                                            |
| Gradient         | Which way to change a weight, and how strongly the loss reacts         |
| Backpropagation  | Works out the gradient for every weight, going backwards from the loss |
| Learning rate    | How big each step is                                                   |
| Gradient descent | Repeatedly stepping against the gradient so the loss goes down         |

**Remember**

- The gradient points where the loss rises fastest, so we step the opposite way
- Positive gradient: decrease the weight; negative gradient: increase the weight
- Steps get smaller as the model gets closer to the right answer
- In PyTorch, `loss.backward()` does backpropagation for us (Day 3)

**Watch out**

- A learning rate that is too big overshoots the minimum and training can blow up
- A learning rate that is too small makes learning very slow

---
