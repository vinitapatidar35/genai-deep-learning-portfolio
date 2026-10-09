# DA2001 Week 1 Notes: Neurons, Feedforward Networks and PyTorch Basics


**How to use these notes:** read Parts 1-5 (theory) first, then Parts 6-8 (PyTorch), then Part 9 (the three tricks), then Part 10 (exam traps).

---

## Part 1. What is deep learning?

**Machine learning (ML)**: instead of writing rules by hand, you give a computer examples (data) and it finds the rules itself.

**Deep learning (DL)**: ML that uses *neural networks with many layers*. "Deep" means many layers stacked.

| | Classical ML | Deep learning |
|---|---|---|
| Features | You design them by hand (e.g. "average pixel brightness") | The network learns features itself from raw data |
| Data needed | Works with small/medium data | Needs lots of data |
| Compute | Cheap | Expensive (GPUs) |
| Best for | Tabular data | Images, text, audio |

**ANN** = Artificial Neural Network = a network of simple math units (neurons) connected by numbers (weights). Training = adjusting those numbers until outputs are good.

**Nesting:** AI ⊃ Machine Learning ⊃ Deep Learning. Traditional programming = you write the rules. ML = the computer learns the rules from data. DL = ML with multi-layer neural networks that learn features automatically from raw data (images, text, audio).

**Why deep learning works now (3 reasons):**
1. **Massive data:** the internet produces billions of images and texts every day.
2. **Powerful GPUs:** trillions of operations per second.
3. **Better algorithms:** ReLU, Adam, batch normalization, Transformers.

**A neural network has 3 kinds of layers:**
- **Input layer:** receives the data (e.g. pixel values).
- **Hidden layer(s):** transform the data step by step.
- **Output layer:** produces the prediction.

A single neuron does 3 things: multiply the inputs by weights → add them up → apply an activation.

**Types of learning**
| Type | How it works | Example |
|---|---|---|
| Supervised | Learn from labeled examples (question + answer) | Image classification |
| Unsupervised | Find hidden patterns, no labels | Customer clustering |
| Reinforcement | Trial and error with rewards | Game-playing AI |

This course (and the regression/classification examples in Week 1) uses **supervised** learning.

---

## Part 2. The neuron: from biology to math

### 2.1 Biological neuron
- Receives signals through **dendrites**, adds them up in the **cell body**, and if the total is big enough it "fires" a signal along the **axon** to the next neurons.
- Key idea copied by ANNs: *many inputs, combine, fire if strong enough.*

### 2.2 McCulloch-Pitts (MP) neuron (1943)
- Inputs are 0 or 1. Output is 0 or 1.
- Rule: add up the inputs. If sum ≥ θ (threshold), output 1, else 0.
- **AND gate:** θ = 2 (both inputs must be on). **OR gate:** θ = 1 (at least one on).
- Limits: no learnable weights, only binary inputs, and it **cannot output "more inputs on → output off"** (it is monotonic). If a PYQ needs that behaviour, the answer is "impossible for any θ".

### 2.3 Perceptron (math version)
```
z = w1*x1 + w2*x2 + ... + wn*xn + b      (written z = w·x + b)
output = f(z)
```
- **w (weights):** how important each input is. Learned.
- **b (bias):** shifts the decision point. Bias b = −θ (the old threshold, moved to the other side).
- **f:** the activation function (Part 3).
- With a step function, this is the classic perceptron. With sigmoid, it's a "smooth" neuron.

### 2.4 Linear separability and XOR
- A single neuron draws **one straight line** (in 2D) to split the classes. It can solve a problem only if one line can separate the classes: **linearly separable**.
- AND, OR: separable. **XOR: not separable** (the two "1" points are on opposite diagonal corners; no single line splits them from the "0" points).
- XOR fix: add a **hidden layer** or use a **nonlinear** transformation (e.g. f(z) = z² can represent XOR).
- Trap: a "sandwiched middle band" pattern (e.g. outputs −1, +1, −1 as x1+x2 goes 0, 1, 2) is also NOT linearly separable.

### 2.5 Universal Approximation Theorem (UAT)
- A network with one hidden layer and enough neurons (and a nonlinear activation) **can approximate any continuous function** as closely as you like.
- What it does NOT say: how many neurons you need, that training will find the answer, or how much data you need. It is an **existence** statement only. In MCQs "none of these" is often the right answer for options claiming guarantees.

---

## Part 3. Activation functions

**Why we need them:** without an activation, stacking layers gives just another linear function (a linear layer of a linear layer is linear). Nonlinear activations are what make depth useful.

**Proof that linear layers collapse into one:** layer 1 gives `h = W1·x + b1`. Layer 2 gives `y = W2·h + b2 = W2·(W1·x + b1) + b2 = (W2·W1)·x + (W2·b1 + b2)`. That is still the form `W·x + b`, which is just one linear layer. So 10 linear layers without activations are no more powerful than 1.

| Function | Formula | Output range | Derivative | Max derivative |
|---|---|---|---|---|
| **Sigmoid** | 1 / (1 + e^−z) | (0, 1) | σ(z)(1 − σ(z)) | 0.25 at z = 0 |
| **Tanh** | (e^z − e^−z)/(e^z + e^−z) | (−1, 1) | 1 − tanh²(z) | 1 at z = 0 |
| **ReLU** | max(0, z) | [0, ∞) | 1 if z > 0, else 0 | 1 |
| **Leaky ReLU** | z if z > 0, else 0.01z | (−∞, ∞) | 1 or 0.01 | 1 |

Facts to remember:
- **Sigmoid(0) = 0.5.** If a question says "z = 0 gives output 0.5", it's sigmoid.
- **Vanishing gradient:** sigmoid/tanh flatten out for large |z|, so derivatives become ~0 and early layers stop learning. Sigmoid is worse (max slope only 0.25, so multiplying many of them shrinks fast).
- **Sigmoid is not zero-centered** (outputs always positive). Tanh is zero-centered.
- **ReLU:** cheap to compute and doesn't saturate for z > 0, so it is the default for hidden layers. Problem: **dying ReLU** (if z is always negative, the gradient is 0 forever and the neuron is dead). Leaky ReLU fixes this.
- **Output layer choice:** sigmoid for binary classification, softmax for multi-class, no activation for regression.
- **Softmax:** outputs are in (0, 1) and **sum to 1**, so they can be read as class probabilities. Used for multi-class output.
- **GELU:** a smooth ReLU-like function, used in Transformers (later weeks).
- **Tanh** is mentioned for RNN hidden layers (later weeks).
- **Practical rule:** ReLU for hidden layers, sigmoid for a binary output, softmax for a multi-class output, none for regression.

---

## Part 4. Feedforward network (MLP) architecture

**Feedforward** = data flows one way: input layer → hidden layer(s) → output layer. No loops.

**Dense / fully connected / Linear layer:** every neuron connects to every neuron in the previous layer.

### 4.1 Parameter counting (a favourite PYQ topic)
For a layer with `n_in` inputs and `n_out` neurons:
```
parameters = n_in × n_out  (weights)  +  n_out  (biases)
```
Example: network 2 → 3 → 1
- Hidden layer: 2×3 + 3 = 9
- Output layer: 3×1 + 1 = 4
- **Total = 13**

(Activation functions have no parameters.)

### 4.2 Forward propagation: worked example
Network 2 → 3 → 1, ReLU in the hidden layer, no activation at output.
- x = [1, 2]
- W1 (2×3) = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], b1 = [0, 0, 0]
- Hidden z = x·W1 + b1 = [1×0.1 + 2×0.4, 1×0.2 + 2×0.5, 1×0.3 + 2×0.6] = [0.9, 1.2, 1.5]
- After ReLU: [0.9, 1.2, 1.5] (all positive, unchanged)
- W2 (3×1) = [1, −1, 0.5], b2 = 0.1
- Output = 0.9×1 + 1.2×(−1) + 1.5×0.5 + 0.1 = 0.9 − 1.2 + 0.75 + 0.1 = **0.55**

### 4.3 Shape rule
Matrix multiplication (a × b) @ (b × c) → (a × c). The inner numbers must match. With a batch of N samples: input (N × n_in) → output (N × n_out).

### 4.4 Architecture notation
`784 → 256 → 128 → 10` means 784 inputs, two hidden layers (256 and 128 neurons) and 10 outputs. For example, 784 could be a 28×28 image flattened, and 10 could be the digits 0-9.

Parameters for this network:
- Layer 1: 784×256 + 256 = 200,960
- Layer 2: 256×128 + 128 = 32,896
- Layer 3: 128×10 + 10 = 1,290
- **Total = 235,146**

### 4.5 Hands-on: XOR with a hidden layer
A single neuron cannot solve XOR (Part 2.4), but a network with one hidden layer can. Here is a solution by hand, using a step activation (output 1 if the sum is above 0, else 0):
- Hidden neuron h1 = step(x1 + x2 − 0.5) → acts like **OR**
- Hidden neuron h2 = step(x1 + x2 − 1.5) → acts like **AND**
- Output = step(h1 − h2 − 0.5) → "OR but not AND" = **XOR**

| x1 | x2 | h1 (OR) | h2 (AND) | h1 − h2 | Output |
|---|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 0 | 0 |
| 0 | 1 | 1 | 0 | 1 | 1 |
| 1 | 0 | 1 | 0 | 1 | 1 |
| 1 | 1 | 1 | 1 | 0 | 0 |

The hidden layer transforms the points into a new space where one line can separate the classes. The same thing in PyTorch (standard code, not copied from the lecture):
```python
X = torch.tensor([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
y = torch.tensor([[0.],[1.],[1.],[0.]])
model = nn.Sequential(nn.Linear(2, 4), nn.ReLU(), nn.Linear(4, 1), nn.Sigmoid())
loss_fn = nn.BCELoss()                      # binary cross-entropy for a sigmoid output
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
for epoch in range(1000):
    loss = loss_fn(model(X), y)
    optimizer.zero_grad(); loss.backward(); optimizer.step()
```
Training can occasionally get stuck with a bad random start. Re-running with a different seed usually fixes it.

---

## Part 5. Training: loss, gradient descent, backpropagation

### 5.1 Loss functions (how wrong are we?)
- **Squared error (MSE):** L = (ŷ − y)². Squaring stops positive and negative errors from cancelling and punishes big errors more. Used for regression.
- **Binary cross-entropy (BCE):** L = −[y·log(ŷ) + (1−y)·log(1−ŷ)]. Used for binary classification with sigmoid output. Punishes *confident wrong* answers very heavily.

### 5.2 Gradient descent
Picture the loss as a bowl. We want the bottom.
```
w_new = w_old − η × (dL/dw)
```
- **η (learning rate):** step size. Too big: overshoots or diverges. Too small: very slow.
- Stop when the gradient ≈ 0.
- **Epoch** = one full pass through the training data.

Worked example (one weight, x = 2, y = 10, η = 0.1, ŷ = w·x, L = (wx − y)²):
- dL/dw = 2(wx − y)·x
- w = 3 → dL/dw = 2(6 − 10)(2) = −16 → w = 3 + 1.6 = 4.6
- w = 4.6 → dL/dw = 2(9.2 − 10)(2) = −3.2 → w = 4.92
- It converges to w = 5 (since 5 × 2 = 10).

### 5.3 Backpropagation = the chain rule
Backprop computes dL/dw for every weight by multiplying local derivatives backwards from the loss. For one neuron:
```
dL/dw = (dL/dŷ) × (dŷ/dz) × (dz/dw)
```
- **δ = dL/dz** (the "bridge value") is what you pass backwards.
- **Sigmoid + BCE shortcut:** δ = ŷ − y (the ŷ(1−ŷ) terms cancel). This works ONLY for the sigmoid + BCE pairing.
- Then dL/dw_i = δ × x_i and dL/db = δ.
- Comparison: δ for BCE is at least 4× larger than δ for MSE with sigmoid, because ŷ(1−ŷ) ≤ 0.25.

Full worked example (sigmoid + BCE): x1 = 2, x2 = 1, w1 = −0.3, w2 = 0.4, b = 0, y = 0, η = 0.2
- z = −0.6 + 0.4 = −0.2, ŷ = σ(−0.2) ≈ 0.45
- δ = ŷ − y ≈ 0.45
- dL/dw1 = 0.45 × 2 = 0.90 → w1 = −0.3 − 0.2×0.90 ≈ **−0.48**
- dL/dw2 = 0.45 × 1 = 0.45 → w2 = 0.4 − 0.2×0.45 ≈ **0.31**

**Practice problem (do this yourself):** x1 = 1, x2 = 3, w1 = 0.2, w2 = −0.1, b = 0, y = 1, sigmoid + BCE, η = 0.1. Find the new w1 and w2.

---

## Part 6. PyTorch tensors

**Tensor** = a multi-dimensional array (like NumPy's array, but it can run on a GPU and track gradients).

| Name | Dimensions | Example |
|---|---|---|
| Scalar | 0-D | `torch.tensor(7)` |
| Vector | 1-D | `torch.tensor([1, 2, 3])` |
| Matrix | 2-D | `torch.tensor([[1, 2], [3, 4]])` |
| Tensor | 3-D+ | images: (channels, height, width) |

### 6.1 Creating tensors
```python
torch.tensor([1., 2.])          # from data
torch.zeros(2, 3)               # all 0
torch.ones(2, 3)                # all 1
torch.rand(2, 3)                # uniform random in [0, 1)
torch.randn(2, 3)               # normal random (mean 0, std 1)
torch.arange(0, 10, 2)          # 0, 2, 4, 6, 8
torch.zeros_like(x)             # same shape as x
```

### 6.2 Key attributes
- `x.shape` (or `x.size()`): the dimensions
- `x.dtype`: data type. **Default float is float32; integer lists become int64.**
- `x.device`: `cpu` or `cuda`
- `x.ndim`: number of dimensions
- `x.numel()`: total number of elements (e.g. shape (2,3,4) → 24)

### 6.3 Reshaping (very common in exams)
| Operation | What it does |
|---|---|
| `x.view(a, b)` | Reshape without copying. **Needs contiguous memory**, or it errors. |
| `x.reshape(a, b)` | Reshape; works on any tensor (copies if needed). |
| `x.permute(2, 0, 1)` | Reorder dimensions (e.g. H,W,C → C,H,W). Result is usually non-contiguous. |
| `x.T` / `x.transpose(0, 1)` | Swap two dimensions. |
| `x.squeeze()` | Remove dimensions of size 1. |
| `x.unsqueeze(d)` | Add a size-1 dimension at position d. |
| `torch.stack([a, b])` | Join tensors along a **NEW** dimension. |
| `torch.cat([a, b], dim=0)` | Join along an **EXISTING** dimension. |
| `-1` in a shape | "Figure this one out for me": `x.view(-1, 4)` |

The total number of elements must stay the same: a (2, 6) tensor can become (3, 4) or (12,), but not (5, 3).

### 6.4 Indexing and slicing
```python
x[0]         # first row
x[:, 1]      # column 1, all rows
x[0, 1]      # one element
x[1:3]       # rows 1 and 2 (end is excluded)
x[x > 0]     # boolean mask: elements greater than 0
```

### 6.5 Math operations
- Element-wise: `a + b`, `a * b` (same shape, or broadcastable).
- Matrix multiply: `a @ b` or `torch.matmul(a, b)`.
- **`a * b` is NOT matrix multiplication.**
- Broadcasting: PyTorch stretches size-1 dimensions to match (e.g. (3, 1) + (1, 4) → (3, 4)). Two shapes are compatible when, going from the last dimension backwards, each pair is **equal or one of them is 1**.

### 6.5b Aggregation (sum, mean, max...)
```python
x = torch.tensor([[1., 2.], [3., 4.]])
x.sum()          # 10.  (all elements, a single number)
x.sum(dim=0)     # tensor([4., 6.])  collapse the ROWS: one result per column
x.sum(dim=1)     # tensor([3., 7.])  collapse the COLUMNS: one result per row
x.mean(), x.max(), x.min()
x.argmax()       # the INDEX of the biggest element (not the value)
```
**Rule for `dim`:** the dimension you name is the one that disappears. In a (2, 2) tensor, `dim=0` removes the row dimension, so the answer has one value per column. `dim=1` removes the column dimension, so the answer has one value per row.

**Exam example (a real Quiz PYQ):**
```python
x = torch.tensor([[1., 2.], [3., 4.]])
y = x.view(4)       # y is a flat view of the SAME memory as x
y[1] = 10           # changes x[0][1] too, so x = [[1, 10], [3, 4]]
z = x.sum(dim=1)    # row sums: [1+10, 3+4]
print(z)            # tensor([11., 7.])
```
The lesson: `view` shares memory with the original, so changing `y` changes `x`.

### 6.6 NumPy, reproducibility, GPU
- `torch.from_numpy(arr)` and `x.numpy()` **share memory** with the original (changing one changes the other, on CPU). `torch.tensor(arr)` makes a copy.
- NumPy defaults to float64, PyTorch to float32. Mixing them is a common error source.
- **Reproducibility:** `torch.manual_seed(42)` before creating random tensors or models gives the same numbers every run.
- **GPU:**
```python
device = "cuda" if torch.cuda.is_available() else "cpu"
x = x.to(device)
model = model.to(device)
```
  The model and data must be on the **same device**.

### 6.7 Autograd (automatic gradients)
PyTorch can compute gradients for you. This is how `loss.backward()` works.
```python
x = torch.tensor([-2.0, 0.0, 3.0], requires_grad=True)   # "track gradients for this tensor"
y = torch.relu(x)       # [0, 0, 3]
z = y.sum()             # 3
z.backward()            # backpropagation: computes dz/dx
print(x.grad)           # tensor([0., 0., 1.])
```
- `requires_grad=True` tells PyTorch to record every operation so it can differentiate later.
- `.backward()` computes the gradient of a **single number** (the loss) with respect to every tracked tensor. The result is stored in `.grad`.
- Why `[0, 0, 1]`? ReLU's derivative is 1 where x > 0 and 0 otherwise. In PyTorch it is **0 at x = 0**. `sum` passes a gradient of 1 to every element. So the gradients are 0 (x = −2), 0 (x = 0) and 1 (x = 3).
- **Gradients accumulate:** calling `backward()` twice adds the new gradients to `.grad`. That is why the training loop calls `zero_grad()`.
- `x.detach()` returns a copy of the tensor that is cut off from the graph. `with torch.no_grad():` turns tracking off for a block of code.
- Model parameters (`nn.Linear` weights and biases) already have `requires_grad=True`.

### 6.8 Environment setup
- **Easiest option: Google Colab.** Go to colab.research.google.com. It is free, PyTorch is already installed, and it offers a free GPU (Runtime → Change runtime type → GPU).
- **Local install:** in a terminal run `pip install torch`. The package is called `torch`, not `pytorch`. In a notebook cell use `%pip install torch` and restart the kernel afterwards.
- **Verify:**
```python
import torch
print(torch.__version__)
print(torch.cuda.is_available())   # True if a GPU can be used
```

---

## Part 7. Linear regression in PyTorch

**Goal:** learn y = w·x + b from data.

### 7.1 The workflow (same for every PyTorch model)
1. Get/create data and split into train and test
2. Define the model
3. Pick a loss function and an optimizer
4. Training loop
5. Evaluate
6. Save the model

### 7.2 Data creation
```python
weight, bias = 0.7, 0.3
X = torch.arange(0, 1, 0.02).unsqueeze(1)   # shape (50, 1)
y = weight * X + bias
split = int(0.8 * len(X))
X_train, y_train = X[:split], y[:split]
X_test,  y_test  = X[split:], y[split:]
```
`unsqueeze(1)` makes X shape (50, 1): 50 samples, 1 feature, which is what `nn.Linear` expects.

**Visualization (always do this before modeling):** plot the data to see whether it looks linear, whether there are outliers, and what the spread is. Never test on training data, because you could not tell whether the model learned or just memorized.
```python
import matplotlib.pyplot as plt
plt.scatter(X_train, y_train, c="b", label="Train")
plt.scatter(X_test, y_test, c="g", label="Test")
plt.legend(); plt.show()
```
Later you add the model's predictions to the same plot, in a third colour, to see how close they are to the test data.

### 7.3 Model definition
```python
class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(in_features=1, out_features=1)
    def forward(self, x):
        return self.linear(x)
```
- `nn.Linear(in, out)` has a weight of shape **(out, in)** and a bias of shape **(out)**. Parameters = in × out + out.
- `forward()` defines the computation. You call `model(x)`, not `model.forward(x)`.

### 7.4 Loss and optimizer
```python
loss_fn = nn.MSELoss()         # or nn.L1Loss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
```
**Custom loss function (written by hand):**
```python
def my_mse(y_pred, y_true):
    return ((y_pred - y_true) ** 2).mean()
```
It works with autograd as long as you use PyTorch operations.

### 7.5 The training loop (memorize the order)
```python
for epoch in range(epochs):
    model.train()
    y_pred = model(X_train)             # 1. forward pass
    loss = loss_fn(y_pred, y_train)     # 2. compute loss
    optimizer.zero_grad()               # 3. clear old gradients
    loss.backward()                     # 4. backpropagation
    optimizer.step()                    # 5. update weights
```
- **Why `zero_grad()`?** PyTorch *accumulates* gradients. Without clearing, each step uses the sum of all past gradients, which is wrong.
- Evaluation:
```python
model.eval()
with torch.inference_mode():     # or torch.no_grad()
    test_pred = model(X_test)
    test_loss = loss_fn(test_pred, y_test)
```
`no_grad` / `inference_mode` turn off gradient tracking, which saves memory and time.

### 7.6 Model serialization (saving and loading)
```python
torch.save(model.state_dict(), "model.pth")        # save (recommended)

new_model = LinearModel()                          # create the same architecture first
new_model.load_state_dict(torch.load("model.pth")) # then load the weights
```
- `state_dict` = a dictionary of the parameters (for `nn.Linear(1,1)`: keys `weight` and `bias`).
- Saving only the `state_dict` is preferred over saving the whole model object, which depends on the exact class and file layout.
- You must build the model with the same architecture before loading weights into it.

---

## Part 8. Common errors (and what causes them)

| Error message (roughly) | Cause | Fix |
|---|---|---|
| `mat1 and mat2 shapes cannot be multiplied` | Inner dimensions don't match (e.g. data has 3 features, layer expects 2) | Check `.shape` and `in_features` |
| `Expected all tensors to be on the same device` | Model on GPU, data on CPU (or the reverse) | `.to(device)` on both |
| `expected scalar type Float but found Double` | float64 data (often from NumPy) into a float32 model | `x = x.float()` |
| `view size is not compatible...` | `.view()` on a non-contiguous tensor | Use `.reshape()` or `.contiguous().view()` |
| `shape '[a, b]' is invalid for input of size N` | The element count doesn't match the new shape | Make a × b = N |
| Loss not decreasing / weird results | Forgot `zero_grad()`, wrong learning rate, wrong mode | Check the loop order |
| Can't call `.numpy()` on a tensor that requires grad | The tensor is part of the graph | `x.detach().numpy()` |

---

## Part 9. Fundamental Tricks in Deep Learning

> Source note: this part comes from an unofficial study page (lecture 1.18, "DL Tricks"), not from the portal video itself. The page lists three tricks. Its code blocks did not come through, so the code below is standard PyTorch that I wrote. After you watch the portal video, check that it covers the same three tricks.

There are three practical tricks that make training work: **weight initialization, feature normalization and batch processing.**

### 9.1 Weight initialization
**Problem:** weights start as random numbers, and the starting size matters a lot.
- **Too large:** each layer multiplies the signal by big numbers, so values blow up as they pass through layers (**exploding signals**). Sigmoid/tanh also get pushed into their flat regions.
- **Too small:** the signal shrinks layer after layer until it is almost 0 (**vanishing signals**), and the early layers learn almost nothing.
- **All zeros is also bad:** every neuron in a layer computes the same thing and gets the same gradient, so they never become different (the symmetry problem). Biases can start at 0, but weights should be random.

**The fix: scale the random weights by the layer size.**
| Method | Use with | Rule (n_in = number of inputs to the layer) |
|---|---|---|
| **He (Kaiming) init** | **ReLU** (the default for hidden layers) | std = sqrt(2 / n_in) |
| **Xavier (Glorot) init** | sigmoid / tanh | std = sqrt(2 / (n_in + n_out)) |

```python
nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")   # He init
nn.init.zeros_(layer.bias)                                   # biases start at 0
```
**Why the "2" in He init?** ReLU sets about half of the values to 0, so the factor of 2 makes up for the lost half and keeps the signal size steady from layer to layer.

### 9.2 Feature normalization
**Problem:** features can have very different scales (e.g. age 0-100 vs salary 0-1,000,000). The loss surface becomes a long, narrow valley, and gradient descent zig-zags and converges slowly. A single learning rate cannot suit both features.

**The fix (standardization / z-score):** for each feature, subtract its mean and divide by its standard deviation, so every feature has **mean 0 and std 1**.
```
x_norm = (x - mean) / std
```
```python
mean = X_train.mean(dim=0)          # one mean per feature (column)
std  = X_train.std(dim=0)
X_train = (X_train - mean) / std
X_test  = (X_test  - mean) / std    # use the TRAIN mean and std on the test data too
```
- **Trap:** compute mean and std from the **training data only**, then reuse those numbers on the test data. Computing them on the test set leaks information.
- Normalization is needed at prediction time too, using the same saved mean and std.
- You will meet the same idea inside the network later as **BatchNorm**.

### 9.3 Batch processing
**Idea:** instead of feeding one sample at a time, feed many samples together as one tensor. GPUs do matrix math on many rows in parallel, so this is much faster.
- Input shape becomes `(batch_size, n_features)`, and the output becomes `(batch_size, n_out)`.
- **Batch gradient descent:** the whole dataset per update. **Stochastic GD (SGD):** one sample per update. **Mini-batch GD:** a chunk (e.g. 32 or 64) per update, which is what people use in practice.
- Mini-batch updates are less exact than full-batch ones but much cheaper, and the extra noise can help the model avoid poor spots.
- **Epoch** = one pass over all the data. With 1,000 samples and batch size 100, one epoch = **10 updates**.
```python
loader = DataLoader(dataset, batch_size=32, shuffle=True)   # shuffle the training data every epoch
for X_batch, y_batch in loader:
    ...  # forward, loss, zero_grad, backward, step
```
- The batch size is the first dimension of a tensor. Mistakes here are a common cause of shape errors.

**Exam angle:** "which init for ReLU?" → He. "Why normalize?" → faster, more stable gradient descent with similar feature scales. "Why mini-batches?" → GPU efficiency plus cheaper updates.

---

## Part 10. Exam traps (from the Quiz 1 PYQ patterns)

1. **Tensor shape questions:** write the shape after every operation. Check the element count first.
2. **`view` vs `reshape`:** `view` fails after `permute`/transpose unless you call `.contiguous()`.
3. **`stack` vs `cat`:** stack makes a new dimension, cat does not. Stacking three (2,3) tensors gives (3,2,3); cat along dim 0 gives (6,3).
4. **Parameter counting:** weights + biases, per layer. Don't count activations. Check whether the layer has `bias=False`.
5. **Forward pass numerics:** do it layer by layer and apply the activation before the next layer.
6. **Backprop numerics:** compute ŷ first, then δ, then the gradients, then the update. Watch the sign: w_new = w_old − η × gradient.
7. **MP neuron:** monotonic, so "more inputs on → output off" is impossible.
8. **UAT:** only existence. Any option promising training success or exact neuron counts is wrong.
9. **Sigmoid:** max derivative 0.25, output 0.5 at z = 0.
10. **Training loop order:** forward → loss → zero_grad → backward → step. A swapped order is a typical "find the bug" question.

---

## Part 11. Quick formula sheet

```
Neuron:            z = w·x + b,  a = f(z)
Sigmoid:           σ(z) = 1/(1+e^-z),  σ' = σ(1-σ),  max 0.25
Tanh:              tanh' = 1 - tanh²
ReLU:              max(0, z)
MSE:               (ŷ - y)²
BCE:               -[y log ŷ + (1-y) log(1-ŷ)]
Gradient descent:  w ← w - η · dL/dw
Sigmoid+BCE:       δ = ŷ - y
Layer params:      n_in × n_out + n_out
```

---

