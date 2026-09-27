# 🔥 PyTorch Notes

Topic-by-topic notes from learning PyTorch from zero. New topics are added at the bottom.

---

# 📘 Chapter 0 - PyTorch Fundamentals

---

## 🤖 Topic 0 - What is Machine Learning and Deep Learning?

> **Machine learning (ML)** turns data into numbers and finds patterns in them.
> **Deep learning (DL)** is machine learning that uses neural networks.

### Traditional programming vs machine learning

| Approach                | You give                 | You get             |
|-------------------------|--------------------------|---------------------|
| Traditional programming | Inputs + rules you write | Outputs             |
| Machine learning        | Inputs + correct outputs | The rules (learned) |

**Example:** given only `0 -> 32`, `10 -> 50`, `20 -> 68`, `30 -> 86`, the computer learns the rule `F = C x 1.8 + 32` and can answer new inputs (`25 -> 77`).

### Everything becomes numbers

| Data  | Becomes                                   |
|-------|-------------------------------------------|
| Image | Pixel values (0 to 255)                   |
| Text  | A number (ID) for each word or word-piece |
| Audio | Sound values over time                    |

In PyTorch, these numbers are stored in **tensors**.

---

## 🎯 Topic 1 - Why Use Machine Learning?

> Use ML when a problem is **too complex to write all the rules by hand**.

A 256 x 256 colour photo = 256 x 256 x 3 = **196,608 numbers**. No one can hand-write rules that turn those into "cat" or "not cat", so ML learns the rules from labelled photos.

**ML / DL is good for:**

1. Problems with a very long list of rules (driving, speech)
2. Situations that keep changing (spam filters can be retrained)
3. Finding patterns in huge data (fraud in millions of transactions)

---

## 📏 Topic 2 - The Number One Rule of ML

> If a simple rule-based system can solve the problem, **don't use machine learning**.
> Example: 18% GST is just `price * 0.18`.

**Deep learning is usually NOT a good choice when:**

1. You need to **explain** the decision (learned weights aren't human-readable)
2. **Errors are unacceptable** (99% accurate = 1 wrong in every 100)
3. A **simple rule** already works
4. You have **very little data** (workaround: transfer learning)

---

## ⚖️ Topic 3 - Machine Learning vs Deep Learning

|                    | Traditional ML              | Deep learning                      |
|--------------------|-----------------------------|------------------------------------|
| Best data          | Structured (tables)         | Unstructured (images, text, audio) |
| Features chosen by | A human                     | The network itself                 |
| Typical algorithms | Random forest, XGBoost, SVM | Neural networks (CNN, transformer) |
| Data needed        | Can work with less          | Usually a lot                      |

**Features** = the useful inputs, e.g. "area" and "bedrooms" for house prices. In deep learning, the network finds its own features: edges -> shapes -> "ear" -> "cat".

---

## 🧠 Topic 4 - Anatomy of a Neural Network

### A single neuron

```text
output = input x weight + bias
```

- **Weight:** the number the input is multiplied by
- **Bias:** the number added afterwards
- Both start **random**; **training** adjusts them until the outputs are correct

| Weight | Bias | Input | Output                                       |
|--------|------|-------|----------------------------------------------|
| 2      | 3    | 5     | 5 x 2 + 3 = **13**                           |
| 1.8    | 32   | 20    | **68** (the Celsius converter is one neuron) |

### Layers

| Layer         | What it does                                         |
|---------------|------------------------------------------------------|
| Input layer   | The data as numbers (e.g. 196,608 pixel values)      |
| Hidden layers | Multiply by weights, add up, add a bias              |
| Output layer  | Numbers we interpret (e.g. `[0.92, 0.08]` = 92% cat) |

### ReLU (non-linearity)

ReLU turns **negative numbers into 0** and keeps positive numbers unchanged.

```text
[-1, 4, 0, -7]  ->  ReLU  ->  [0, 4, 0, 0]
```

Without ReLU, a network can only learn straight-line patterns.

**Remember**

- A neural network = many "multiply, add, ReLU" steps stacked in layers.
- **Deep** = many hidden layers.

---

## 📚 Topic 5 - Learning Paradigms

| Paradigm          | What the model gets             | Example                         |
|-------------------|---------------------------------|---------------------------------|
| Supervised        | Data + labels                   | Emails tagged spam / not spam   |
| Unsupervised      | Data only                       | Groups similar photos by itself |
| Self-supervised   | Data only; makes its own labels | Guess the hidden next word      |
| Transfer learning | Pre-trained model + small data  | Fine-tune on 200 photos         |
| Reinforcement     | Actions + rewards / penalties   | Learning to play a game         |

> This course focuses on **supervised learning** and **transfer learning**.

---

## 🌍 Topic 6 - What Deep Learning Is Used For

**Uses:** recommendations, translation, speech recognition, photo tagging, self-driving, spam filters.

| Problem type     | Output                    | Example                     |
|------------------|---------------------------|-----------------------------|
| Classification   | Pick a category           | Spam / not spam             |
| Regression       | Predict a number          | House price, temperature    |
| Seq2seq          | Sequence in, sequence out | Translation, speech-to-text |
| Object detection | Where things are          | A box around each car       |

---

## 🔥 Topic 7 - What is PyTorch?

> **PyTorch** is a free, open-source Python library for building and training neural networks. It was created at Meta. In code it's called `torch`.

**What it gives you:**

1. **Fast maths on a GPU** (NVIDIA GPUs through **CUDA**)
2. **Autograd:** automatically works out how to adjust every weight during training
3. **Ready-made building blocks:** layers, ReLU, pre-trained models

### CPU vs GPU

|         | CPU                              | GPU                                |
|---------|----------------------------------|------------------------------------|
| Workers | Few (e.g. 8), very powerful      | Thousands, simple                  |
| Good at | Complex tasks, one after another | Many simple tasks at the same time |

A neural network is millions of simple multiply-and-adds, so a **GPU trains much faster**. Google Colab gives a free GPU.

---

## 🥚 Topic 8 - What Are Tensors?

> A **tensor** is a container of numbers arranged in dimensions. PyTorch stores **all** data (inputs, weights, outputs) as tensors.

| Name   | Egg analogy      | `ndim` | Example shape |
|--------|------------------|--------|---------------|
| Scalar | One egg          | 0      | `[]`          |
| Vector | A row of eggs    | 1      | `[3]`         |
| Matrix | An egg tray      | 2      | `[3, 2]`      |
| Tensor | A stack of trays | 3+     | `[2, 3, 4]`   |

- **`ndim`** = how many directions you count in
- **`shape`** = how many items in each direction
- A 224 x 224 colour photo = shape `[3, 224, 224]` = 150,528 numbers

---

## 🗺️ Topics 9 to 11 - Workflow, Approach and Resources

### The PyTorch workflow (used in every chapter)

1. Get data ready (as tensors)
2. Build or pick a model
3. Train the model
4. Evaluate it
5. Improve through experiments
6. Save and reload it

**Remember**

- Code along, run the code when in doubt, experiment, visualise, and treat errors as part of learning.

| Resource | Link                        |
|----------|-----------------------------|
| Book     | https://www.learnpytorch.io |
| Docs     | https://pytorch.org/docs    |
| Forum    | https://discuss.pytorch.org |

---

## ⚙️ Topic 12 - Getting Set Up

|         | Local (VS Code / Jupyter)   | Google Colab                   |
|---------|-----------------------------|--------------------------------|
| Runs on | Your laptop                 | Google's servers               |
| Setup   | Install PyTorch yourself    | Nothing to install             |
| GPU     | Only if your laptop has one | Free GPU                       |
| Session | Stays until you close it    | Resets when idle; re-run cells |

```python
%pip install torch        # install PyTorch from a notebook cell
import torch              # load the PyTorch library
print(torch.__version__)  # print the version to confirm it works
```

**Watch out**

- The package name is `torch`, **not** `pytorch`
- Run `%pip` inside a notebook cell, not in the terminal
- Restart the kernel after installing

---

## 📦 Topic 13 - Introduction to Tensors

### Creating tensors

```python
scalar = torch.tensor(7)             # create a tensor holding a single number
vector = torch.tensor([20, 25, 30])  # create a tensor holding a row of numbers

# create a tensor with 3 rows and 2 columns
MATRIX = torch.tensor([[1, 2],
                       [3, 4],
                       [5, 6]])

# create a 3-D tensor: 2 trays, each with 2 rows and 3 columns
TENSOR = torch.tensor([[[1, 2, 3],
                        [4, 5, 6]],
                       [[7, 8, 9],
                        [10, 11, 12]]])
```

### Looking inside a tensor

```python
print(MATRIX.ndim)    # number of dimensions
print(MATRIX.shape)   # size along each dimension (no brackets)
print(MATRIX.size())  # same as .shape, written with brackets
print(MATRIX[0])      # select the first row (counting starts at 0)
print(TENSOR[0])      # select the first tray
print(scalar.item())  # take the number out as a plain Python number
```

| Tensor | `ndim` | `shape`                 |
|--------|--------|-------------------------|
| scalar | 0      | `torch.Size([])`        |
| vector | 1      | `torch.Size([3])`       |
| MATRIX | 2      | `torch.Size([3, 2])`    |
| TENSOR | 3      | `torch.Size([2, 2, 3])` |

**Remember**

- Count the opening brackets at the start -> that's `ndim`
- Read shape from the outside in: `[2, 2, 3]` = 2 trays, 2 rows, 3 columns
- Total numbers = multiply the shape values (2 x 2 x 3 = 12)
- Selecting one item removes one dimension (tray -> row -> number)

**Watch out**

- `x.shape()` fails; use `x.shape` or `x.size()`
- `.item()` only works on a tensor with exactly one number
- `MATRIX[3]` on a 3-row matrix fails; rows are counted 0, 1, 2

---

## 🏗️ Topic 14 - Creating Tensors

### Random, zeros, ones

```python
random_tensor = torch.rand(3, 4)        # create a 3x4 tensor of random numbers (0 to 1)
image = torch.rand(size=(224, 224, 3))  # create a random image: height, width, channels
zeros = torch.zeros(3, 4)               # create a 3x4 tensor filled with zeros
ones = torch.ones(3, 4)                 # create a 3x4 tensor filled with ones
```

- You give the **shape**; PyTorch fills in the numbers
- `torch.rand` gives new numbers on every run
- `size=(2, 3)` is the same as `(2, 3)`; it only labels the input
- Zeros and ones print with a dot (`0.`, `1.`) because they're decimals

### Image shapes

| Part            | Meaning                                        |
|-----------------|------------------------------------------------|
| Image shape     | `[height, width, colour channels]`             |
| Colour pixel    | 3 numbers (Red, Green, Blue)                   |
| Grayscale pixel | 1 number (brightness: 0 = black, 1 = white)    |
| PyTorch models  | Usually expect channels first: `[3, 224, 224]` |

### Element-wise multiplication

```python
prices = torch.tensor([10, 20, 30])  # price of each of the 3 items
quantity = torch.tensor([2, 1, 3])   # how many of each item we buy
print(prices * quantity)             # multiply each price by its matching quantity
```

Result: `tensor([20, 20, 90])`

> `*` multiplies **matching positions**. Both tensors need the **same shape**, or one side must be a **single number**.

### Ranges and copies

```python
torch.arange(1, 11)                    # numbers from 1 up to (not including) 11
torch.arange(start=0, end=10, step=2)  # even numbers from 0 up to (not including) 10
torch.arange(10, 0, -2)                # count down from 10 in steps of 2 (stops before 0)
torch.zeros_like(MATRIX)               # zeros with the same shape and dtype as MATRIX
torch.ones_like(MATRIX)                # ones with the same shape and dtype as MATRIX
```

**Watch out**

- `arange` does **not** include the end number: use `torch.arange(1, 31)` for 1 to 30
- A negative step needs start bigger than end
- Use `torch.arange`, not the outdated `torch.range`
- `torch.random(...)` is wrong; the function is `torch.rand(...)`

---

## 🔢 Topic 17 - Tensor Datatypes

### int vs float

| Type    | Holds                       | Example  |
|---------|-----------------------------|----------|
| `int`   | Whole numbers (counting)    | 3 people |
| `float` | Decimal numbers (measuring) | 3.75 kg  |

```python
whole = torch.tensor([1, 2, 3])     # create a tensor of whole numbers
decimal = torch.tensor([1.5, 2.5])  # create a tensor of decimal numbers
print(whole.dtype, decimal.dtype)   # show both datatypes
```

### Default datatypes

| Created from                      | Default datatype     |
|-----------------------------------|----------------------|
| Whole numbers, `arange`           | `torch.int64`        |
| Decimals, `zeros`, `ones`, `rand` | `torch.float32`      |
| `zeros_like`, `ones_like`         | Same as the original |

> One tensor = one datatype. **One decimal makes the whole tensor float.**

### Choosing and changing a datatype

```python
half = torch.tensor([3.14159265], dtype=torch.float16)  # store as a 16-bit decimal
as_int = decimal.type(torch.int64)                      # make a whole-number copy
decimal = decimal.type(torch.float16)                   # store the copy back to change the original
```

| Datatype  | Bytes | Precision       | Speed and memory               |
|-----------|-------|-----------------|--------------------------------|
| `float16` | 2     | 3 to 4 digits   | Fastest, least memory          |
| `float32` | 4     | About 7 digits  | The default, the usual balance |
| `float64` | 8     | 15 to 16 digits | Slowest, most memory           |

**Watch out**

- `.type()` / `.to()` return a **new** tensor; store it back: `x = x.type(...)`
- Float to int **cuts off** decimals, it doesn't round (1.9 -> 1)
- Never name a variable `range`, `list` or `sum`

**Remember**

- The 3 most common PyTorch errors: **wrong datatype**, **wrong shape**, **wrong device**.

---

## 🏷️ Topic 18 - Tensor Attributes

```python
x = torch.rand(32, 32, 3)        # create a random image-shaped tensor
print(f"Datatype: {x.dtype}")    # kind of numbers stored
print(f"Shape: {x.shape}")       # size along each dimension
print(f"Device: {x.device}")     # where the tensor lives (cpu by default)
print(f"Dimensions: {x.ndim}")   # number of dimensions
```

| Attribute | Answers               | Example                   |
|-----------|-----------------------|---------------------------|
| `.dtype`  | What kind of numbers? | `torch.float32`           |
| `.shape`  | How big?              | `torch.Size([32, 32, 3])` |
| `.device` | Where is it stored?   | `cpu`                     |

**Remember**

- When code breaks, check `.dtype`, `.shape` and `.device` first.

**Watch out**

- `torch.tensor` takes the **actual numbers**; `zeros`, `ones` and `rand` take a **shape**.
- `torch.tensor(32, 32, 3)` fails; use `torch.rand(32, 32, 3)`.

---