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

## Topic 19 - Manipulating Tensors

Basic maths on tensors: add, subtract, multiply and divide. With a single number, the operation is applied to every element.

```python
prices = torch.tensor([100, 200, 300])  # create a tensor of item prices
prices + 10                             # add 10 to every price
prices - 10                             # subtract 10 from every price
prices * 2                              # multiply every price by 2
prices / 2                              # divide every price by 2
prices = prices + 10                    # store the result back to keep the change
```

| Operation     | Symbol | Function           | Result for `[100, 200, 300]` |
|---------------|--------|--------------------|------------------------------|
| Add 10        | `+`    | `torch.add(x, 10)` | `[110, 210, 310]`            |
| Subtract 10   | `-`    | `torch.sub(x, 10)` | `[90, 190, 290]`             |
| Multiply by 2 | `*`    | `torch.mul(x, 2)`  | `[200, 400, 600]`            |
| Divide by 2   | `/`    | `torch.div(x, 2)`  | `[50., 100., 150.]`          |

**Remember**

- Each operation returns a new tensor; the original is unchanged
- Store it back to keep the change: `x = x + 10`
- Tensor with tensor works position by position (same shape needed)

**Watch out**

- Division always gives float results, even for whole-number tensors
- The library is `torch`: write `torch.add(...)`, not `tensor.add(...)`

---

## Topic 20 - Matrix Multiplication

Matrix multiplication multiplies matching pairs and then adds them up. It is the core operation of deep learning.

### Element-wise vs matrix multiplication

```python
prices = torch.tensor([10, 20, 30])  # price of each item
quantity = torch.tensor([2, 1, 3])   # how many of each item we buy
prices * quantity                    # element-wise: cost of each item
prices @ quantity                    # matrix multiplication: total bill
torch.matmul(prices, quantity)       # same as prices @ quantity
```

| Operation             | Symbol                | Result for the example |
|-----------------------|-----------------------|------------------------|
| Element-wise          | `*`                   | `[20, 20, 90]`         |
| Matrix multiplication | `@` or `torch.matmul` | `130` (20 + 20 + 90)   |

For two vectors this is called the **dot product**. A neuron's weighted sum (inputs x weights, added up) is a dot product.

### Matrix x matrix

Result at [row i, column j] = row i of the first matrix dot column j of the second.

```python
A = torch.tensor([[1, 2], [3, 4]])  # create a 2x2 matrix
B = torch.tensor([[5, 6], [7, 8]])  # create another 2x2 matrix
A @ B                               # rows of A dot columns of B
```

|                     | Column 0 of B `[5, 7]` | Column 1 of B `[6, 8]` |
|---------------------|------------------------|------------------------|
| Row 0 of A `[1, 2]` | 1x5 + 2x7 = 19         | 1x6 + 2x8 = 22         |
| Row 1 of A `[3, 4]` | 3x5 + 4x7 = 43         | 3x6 + 4x8 = 50         |

### The 2 shape rules

1. The inner dimensions must match: `(a x b) @ (b x c)`
2. The result has the outer dimensions: `(a x c)`

| Shapes        | Inner dimensions | Result |
|---------------|------------------|--------|
| (2x3) @ (3x2) | 3 = 3            | 2x2    |
| (3x2) @ (2x3) | 2 = 2            | 3x3    |
| (1x6) @ (6x1) | 6 = 6            | 1x1    |
| (6x1) @ (1x6) | 1 = 1            | 6x6    |
| (3x2) @ (3x2) | 2 and 3 differ   | Error  |

### Fixing shape errors with transpose

Transpose swaps rows and columns, so a (2 x 3) becomes (3 x 2).

```python
t3 = torch.rand(2, 3)  # create a random 2x3 matrix
t4 = torch.rand(2, 3)  # create another random 2x3 matrix
t3 @ t4.T              # (2x3) @ (3x2) gives a 2x2 result
t3.T @ t4              # (3x2) @ (2x3) gives a 3x3 result
```

**Remember**

- `x.T` (or `torch.transpose(x, 0, 1)`) swaps rows and columns; the original is unchanged
- A neural network layer computes `inputs @ weights.T + bias`
- With two vectors, PyTorch treats the first as a row and the second as a column
- `torch.matmul` is much faster than a Python for loop

**Watch out**

- `A @ B` is usually not the same as `B @ A`
- For a dot product, both vectors must have the same length
- Shape errors look like: `RuntimeError: mat1 and mat2 shapes cannot be multiplied (3x2 and 3x2)`
- Check the two shapes side by side before every `@`

---

## Extras - Other Names, Integer Types and Timing Code

### Other names for the same thing

Some functions and datatypes have more than one name. They do exactly the same thing.

| Main name            | Other names                     | What it does                           |
|----------------------|---------------------------------|----------------------------------------|
| `torch.matmul(A, B)` | `A @ B`, `torch.mm(A, B)`       | Matrix multiplication (`mm`: 2-D only) |
| `torch.mul(x, y)`    | `torch.multiply(x, y)`, `x * y` | Element-wise multiplication            |
| `torch.float32`      | `torch.float`                   | 32-bit decimals                        |
| `torch.float16`      | `torch.half`                    | 16-bit decimals                        |
| `torch.float64`      | `torch.double`                  | 64-bit decimals                        |
| `torch.int64`        | `torch.long`                    | 64-bit whole numbers                   |

### Integer datatypes

Each integer type can only hold numbers within a certain range.

| dtype         | Range                                      | Memory for 1 million numbers |
|---------------|--------------------------------------------|------------------------------|
| `torch.int8`  | -128 to 127                                | 1 MB                         |
| `torch.uint8` | 0 to 255 (no negatives)                    | 1 MB                         |
| `torch.int16` | about -32 thousand to +32 thousand         | 2 MB                         |
| `torch.int32` | about -2.1 billion to +2.1 billion         | 4 MB                         |
| `torch.int64` | about -9.2 quintillion to +9.2 quintillion | 8 MB                         |

```python
big = torch.tensor([100, 200, 300])                      # create whole numbers (default int64)
small = big.type(torch.int8)                             # convert to 8-bit whole numbers (-128 to 127)
pixels = torch.tensor([0, 128, 255], dtype=torch.uint8)  # 8-bit unsigned: 0 to 255, like pixel values
```

`small` becomes `[100, -56, 44]`: 200 and 300 are too big for int8, so they wrap around.

### Timing code

```python
%%time
vec @ vec  # %%time on the first line of a notebook cell measures how long the cell takes
```

- "Wall time" in the output is the real time you waited
- `%time` (single %) times just one line
- Magic commands like `%%time` work in notebooks, not in `.py` files

**Remember**

- `torch.mm` only works on 2-D matrices; `torch.matmul` and `@` also handle vectors
- Error messages often say "Long", which means `int64`
- `uint8` (0 to 255) is common for image pixel values
- Smaller types save memory; 8-bit models are used on mobile devices

**Watch out**

- Numbers too big for an integer type wrap around silently, with no error
- Python for loops over tensors are very slow; use PyTorch operations like `@` instead

---

## Topic 23 - Min, Max, Mean and Sum

These operations reduce a whole tensor to a single number (also called aggregation).

```python
marks = torch.tensor([45, 60, 72, 38, 85])  # create a tensor of 5 students' marks
marks.min()                                 # lowest mark
marks.max()                                 # highest mark
marks.sum()                                 # total of all marks
marks.type(torch.float32).mean()            # average mark (mean needs a float tensor)
marks.argmin()                              # position of the lowest mark
marks.argmax()                              # position of the highest mark
marks[marks.argmax()]                       # use the position to look up the highest mark
```

| Operation   | Method           | Function             | Result for the example |
|-------------|------------------|----------------------|------------------------|
| Lowest      | `x.min()`        | `torch.min(x)`       | `tensor(38)`           |
| Highest     | `x.max()`        | `torch.max(x)`       | `tensor(85)`           |
| Total       | `x.sum()`        | `torch.sum(x)`       | `tensor(300)`          |
| Average     | `x.mean()`       | `torch.mean(x)`      | `tensor(60.)`          |
| Lowest at   | `x.argmin()`     | `torch.argmin(x)`    | `tensor(3)`            |
| Highest at  | `x.argmax()`     | `torch.argmax(x)`    | `tensor(4)`            |

**Remember**

- `argmin` / `argmax` give the position (index), not the value
- If the max appears more than once, `argmax` returns the first position
- Classification models use `argmax` to turn scores into a predicted label
- `.item()` turns a one-number result into a plain Python number

**Watch out**

- `mean` fails on whole-number tensors (`Got: Long`); convert first: `x.type(torch.float32).mean()`
- `torch.argmax()` needs a tensor inside the brackets: `torch.argmax(x)`

---

## Extra - Using dim with sum, mean, max and argmax

`dim` tells PyTorch which direction to work in. The `dim` you choose is the dimension that disappears.

```python
# rows = students, columns = subjects
marks = torch.tensor([[80, 70, 90],
                      [60, 85, 75]])
marks.sum(dim=0)                            # add down each column: total per subject
marks.sum(dim=1)                            # add across each row: total per student
marks.type(torch.float32).mean(dim=1)       # average per student (mean needs float)
marks.argmax(dim=1)                         # position of each student's best subject
marks.max(dim=0).values                     # highest mark in each subject
```

| Code           | Works            | Result for a `[2, 3]` tensor |
|----------------|------------------|------------------------------|
| `x.sum()`      | On everything    | One number                   |
| `x.sum(dim=0)` | Down each column | Shape `[3]`, one per column  |
| `x.sum(dim=1)` | Across each row  | Shape `[2]`, one per row     |

**Remember**

- "Per row" means `dim=1`; "per column" means `dim=0`
- `x.max(dim=...)` returns values and positions: use `.values` or `.indices`
- `dim` in PyTorch works like `axis` in NumPy

---

## Topic 25 - Reshaping, Viewing and Stacking

### reshape

Rearranges the same numbers into a new shape. The numbers are filled in order, row by row.

```python
eggs = torch.arange(1, 13)  # create 12 numbers: 1 to 12
eggs.reshape(3, 4)          # 3 rows of 4
eggs.reshape(1, 12)         # add an extra dimension: 1 row of 12
eggs.reshape(3, -1)         # 3 rows; PyTorch works out the columns
eggs.reshape(-1)            # flatten into one long row
```

### view

Shows the same data in a new shape without copying. Changing one changes the other.

```python
x = torch.tensor([10, 20, 30, 40])  # create 4 numbers
z = x.view(2, 2)                    # the same numbers seen as 2 rows of 2
z[0, 0] = 100                       # change a number through the view (x changes too)
copy = x.clone().view(2, 2)         # an independent copy, so changes don't affect x
```

|                        | `view`                    | `reshape`                               |
|------------------------|---------------------------|-----------------------------------------|
| Same data or copy?     | Always the same data      | Same data if possible, otherwise a copy |
| If it can't share data | Gives an error            | Makes a copy and works                  |
| When to use            | When you want shared data | By default                              |

### stack

Joins tensors of the same shape and creates a new dimension.

```python
a = torch.tensor([80, 70, 90])  # marks in test 1
b = torch.tensor([60, 85, 75])  # marks in test 2
torch.stack([a, b], dim=0)      # one below the other: each tensor is a row
torch.stack([a, b], dim=1)      # side by side: each tensor is a column
```

| Code                         | Result for 2 tensors of shape `[3]` |
|------------------------------|-------------------------------------|
| `torch.stack([a, b], dim=0)` | `[2, 3]`, each tensor is a row      |
| `torch.stack([a, b], dim=1)` | `[3, 2]`, each tensor is a column   |

**Remember**

- The product of the new shape must equal the total number of items
- `-1` means "work this size out"; only one `-1` is allowed
- `x.clone()` makes an independent copy (like NumPy's `.copy()`)
- The original tensor keeps its shape; store the reshaped result

**Watch out**

- Wrong total gives: `RuntimeError: shape '[5, 3]' is invalid for input of size 12`
- After `.T` (transpose), use `reshape`, not `view`
- `stack` needs every tensor to have the same shape

---

## Topic 26 - Squeezing, Unsqueezing and Permuting

A dimension of size 1 is extra wrapping: an extra bracket, but no extra numbers.

### squeeze and unsqueeze

```python
x = torch.tensor([[10, 20, 30]])  # shape [1, 3]
x.squeeze()                       # remove every size-1 dimension: shape [3]

v = torch.tensor([10, 20, 30])    # shape [3]
v.unsqueeze(dim=0)                # add a size-1 dimension at the front: shape [1, 3] (a row)
v.unsqueeze(dim=1)                # add a size-1 dimension at the end: shape [3, 1] (a column)
```

| Start shape     | Code               | New shape          |
|-----------------|--------------------|--------------------|
| `[1, 3, 1, 4]`  | `squeeze()`        | `[3, 4]`           |
| `[2, 3]`        | `squeeze()`        | `[2, 3]` (no change) |
| `[3]`           | `unsqueeze(dim=0)` | `[1, 3]`           |
| `[3]`           | `unsqueeze(dim=1)` | `[3, 1]`           |
| `[3, 224, 224]` | `unsqueeze(dim=0)` | `[1, 3, 224, 224]` |

### permute

Reorders the dimensions. The numbers are the old positions, written in the new order.

```python
img = torch.rand(224, 224, 3)  # height, width, colour
img.permute(2, 0, 1)           # colour, height, width: shape [3, 224, 224]
```

**Remember**

- `squeeze` removes size-1 dimensions; it does not subtract 1 from sizes
- `unsqueeze` needs `dim`: the position where the new 1 goes
- Common uses: add a batch dimension (`unsqueeze(dim=0)`), one feature per sample (`unsqueeze(dim=1)`), remove extra dimensions from model outputs (`squeeze()`)
- `permute` keeps every value with its correct row, column and channel
- For 2 dimensions, `permute(1, 0)` is the same as `.T`

**Watch out**

- `reshape` to the same shape scrambles data; use `permute` to reorder dimensions
- `permute` needs exactly one number per dimension
- `torch.tensor([10, 20, 30])` holds three numbers; `torch.rand(10, 20, 30)` has shape `[10, 20, 30]`
- `permute` shares data with the original (like `view`); after permuting, use `reshape`, not `view`

---

## Topic 27 - Selecting Data (Indexing)

Indexing picks values out of a tensor with `[ ]`. Counting starts at 0.

### Rows, columns and single values

```python
# rows = students, columns = subjects
marks = torch.tensor([[80, 70, 90],
                      [60, 85, 75],
                      [95, 65, 88]])
marks[1]        # row 1
marks[1, 2]     # row 1, column 2 (same as marks[1][2])
marks[-1]       # the last row
marks[:, 0]     # all rows, column 0
marks[0:2, :]   # rows 0 and 1, all columns
marks[:, 1:]    # all rows, columns 1 to the end
```

### 3-D tensors

A 3-D tensor is indexed as `[tray, row, column]`.

```python
x = torch.arange(1, 10).reshape(1, 3, 3)  # 1 tray, 3 rows, 3 columns
x[0, 2, 2]                                # tray 0, row 2, column 2: the number 9
x[0, :, 2]                                # tray 0, all rows, column 2: [3, 6, 9]
x[:, :, 1]                                # all trays, all rows, column 1: [[2, 5, 8]]
```

| Index type     | Example  | Effect on the dimension |
|----------------|----------|-------------------------|
| Single number  | `x[3]`   | Removed                 |
| Colon          | `x[:, 0]`| Kept (all of it)        |
| Range          | `x[3:]`  | Kept (part of it)       |

**Remember**

- `:` means "all" of that dimension
- `start:end` selects a range; the end is not included
- Negative indexes count from the end: `-1` is the last
- Shape shortcut: each picked number removes its size, each `:` or range keeps it
- `x[i][j]` and `x[i, j]` give the same result; the comma style is preferred

**Watch out**

- `x[3]` has shape `[3]` but `x[3:]` has shape `[1, 3]`: a single index removes the dimension, a range keeps it
- A selected column comes out as a 1-D tensor (printed as a row)
- Extra brackets in a result mean `:` kept a dimension of size 1

---

## Topic 28 - PyTorch and NumPy

Data often arrives as NumPy arrays (from pandas, image loaders and so on). PyTorch models need tensors, so we convert between them.

### NumPy to PyTorch

```python
import numpy as np                                      # load the NumPy library

array = np.arange(1.0, 8.0)                             # a NumPy array (float64 by default)
tensor = torch.from_numpy(array)                        # NumPy to PyTorch: keeps float64
tensor32 = torch.from_numpy(array).type(torch.float32)  # convert, then change to float32 for models
```

### PyTorch to NumPy

```python
t = torch.ones(3)                                       # a float32 tensor
a = t.numpy()                                           # PyTorch to NumPy: keeps float32
safe = t.clone().numpy()                                # copy first for a fully independent array
```

### Same data or separate?

| Conversion                                  | Same data or separate?   |
|---------------------------------------------|--------------------------|
| `torch.from_numpy(arr)`                     | Same data                |
| `t.numpy()`                                 | Same data                |
| `torch.from_numpy(arr).type(torch.float32)` | Separate (new dtype)     |
| `torch.tensor(arr)`                         | Separate (always copies) |
| `t.clone().numpy()`                         | Separate                 |

When they share data:

| How you change it       | Does the other one change?    |
|-------------------------|-------------------------------|
| `x[0] = 100` (in place) | Yes, the data is shared       |
| `x = x + 1` (reassign)  | No, a new object is created   |
| After `.clone()`        | No, they no longer share data |

**Remember**

- NumPy's default decimal type is float64; PyTorch's is float32
- `from_numpy` and `.numpy()` keep the original dtype
- Add `.type(torch.float32)` after converting from NumPy, because models work in float32
- Use `.clone()` before `.numpy()` when you need an independent copy

**Watch out**

- float64 data in a float32 model causes a "wrong datatype" error
- In-place changes affect both the tensor and the array when they share data
- `.numpy()` only works on CPU tensors (use `.cpu()` first for GPU tensors)

---

## Topic 29 - Reproducibility

Random numbers in PyTorch are pseudo-random: they come from a fixed list. A seed picks which list to use and starts at its beginning, so the same seed always gives the same numbers.

```python
torch.manual_seed(42)  # go to the start of list 42
A = torch.rand(3, 4)   # read the first 12 numbers
torch.manual_seed(42)  # go back to the start of list 42
B = torch.rand(3, 4)   # read the same 12 numbers again
print(A == B)          # all True: compare position by position
```

| What you do                         | Same numbers? |
|-------------------------------------|---------------|
| No seed, call `rand` twice          | No            |
| Set the seed before each `rand`     | Yes           |
| Set the seed once, call `rand` twice | No (it keeps reading the list) |

**Remember**

- `torch.manual_seed(n)` picks list number n; any whole number works
- 42 has no special meaning; it's a popular joke from a novel
- `a == b` compares two tensors position by position (True / False)
- Seeds make results repeatable: a model starts with the same random weights every run

**Watch out**

- The seed only sets the starting point; set it again to repeat the numbers

---

## Topic 30 - Accessing a GPU

A GPU does thousands of simple calculations at the same time, which makes deep learning much faster. CUDA is NVIDIA's software that lets PyTorch use an NVIDIA GPU: PyTorch -> CUDA -> GPU.

### Checking for a GPU

```python
torch.cuda.is_available()          # True if PyTorch can use an NVIDIA GPU
torch.cuda.device_count()          # how many NVIDIA GPUs PyTorch can see
torch.backends.mps.is_available()  # Apple Silicon Macs only
```

```python
!nvidia-smi   # run NVIDIA's tool to show the GPU's name and memory
```

| Check                       | Question it asks               | Laptop (Intel only) | Colab (T4 GPU)         |
|-----------------------------|--------------------------------|---------------------|------------------------|
| `torch.cuda.is_available()` | Can PyTorch use an NVIDIA GPU? | `False`             | `True`                 |
| `torch.cuda.device_count()` | How many NVIDIA GPUs?          | `0`                 | `1`                    |
| `!nvidia-smi`               | What GPU is this?              | Error               | Table showing Tesla T4 |

### Getting a free GPU on Google Colab

1. colab.research.google.com -> New notebook
2. Runtime -> Change runtime type -> T4 GPU -> Save
3. Run the checks above (PyTorch is already installed)

**Remember**

- `cuda` in PyTorch code means "the NVIDIA GPU"; `cuda:0` is the first one
- `!` at the start of a notebook cell runs a terminal command
- Without a GPU, PyTorch uses the CPU: slower, but everything works
- Use the GPU only when needed; for small tensors the CPU is fine

**Watch out**

- CUDA only works with NVIDIA GPUs, not Intel or AMD
- Changing Colab's runtime type resets the session; rerun cells from the top
- Free Colab GPU time is limited, and sessions reset when idle
- On Windows, the default `pip install torch` is CPU-only, even with an NVIDIA GPU

---
## Topic 31 - Device-Agnostic Code

Device-agnostic code runs on CPU or GPU without changing any lines.

```python
device = "cuda" if torch.cuda.is_available() else "cpu"  # use the NVIDIA GPU if there is one, otherwise the CPU

x = torch.tensor([1, 2, 3])  # tensors are created on the CPU by default
x = x.to(device)             # move a copy to the chosen device and store it back
array = x.cpu().numpy()      # copy back to the CPU before converting to NumPy
```

| Code              | What it does                                           |
|-------------------|--------------------------------------------------------|
| `x.to(device)`    | Copy x to the chosen device                            |
| `x.cpu()`         | Copy x back to the CPU (does nothing if already there) |
| `x.cpu().numpy()` | Convert to NumPy from any device                       |
| `x.device`        | Where x lives: `cpu` or `cuda:0`                       |

**Remember**

- Write the `device` line once, at the top of the notebook
- `.to(device)` returns a new tensor: store it back with `x = x.to(device)`
- A GPU tensor prints `device='cuda:0'` (the first NVIDIA GPU)
- The 3 common errors: what shape, what datatype, where (device)

**Watch out**

- `.numpy()` on a GPU tensor fails: use `x.cpu().numpy()`
- Tensors on different devices can't be combined: `Expected all tensors to be on the same device`
- Fix device errors by moving every tensor with `.to(device)`

---