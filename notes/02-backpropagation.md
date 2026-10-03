# Backpropagation

## What I learned
- Loss function measures how wrong a prediction is; Squared Error L=(ŷ−y)² squares the error so signs don't cancel and big mistakes count more
- Gradient descent: Loss plotted against a weight is a bowl shape with one minimum; the gradient (slope) tells which direction to move
- Update rule: w_new = w_old − η×(dL/dw) — the minus sign always points downhill regardless of the slope's sign
- Epochs = how many times the update loop repeats; watch the loss curve to decide when to stop; too many risks overfitting
- Backpropagation = the chain rule applied backward through every hop from the Loss to a given weight, however deep it's buried (e.g. w_A → z_A → a_A → z_C → a_C → Loss)
- Binary Cross-Entropy (BCE) loss: L = −[y·log(ŷ) + (1−y)·log(1−ŷ)] — built for yes/no problems, punishes confident wrong answers far more harshly than squared error
- δ (delta) = shorthand for dLoss/dz, the bridge value between the loss and a weight
- δ_BCE = ŷ − y is a derived shortcut, valid only when sigmoid activation is paired with BCE loss — the ŷ terms in sigmoid's derivative and BCE's derivative cancel out exactly

## Key formulas
- Squared Error: L = (ŷ−y)²
- Gradient descent update: w_new = w_old − η×(dL/dw)
- BCE loss: L = −[y·log(ŷ) + (1−y)·log(1−ŷ)]
- δ_BCE = ŷ − y (sigmoid + BCE only)