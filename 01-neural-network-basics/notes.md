# Neural Network Fundamentals

## What I learned
- A neuron computes z = w1x1 + w2x2 + ... + b, then passes z through an activation function
- McCulloch-Pitts neuron (1943): binary inputs, fixed weights, fires if sum ≥ threshold θ
- A "gate" is a fixed input→output rule; AND needs all inputs on (θ=2), OR needs at least one (θ=1)
- Perceptron (1958): folds the threshold into bias, b = −θ, giving z = w·x + b
- Activation functions exist to (1) reshape z into something useful like a probability, and (2) introduce nonlinearity — without it, stacking layers is pointless, since it collapses into one straight line
- Step function jumps suddenly between 0/1; sigmoid is smooth and reads as a confidence/probability (max derivative 0.25, at z=0)
- "Linearly separable" = one straight line can separate all same-label points — a linear-looking rule doesn't guarantee this
- XOR is not linearly separable (diagonal corners can't be split by one line)
- XOR can be solved either by a nonlinear single-neuron activation (e.g. f(z)=z²) or by combining neurons in a hidden layer
- Universal Approximation Theorem: one hidden layer + enough neurons + nonlinear activation can approximate any continuous function — but gives no bound on neuron count and no training-success guarantee

## Key formulas
- Neuron: z = w·x + b
- Bias: b = −θ
- Sigmoid: 1/(1+e⁻ᶻ), max derivative = 0.25 at z=0