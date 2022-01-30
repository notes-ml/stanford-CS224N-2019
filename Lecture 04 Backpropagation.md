# Backpropagation
## Updating word gradients in window model
Push word vectors around so that they will (in principle) be more helpful:
![](img/e4fa4c85.png)
- words that are in the training data move around, e.g. TV and telly
- words not in the training data stay where they were, e.g. television

> if you only have a small training data set, don’t train / fine-tune / update the word vectors

## Computation graphs and backpropagation
![](img/2aaa5997.png)

**Backpropagation:**
 - recursively apply the chain rule along computation graph
 - [downstream gradient] = [upstream gradient] x [local gradient]
 - Forward pass: compute results of operations and save intermediate values
 - Backward pass: apply chain rule to compute gradients

### Forward Propagation
*expression evaluation*

Edges pass along result of the operation:
![](img/1fc289fc.png)

### Backpropagation
*pass along gradients*

![](img/8fc0adbe.png)
![](img/5f9d4cab.png)

> compute all gradients at once

**Single node**
![](img/542d7876.png)
- node receives an *upstream gradient*
- pass on the correct *downstream gradient*

Each node has a local gradient as the gradient of it’s output with respect to it’s input:
![](img/d732cf3d.png)
- [downstream gradient] = [upstream gradient] x [local gradient]

**Multiple inputs means multiple local gradients**
![](img/dd7ee172.png)

**Gradients sum at outward branches**
Distributes the upstream gradient to each summand:
![](img/9cd7c980.png)

![](img/a6e9bbf3.png)
- `+`: *distributes* the upstream gradient
- `max`: *routes* the upstream gradient
- `*`: *switches/flips* the upstream gradient

**Gradient checking: Numeric Gradient**
Slope/two-sided estimate:
![](img/d14b7107.png)
- approximate and very slow, i.e. recompute for every parameter of our model

> In the old days when we hand-wrote everything, it was key to do this everywhere.

## Deep Learning
### Regularization
*prevents overfitting when we have a lot of features/parameters*

A full loss function in practice includes regularization overall parameters:
![](img/a69679e5.png)

Overfitting/memorization:
![](img/08b6938a.png)

### Nonlinearities
![](img/8433a201.png)
![](img/f7e36749.png)
- left: death neurons
- right: identity

### Parameter initialization
Initialize weights to small random values to avoid/break symmetries that prevent learning/specialization

### Optimizers
Adaptive optimizers scale the parameter adjustment by an accumulated gradient, e.g. Adam, RMSprop.

### Learning rates
Better results can generally be obtained by allowing learning rates to decrease as you train

> Fancier optimizers still use a learning rate but it may be an initial rate that the optimizer shrinks 
