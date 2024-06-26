Short test 2: Dense network, optimization and regularization, 7min Single-choice problem: All questions in this problem can be answered *independently*. Only one answer per question is correct. Please *circle* this answer on the problem sheet.

1. What is correct for a dense neural network?

A: All neurons of the network are connected to each other.

B: All neurons of a layer are connected to each other.

C: All neurons of a layer are connected to all neurons of the adjacent layers.

D: All output neurons are connected to the input neurons.

2. Consider a neuron with the input vector x, weight vector w, bias b and the activation function φ(·). What is the output y of the neuron?

A: y = w T φ(x) + b B: y = φ(w Tx + b)
C: y = φ(w)
Tx + b D: y = φ(w Tx) + b 3. Let φ(a) = 1/(1 + e
−a) be the sigmoid activation function. What is the correct derivative of φ(a)?

A: dφ(a)/da = φ(a)φ(a) B: dφ(a)/da = φ(a)φ(−a)
C: dφ(a)/da = φ(−a)φ(−a) D: dφ(a)/da = φ(a)
4. Which statement is *wrong* about a minibatch-based training of a neural network? During one minibatch, A: the model parameters θ are updated after every sample of the minibatch.

B: the model parameters θ are fixed for the forward pass.

C: a gradient vector averaged over all samples of the minibatch is calculated.

D: the model parameters θ are updated once per minibatch.

5. The training set consists of 20000 training samples. The training of a neural network requires 20 epochs with a minibatch size of 400. How many training steps (iterations) are required for a complete training?

A: 400000 B: 1000 C: 50 D: 8000 E: none of them 6. The hyperparameter optimization is done by using A: the training set. B: the test set.

C: the validation set. D: all three sets above.

7. Which statement is correct about dropout?

A: It is applied to training only. B: It is applied to inference only.

C: It is applied to both training and inference.

D: It has to be applied to all layers. E: none of them