UNIVERSITAT STUTTGART ¨
LEHRSTUHL FUR SYSTEMTHEORIE UND SIGNALVERARBEITUNG ¨
Prof. Dr.-Ing. B. Yang Written exam in Deep learning 11.02.2019 14:00–15:00 Uhr 3 Problems All resources are permitted except electronic calculators.

- Please write only on one side of each page.

- Please submit the cover sheet/envelope, not the problem sheets.

## Problem 1: Machine Learning Basics (14 Points)

Single-choice problem: All questions in this problem can be answered independently. Only one answer per question is correct. Please circle this answer on the problem sheets like A and return the sheets.

1.1 What is correct?

A: Machine learning is equivalent to artificial intelligence.

B: Deep learning is equivalent to machine learning.

C: Deep learning is equivalent to artificial intelligence. D: Machine learning is a subarea of artificial intelligence. E: none of them 1.2 Of which type can be the ground truth in supervised learning?

A: only continuous-valued B: only discrete-valued C: both D: none of them 1.3 Which of the following tasks lacks a signal model?

A: Channel estimation for a mobile communication channel B: Range estimation of a radar C: Delay estimation from a transmitted and a received signal D: Object classification by a radar E: none of them 1.4 For supervised learning, A: only the training set is labeled. B: only the test set is labeled. C: only the training and test set are labeled. D: all training, validation and test set are labeled. E: none of them 1.5 What is correct about the MNIST dataset?

A: It contains multiple digits in one image. B: It contains typed digits.

C: It contains carefully and clearly written digits. D: It contains handwritten digits of one person. E: none of them 1.6 Which of the following datasets is not collected for image recognition, i.e. assign a class label to an entire image?

A: Cityscapes B: MNIST C: Fashion MNIST D: CIFAR-10 E: none of them 1.7 Given x = [0, 8, −6]T with the vector norm kxkp = 14. Which value of p is correct?

A: p = 0 B: p = 1 C: p = 2 D: none of them 1.8 Let L(x(y(θ))) be a cascade of functions with x ∈ R
2, y ∈ R
3, θ ∈ R. Which of the

dimensions of the Jacobi matrices ∂L
 Mathematic: When = < 2x , $\frac{\partial L}{\partial x}$ and $\frac{\partial y}{\partial\theta}$ are correct? 
A: 2x1 and 3x1 B: 2x1 and 1x3 C: 1x2 and 3x1 D: 1x2 and 1x3 E: none of them
1.9 What is a correct one-hot coding for the class labels of a 3-class classification problem?
at is a correct one-not coming for the class labels of a 3-class classification A: $y\in\{1,2,3\}\qquad\text{B:}\;y\in\{\omega_1,\omega_2,\omega_3\}\\ \text{C:}\;\underline{y}\in\left\{\left[\begin{array}{c}1\\ 0\\ 0\end{array}\right],\left[\begin{array}{c}1\\ 1\\ 0\end{array}\right],\left[\begin{array}{c}1\\ 1\\ 1\end{array}\right]\right\}\qquad\text{D:}\;\underline{y}\in\left\{\left[\begin{array}{c}0\\ 1\\ 0\end{array}\right],\left[\begin{array}{c}1\\ 0\\ 0\end{array}\right],\left[\begin{array}{c}1\\ 0\\ 1\end{array}\right]\right\}$ E: none of them
$$\mathrm{\boldmath~h~}{\underline{{x}}}\;\in\;\mathbb{K}^{2},{\underline{{y}}}\;\in\;\mathbb{L}$$
1.10 Which expression is correct for the probability mass function P(x) of a Bernoulli distributed random variable X with P(X = 0) = q?

A: q 1−x(1 − q)
x B: q x(1 − q)
1−x C: x ln(q) + (1 − x) ln(1 − q)
D: (1 − x) ln(q) + x ln(1 − q) E: none of them 1.11 Let δ(x) be the Dirac function and k(x) be another kernel function. What is the empirical distribution ˆp(x) of a random vector X with N i.i.d. samples x(n), 1 ≤ n ≤ N?

A: $\frac{1}{N}\sum_{n=1}^{N}k(\underline{x}-\underline{x}(n))$ B: $\frac{1}{N}\sum_{n=1}^{N}\delta(\underline{x}-\underline{x}(n))$ C: $\frac{1}{N}\sum_{n=1}^{N}k(\underline{x}(n))$ D: $\frac{1}{N}\sum_{n=1}^{N}\delta(\underline{x}(n))$ E: none of them
1.12 Given two probability density functions f1(x) and f2(x). What is the expression R ∞
−∞ f2(x) ln(f1(x))dx −R ∞
−∞ f2(x) ln(f2(x))dx?

A: DKL(f1||f2) B: DKL(f2||f1) C: −DKL(f1||f2) D: −DKL(f2||f1) E: none of them 1.13 Which cost function is typically used to train a neural network for classification?

A: l2-loss (mean square error) B: l1-loss (mean magnitude error)
C: categorical cross entropy loss D: none of them Problem 2: Neural networks, optimization, regularization **(26 points)**
All questions in this problem can be answered independently. Only one answer per question is correct. Please circle this answer on the problem sheets like A and return the sheets.

2.1 Consider a neuron with the input vector x, weight vector w, bias b and the activation function φ(·). What is the output y of the neuron as defined in the lecture?

A: y = w T φ(x) + b B: y = φ(w T x + b)
C: y = φ(w)
T x + b D: y = φ(w T x) + b E: none of them 2.2 What is correct for a nonlinear fully connected layer? Its function is determined by A: the weight vector w only. B: the bias b only. C: the weight vector w and bias b only. D: none of them 2.3 An 1D fully connected neural network consists of 20 input neurons, one hidden layer of 100 hidden neurons and 10 output neurons. What are the total number of model parameters Np and total number of multiplications N× per input sample, respectively?

A: Np = 3000, N× = 3000 B: Np = 3000, N× = 3110 C: Np = 3110, N× = 3000 D: Np = 3100, N× = 3000 E: none of them 2.4 Which activation function is suitable for the output neuron of a binary classification problem?

A: tanh B: ReLU C: sigmoid D: sign function E: none of them 2.5 The input of a softmax activation function φ(·) is a = [0, −1, 0.5]T. Which answer below could be output of the activation function?

A: φ(a) = [0.33, 0.12, 0.55]T B: φ(a) = [0.12, 0.33, 0.55]T C: φ(a) = [0.33, 0.12, 0.65]T D: φ(a) = [0.33, −0.12, 0.56]T
2.6 Which type of neural networks is mostly used for processing images today?

A: fully connected neural network B: convolutional neural network C: recurrent neural network D: generative adversarial network 2.7 Let Np and N× be the number of model parameters and number of multiplications for one input sample in a convolutional layer, respectively. Which statement is correct?

A: Np = N× B: Np ≈ N× C: Np ≪ N× D: Np ≫ N×
2.8 The first 4 convolutional layers of a CNN have the kernel width K1 = K2 = 5 and K3 = K4 = 3 without padding, stride and dilation. The input image has the spatial size of 100 × 100. Which spatial sizes have the output of the 1. and 4. layer?

A: 95 × 95 and 84 × 84 B: 96 × 96 and 84 × 84 C: 96 × 96 and 87 × 87 D: 95 × 95 and 88 × 88 E: none of them 2.9 A convolutional layer applies a 3 × 3 kernel with padding P = 1, stride 2 and dilation distance 2 to an input image of the spatial size 40×40. Which spatial size has the output image?

A: 40 × 40 B: 19 × 19 C: 38 × 38 D: 20 × 20 E: none of them 2.10 Which statement is wrong for a recurrent neural network?

A: All layers have to be recurrent.

B: It has a longer memory than a feedforward network.

C: It contains feedback of neuron outputs along the time axis. D: Backpropagation through time is required. E:none of them 2.11 Which statement is wrong for an autoencoder?

A: It does not require any label. B: It consists of an encoder-decoder structure. C: It encodes the input samples into a latent space. D: The goal is to reconstruct the input from a lower dimensional latent space. E: It is a generative model.

2.12 Which statement is wrong for a discriminative model?

A: It learns the joint distribution p(**x, y**). B: It is able of classification. C: It is able of regression. D: It focuses on the decision boundaries.

2.13 Which neural network is a generative model?

A: CNN B: RNN C: Autoencoder D: GAN E: none of them 2.14 Which statement is wrong for a minibatch-based training of a neural network? During one minibatch, A: the model parameters θ are updated after every sample of the minibatch. B: the model parameters θ are fixed for the forward pass.

C: a gradient vector averaged over all samples of the minibatch is calculated.

D: the model parameters θ are updated once per minibatch.

E: none of them 2.15 The training set consists of 10000 training samples. The training of a neural network requires 40 epochs with a minibatch size of 200. How many iterations are required for a complete training?

A: 400000 B: 50 C: 8000 D: 2000 E: none of them 2.16 Which parameter below is not a hyperparameter?

A: biases bl B: minibatch size C: initial values of the model parameters D: dropout rate E: none of them 2.17 Which hyperparameter is not discrete-valued?

A: type of activation function B: number of epochs C: weight penalty regularization parameters λl D: momentum or not E: none of them 2.18 Why are the hyperparameters not trained together with the model parameters?

A: We are only interested in the model parameters.

B: Some hyperparameters are discrete-valued and hard to optimize.

C: Hyperparameters have no influence on the trained neural network.

D: Hyperparameters can be easily chosen. E: none of them 2.19 What is not a regularization technique?

A: data augmentation B: dropout C: batch normalization D: weight norm penalty E: none of them 2.20 Which statement is wrong for early stopping?

A: It optimizes the number of epochs. B: It requires the training set only. C: It is a hyperparameter optimization. D: It is a regularization method. E: none of them 2.21 Which statement is correct for data augmentation?

A: It is an advanced optimization technique. B: It extends the test set. C: It is easier for regression than for classification. D: It changes the model. E: none of them 2.22 Which statement is correct for dropout?

A: It is applied to training only. B: It is applied to inference only.

C: It is applied to both training and inference. D: It has to be applied to all layers. E: none of them

## Problem 3: Calculations (20 Points)

The following three parts can be answered independently. Please do calculations on your own sheets and return them. Part I: KL divergence Let p(x) ∼ N(µ1, σ2 1
) and q(x) ∼ N(µ2, σ2 2
) be the probability density function of two Gaussian distributions.

3.1 Calculate ln( p(x)
q(x)
).

3.2 Calculate the KL divergence DKL(p||q) = EX∼p hln( p(X)
q(X)
)
ias a function of µ1, µ2, σ2 1, σ2 2.

Part II: Jacobi matrix of the softmax activation function Let a = [ai] ∈ R
c be the input and x = φ(a) = [xi] ∈ R
c be the output of the softmax

```
activation function φ(·) with xi =
                                      e
                                       ai
                                        
                                  Pck=1 e
                                          ak
                                           
                                            .

```

Hint: All details (intermediate results) are required for the following calculations. No score for just giving the final answers.

3.3 Calculate the partial derivative ∂xi/∂ajfor i 6= j.

3.4 Calculate the partial derivative ∂xi/∂ai.

3.5 Calculate the Jacobi matrix of softmax ∂x
$${\frac{\partial x}{\partial\underline{{{a}}}}}=\left[{\frac{\partial x_{i}}{\partial a_{j}}}\right]_{i j}.$$
$$\mathrm{with}\quad\bar{\lambda}=\lambda\gamma.$$
$\text{\rm{101}}^{\circ}$ . 
Part III: l2-regularization using weight norm penalty Let L(w) be the original cost function of w ∈ R
M. The regularized cost function is Lr(w) =
L(w) + λkwk 2 with the regularization parameter λ > 0.

3.6 Calculate the gradient vector ∇Lr(w) as a function of ∇L(w) .

The SGD for min Lr(w) using a fixed learning rate γ t = γ is w t+1 = w t − γ∇Lr(w)w=wt = (1 − 2λ¯)w t − γ∇L(w)w=wt with λ¯ = λγ.

3.7 Interpret the SGD update for λ¯ = 0.

3.8 Interpret the SGD update for λ¯ = 0.5.

3.9 Interpret the SGD update for λ >¯ 0.5.

3.10 Interpret the SGD update for λ <¯ 0.

3.11 What is the reasonable range for λ¯?