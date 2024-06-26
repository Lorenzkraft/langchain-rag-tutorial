
## Problem 3: Calculations (22 Points)

Different parts of this problem can be solved *independently*.

## Part 1: Loss Functions

We perform a *regression* task on a dataset of N samples: D = {(x1, y1),(x2, y2), . . . ,(xN , yN )},
where each sample xi ∈ R and its label yi ∈ R. The prediction of the i th sample is f(xi). Let l(·, ·) denote the loss for one sample, and the cost function is accordingly defined as

$$L={\frac{1}{N}}\sum_{i=1}^{N}l(f(x_{i}),y_{i})$$

In this regression task, we are studying two losses l1 and l2 defined as l1(f(x), y) = |y − f(x)|, l2(f(*x), y*) = y − f(x)2.

3.1 Let e = y − f(x) as the prediction error, please draw g1(e) = |e| and g2(e) = e 2 versus e ∈ R on one plot and mark the locations where e = 1. *Hints: Please pay attention to* both axis labels of your plot. (2P)
3.2 An outlier is a datapoint which is very different from other datapoints. Based on your plots and the definition above, which loss do you think is more reasonable when there is a large number of outliers in your dataset? (1P)
3.3 "Using l1 can lead to sparsity on the weights of a neural network." Do you agree with this statement in our case? Why/Why not? (2P)
3.4 "Using l2 can lead to very small weights for a neural network." Do you agree with this statement in our case? Why/Why not? (2P)
Now we turn to a binary classification task with xi ∈ R and yi *∈ {−*1, +1} for i = 1*, . . . , N*
and we follow the same notations for predictions and cost function used above. Then, we are studying two losses lA and lB defined as lA(f(x), y) = max{0, 1 − *yf(x*)},
lB(f(x), y) = ln 1 + exp − *yf(x*).

3.5 Rewrite lB in terms of the sigmoid function. (2P)
3.6 If a sample xi has the label yi = −1, what is the minimum possible value of lA and which value of f(xi) can achieve this minimum of lA? (3P)
3.7 If a sample xi has the label yi = −1, what is the minimum possible value of lB and which value of f(xi) can achieve this minimum of lB? (2P)
3.8 We want to design a classifier whose output can be interpreted as a probability. Would you choose to use lA or lB? Explain your choice. (2P)

## Part 2: Dense Neural Network

Consider a two-layer dense neural work for a K-class classification. There is a dataset with N samples: D = {(x1
, y1
),(x2
, y1
), . . . ,(xN , yN
)}, of which each sample is a column vector xi ∈ R
D and its label is in one-hot coding representation yi
∈ {0, 1}
K. And we use superscript to denote the entry of a vector, i.e. the i th sample is notated as xi = [x 1 i
, x2 i
, . . . , xD
i
]. The network is defined below:
layer 1: z1 = W1 · x + b1 activation: a = *ReLU* (z1
)
layer 2: z2 = W2 · a + b2 activation: ˆy = *softmax* (z2
)
Assume that there are M hidden neurons in layer 1, i.e. z1 ∈ R
M.

3.9 What are the shapes of W2 and b2
? (2P)
3.10 A batch of samples is denoted as X = [x1 x2
· · · xN ] ∈ R
D×N . What is the shape of the output of the first layer if we feed the entire batch data to the network? (1P)

```
3.11 What is ∂yˆ
                k
              ∂zk
                2
                 ? Simplify your answer in terms of y
                                                     k. (3P)

```
