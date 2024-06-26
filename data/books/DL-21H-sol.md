
## Solutions To Problem 3: Calculations (22 Points) Part 1: Loss Functions

3.1 g1(e) is shown in solid line, while g2(e) is shown in dashed line. 2

![0_image_0.png](0_image_0.png)

$$\boxed{2}$$

3.2 For error |e| > 1, l2 = e 2significantly amplifies the errors and penalizes for outliers. So

we choose l1. 1 3.3 This statement is false. The l1-regularization on weights can lead to sparsity in neural network. Here, we use l1 loss as the learning objective to train the network instead of as a regularization on weights. 2
3.4 This statement is false. The l2-regularization on weights can lead to very small weights in neural network. Here, we use l2 loss as the learning objective to train the network instead of as a regularization on weights. 2 3.5 Let the sigmoid function be denoted as σ(z) = 1 1+exp(−z)
. Then, lB can be rewritten as 2 lB = ln 1 + exp(−yf(x))

$$\begin{array}{l}{{\exp(-y J(x))}}\\ {{(x))^{-1}}}\\ {{f(x))}}\end{array}$$
$=-1)=\max$
$$\mathbf{\tau},1+f(x_{i})\}$$

= ln *σ(yf(x*))−1
= − ln σyf(x)
3.6 If yi = −1, then lA(f(xi), yi = −1) = max{0, 1 + f(xi)}. Consider the following cases: 3
- 1 + f(xi) ≤ 0 ⇔ f(xi) ≤ −1: lA = 0 - 1 + f(xi) > 0 ⇔ f(xi) > −1: lA = 1 + f(xi) > 0 Therefore, the minimum value of lA is 0 and any f(xi) ≤ 0 can achieve this minimum.

3.7 If yi = −1, then lB(f(xi), yi = −1) = ln 1 + exp f(xi). In this case, f(xi) will approach −∞ to minimize lB and in this case lB is approaching 0. 2

3.8 It is better to choose lB. From the previous questions, we know that lB can be written as minimizing the log of sigmoid of the output. This means that the sigmoid of the output can be interpreted as a probability. However, lA has no such interpretation. 2

## Part 2: Dense Neural Network

3.9 W2 ∈ R
K×M and b2 ∈ R
K. 2

$${\mathrm{is~}}M\times N.$$

3.10 The output shape of layer 1 is M × N. 1 3.11  Let Z denote ∑ j =1 exp( z 2 ). The calculation is:

∂∂∂  _ exp( z ) · Z − exp( z ) · exp( z ) Z · Z = exp(z )  Z = î k · (1 − î k )