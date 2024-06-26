
## Solutions To Problem 3: Calculations (22 Points)

Part 1: Input and output data 3.1 (1P) 32 × 32 × 3 × 8 bits 3.2 (1P) 32 bits 3.3 (1P) 32 × 32 × 10 × 32 bits

3.4 (1P) 32 × 32 × 1 bits
Part 2: Modified softmax with temperature 3.5 (1P) φi/φj = e(ai−aj )/T .

3.6 (2P) If T → ∞: φi/φj → 1. If T → 0: φi/φj → ∞.

3.7 (2P) If T → ∞: φ → [1/4, 1/4, 1/4, 1/4]T. If T → 0: φ → [0, 0, 1, 0]T .

3.8 (1P) Change the temperature T gradually and force T → 0.

3.9 (2P) Adjust the temperature T such that the estimated probability φk of the winning class k agrees well with the true class probability from the dataset.
Part 3: KL divergence

3.10 (1P) DKL(p||q) = ! ∞
$$\int_{-\infty}^{\infty}p(x)\ln\left({\frac{p(x)}{q(x)}}\right)\,d x$$
$$3.11\;\;(1\mathrm{P})\;{\frac{p(x)}{q(x)}}={\left\{\begin{array}{l l}{{\frac{b}{a}}e^{|x|/b}}&{|x|\leq a}\\ {0}&{{\mathrm{~others}}}\end{array}\right.}$$
$$3.12\;\;(3\mathrm{P})$$
$$D_{\mathrm{KL}}(p||q)=\int_{-a}^{a}{\frac{1}{2a}}\ln\left({\frac{b}{a}}e^{|x|/b}\right)d x=\ln\left({\frac{b}{a}}\right)+\int_{-a}^{a}{\frac{1}{2a b}}|x|d x=\ln\left({\frac{b}{a}}\right)+{\frac{a}{2b}}={\frac{\rho}{2}}-\ln(\rho).$$
3.13 (2P) Setting the derivative 12 − 1ρ of DKL(p||q) with respect to ρ to zero, we get ρ = 2.

3.14 (1P) DKL(p||q)=1 − ln(2) for ρ = 2.

![0_image_0.png](0_image_0.png)

$$3.15\ \left(2\mathrm{P}\right)\ {\underline{\ \ \ \ \ \ }}$$
3.15 (2P) x