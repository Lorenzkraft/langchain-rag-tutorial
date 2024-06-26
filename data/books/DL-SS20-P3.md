UNIVERSITAT STUTTGART ¨
INSTITUT FUR SIGNALVERARBEITUNG UND SYSTEMTHEORIE ¨
Prof. Dr.-Ing. B. Yang

## Exam: Deep Learning

| Date:                 | 09.09.2020   |
|-----------------------|--------------|
| Last name:            |              |
| First name:           |              |
| Matriculation number: |              |

## General Remarks:

- The duration of the exam is 60 minutes. - Exam is open book! No electronic devices are permitted.

- Problem 1 and 2 are single choice. This means there is only one correct answer. Multiple markings will be considered a wrong answer. There is no deduction for wrong answers.

- Only answers on the provided solution sheet will be considered. Intermediate steps are not considered, only the final answer counts.

- Please use a pen to mark the boxes. - Don't forget to fill out your name and matriculation number on the provided solution sheet.

- Problem 3 is an open question. Use your own sheets for answers. - At the end of the exam, all provided sheets have to be returned!

## Problem 3: Calculations (22 Points)

Di↵erent parts of this problem can be solved *independently*. Part 1: Input and output data An RGB color input image has the size 32 ⇥ 32. Each color of a pixel has the value range
{0, 1*,...,* 255}.

3.1 Give the total number of bits required to store the input image. (A formula is sucient.

No need for explicit calculation.)
3.2 In a binary image recognition, the output layer calculates a sigmoid value in 32-bit floating-point. Give the total number of bits to store the output.

3.3 In a 10-class semantic image segmentation, each image pixel is assigned to one of 10 classes. The output layer calculates for each pixel 10 softmax values in 32-bit floatingpoint. Give the total number of bits to store the output.

3.4 In image-to-image translation, a conditional GAN generates for each input image a corresponding black-white image of the same size. Give the total number of bits to store the output.
Part 2: Modified softmax with temperature Let a = [ai] 2 Rd be the input to a modified softmax activation function with *temperature* T > 0. Its output is defined by

$$\underline{{{\phi}}}=[\phi_{i}]\in\mathbb{R}^{d},\quad\phi_{i}=\frac{e^{a_{i}/T}}{\sum_{j=1}^{d}e^{a_{j}/T}},\quad1\leq i\leq d,$$

and interpreted as d estimated class probabilities for the current input. Clearly, Pdi=1 i = 1.

The conventional softmax activation function is a special case of that with T = 1.

3.5 In order to study the e↵ect of T, calculate i/j .

3.6 Assume ai > aj . What happens with i/j if T ! 1 or T ! 0?

3.7 Assume a = [2.3, 0.7, 1.9, 1.5]T 2 R4. Calculate  for the two limiting cases T ! 1 and T ! 0.

3.8 The modified softmax is sometimes used to gradually change a soft classification with 0 < i < 1 to a hard classification with i = 0 or 1. Explain how to achieve this.

3.9 The modified softmax is also used for calibrating the estimated probability k of the winning class k = arg maxi i, i.e. k should agree well with the true class probability known from the labeled dataset. Explain how could we do that.

## Part 3: Kl Divergence

Given two probability density functions (PDF)

$\downarrow$ . 
$$p(x)=1$$
p(x) = ⇢ 12a |x|  a
$$\begin{array}{l l l}{{p}}&{{\frac{1}{2a}}}&{{|x|\leq a}}\\ {{0}}&{{\mathrm{~others~}}}&{{\mathrm{~and~}}}\end{array}\quad q(x)={\frac{1}{2b}}e^{-|x|/b},$$

![2_image_0.png](2_image_0.png)

one from the uniform distribution and one from the Laplacian distribution.

3.10 Give the definition of the KL divergence DKL(p||q) as an integral of p(x)/q(x).

3.11 Calculate *p(x)/q(x*).
3.12 Calculate the KL divergence DKL(p||q) now explicitly as a function of ⇢ = a/b.

3.13 For which value of ⇢ is the KL divergence minimal, i.e. q(x) is the best approximation for p(x)?

3.14 What is the minimum value of the KL divergence?

3.15 Sketch p(x) and q(x) in one diagram for a = 1. Their shapes and all characteristic points should be correctly depicted.