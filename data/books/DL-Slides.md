

![0_image_1.png](0_image_1.png)

University of Stuttgart Faculty of Computer Science, Electrical Engineering and Information Technology

![0_image_0.png](0_image_0.png) 
Institute of Signal Processing and System Theory (ISS)

Prof. Dr.-Ing. B. Yang Slides to the lecture Deep Learning www.iss.uni-stuttgart.de

## Organization Of The Course When And Where

Monday, 15:45-17:15 and Thursday, 8:00-9:30
•
lecture hall V47.03
•

## Tools In Lecture

laptop projector

+
•
roughly 360 prepared slides writing on laptop
+
•
no complete script, but video recording of lecture

•

## On Ilias (Ilias3.Uni-Stuttgart.De) You Will Find

time schedule

- •
slides of the lecture

•
video recordings of the lecture

•
Python/TensorFlow introduction, programming homeworks, assignments old exam problems solutions
+
•
FAQ for old questions, Q&A forum for new questions and discussion
•
many fundamental papers about deep learning
•
"A recipe of designing and training neural networks for beginners" from ISS
•

## Goals Learn

machine learning basics

•
fundamentals and advanced concepts of deep learning
•
programming in Python and TensorFlow
•

## Prerequisites

The course "Advanced mathematics for signal and information processing" (AM) 
•
advanced vector and matrix computations 
–
probability theory (probability, random variables, stochastic processes) 
–
introduction into optimization
–

```
                                          
is required.
             

```

The course "Detection and pattern recognition" (DPR) is useful, but not mandatory.

•
Almost no overlap between DPR and this course. Both courses can be attended in parallel.

Basic knowledge about some concepts from digital signal processing: 
•
convolution, correlation, non-recursive filter, recursive filter 
–
downsampling, upsampling
- 
These concepts will be reused in deep learning.

## Deep Learning Course (6Cp)

Lecture: comprehensive

•
•
Exercise (traditional blackboard calculation): no Programming (i.e. an integrated minilab) 
•
Python introduction (online materials on Ilias): self-learning
–
Python programming practice (simple tasks on Ilias): self-doing
–
TensorFlow introduction: in lecture hall

–
Assignments 
•
TensorFlow assignments introduction
(online videos on Ilias): self-learning
–
TensorFlow assignments (challenging projects): self-doing 
–
mandatory for applying the Deep Learning Lab
- 
Module description: "Presence time 46h, self study *134h*"

•
Exam: all topics above except for ⋆-slides and TensorFlow assignments Deep Learning Lab in winter term (6CP): Advanced projects limited capacity
•
only for selected Master programs (ETIT, EEng, EMOB, InfoTech, AS, MedTech)
•

## Literature

Machine learning (excellent text books):

[DHS04] R. O. Duda, P. E. Hart, D. G. Stork, "Pattern classification", John Wiley & Sons, 2. edition, 2004 
[B06] C. Bishop, "Pattern recognition and machine learning", Springer, 2006 (online available, google it.)

Deep learning (active research area, no comprehensive text books):

[GBC16] I. Goodfellow and Y. Bengio and A. Courville, "Deep learning",

www.deeplearningbook.org, 2016.

[N18] A. Ng, "Machine Learning Yearning", www.mlyearning.org, 2018.

[Ilias] many fundamental papers about deep learning 
[arXiv] the largest open-access archive with millions of preprint papers on arxiv.org

# Content

1. Introduction 1.1 What is machine learning?

1.2 What is deep learning?

1.3 Examples 2. Tools for deep learning

```
                          
     2.1 Software
                 
     2.2 Hardware
                  
     2.3 Datasets
                 
3. Machine learning basics
                            
     3.1 Linear algebra
                       
     3.2 Random variable and probability distribution

                                              
3.2.1 One random vector
                          

```

3.2.2 Multiple random vectors 3.2.3 Kernel-based density estimation 3.3 Kullback-Leibler divergence and cross entropy 3.4 Probabilistic framework of supervised learning 3.5 Some concepts from digital signal processing 4. Dense neural networks 4.1 Neuron

```
          
4.2 Layer
        

```

4.3 Feedforward neural network 4.4 Activation function 4.5 Universal approximation 5. Model learning 5.1 Loss and cost function 5.1.1 Regression 5.1.2 Classification 5.1.3 Semantic image segmentation 5.1.4 Object detection 5.2 Training and validation 5.3 Implementation in Python 5.4 Challenges in optimization 5.5 Momentum 5.6 Learning rate schedule 5.7 Normalization 5.7.1 Input normalization 5.7.2 Batch normalization 5.8 Parameter initialization 5.9 Preventing vanishing and exploding gradient 6. Overfitting and regularizations 6.1 Model capacity and overfitting 6.2 Weight norm penalty 6.3 Early stopping 6.4 Data augmentation 6.5 Ensemble learning

```
                     
6.6 Dropout
           

```

6.7 Hyperparameter optimization

```
                                             
7. Convolutional neural networks
                                        
      7.1 Convolutional layer
                                  

```

7.2 Variants of convolution 7.3 Downsampling layers 7.4 Upsampling layers 7.5 Normalization layers 7.6 Reshaping layers 7.7 Architecture of CNNs 7.8 Classical networks and modules 8. Recurrent neural networks 8.1 Recurrent layer and recurrent neural network 8.2 Bidirectional recurrent neural network 8.3 Long short-term memory 9. Attention and transformers 9.1 Channel and spatial attention 9.2 Self-attention

```
                 
9.3 Language transformer
                          

```

9.4 Vision transformer 10 Self-supervised learning 10.1 Pretraining and finetuning

```
                             
10.2 Autoencoder
                

```

10.3 Pretext task 10.4 Contrastive learning 10.5 Foundation models 10.6 Transfer learning

```
                             
11. Generative models
                        
     11.1 Variational autoencoder
                                    

```

11.2 Generative adversarial network

```
                                   
11.3 Di
      
       ffusion model
                    

```

12. Further topics and outlook 12.1 Important but unaddressed topics 12.2 AI vs. human intelligence

# Mathematical Notations And Symbols

Basics:

scalar: x, A, α -  x1  = N column [xi] ∈ R  with element xi = [x]i vector: x . . . - = x N M × N matrix: A [ai j]1≤i≤ [ai j] ∈ R  with element ai j = [A]i j - = = M,1≤ j ≤ N
 


 
 

 


 

 

 

 

 

 


 

 

 

3D, 4D
tensor: A
[ai jk], A
[ai jkl]

•
=
=
T

=
T

=
transpose of vector and matrix:
x
[x1, . . . ,
x N], A
[aji]i j

•
 of a square matrix determinant A: |A|
•
−1 of a square matrix inverse A: A
•
Machine learning:

input/output space: X, Y
•
X, Y, X × Y

•
distribution: p

(

x
), p
(y

), p
(

x, y) over 
•
)

}

N
labeled dataset for supervised learning:

D
{

x
(

n
), y

(

n
=

```
                                                                            
                                                                            n
                                                                            
                                                                             =1
                                                                               
•

```

)

}

N
unlabeled dataset for unsupervised learning:

D
{

x
(

n
=

```
                                                                           
                                                                           n
                                                                           
                                                                            =1
                                                                              
•

```

{X, Y, p f : X → Y based on p task: T
(

x, y

)

} to learn a function

(

x, y

)
=

# List Of Acronyms

| Adam   | adaptive moment estimation                 |
|--------|--------------------------------------------|
| AE     | autoencoder                                |
| AI     | artificial intelligence                    |
| BN     | batch normalization                        |
| BRNN   | bidirectional recurrent neural network     |
| CBAM   | convolutional block attention module       |
| cGAN   | conditional generative adversarial network |
| CL     | convolutional layer                        |
| CNN    | convolutional neural network               |
| DL     | deep learning                              |
| DNN    | deep neural network                        |
| DRL    | deep reinforcement learning                |
| DT     | downstream task                            |
| ELU    | exponential linear unit                    |
| FIR    | finite impulse response                    |
| FM     | foundation model                           |
| GAN    | generative adversarial network             |
| GAP    | global average pooling                     |
| GNN    | graph neural network                       |

| GPT      | generative pretrained transformer                 |
|----------|---------------------------------------------------|
| Grad-CAM | gradient-weighted class activation mapping        |
| GRU      | gated recurrent unit                              |
| IIR      | infinite impulse response                         |
| ILSVRC   | ImageNet large scale visual recognition challenge |
| KLD      | Kullback-Leibler divergence                       |
| LLM      | large language model                              |
| LSTM     | long short-term memory                            |
| MAE      | mean absolute error                               |
| MHA      | multi-head attention                              |
| ML       | machine learning                                  |
| MLP      | multilayer perceptron                             |
| MSE      | mean square error                                 |
| NAS      | neural architecture search                        |
| NLL      | negative log-likelihood                           |
| NLP      | natural language processing                       |
| PDF      | probability density function                      |
| PMF      | probability mass function                         |
| PT       | pretext task                                      |
| ReLU     | rectifier linear unit                             |

| ResNet   | residual network                            |
|----------|---------------------------------------------|
| RL       | reinforcement learning                      |
| RNN      | recurrent neural network                    |
| RV       | random variable                             |
| SE       | squeeze and excitation                      |
| SGD      | stochastic gradient descent                 |
| SP       | signal processing                           |
| SSL      | self-supervised learning                    |
| t-SNE    | t-distributed stochastic neighbor embedding |
| TTS      | text-to-speech                              |
| VAE      | variational autoencoder                     |
| YOLO     | you only look once                          |

+

# "The Only Constant In Life Is Change"

Heraclitus (535-475 BC)

an increasing number of tools for signal/information/data processing
•
they are similar and different

•

![14_image_1.png](14_image_1.png)

![14_image_0.png](14_image_0.png)

![14_image_3.png](14_image_3.png)

![14_image_5.png](14_image_5.png)

| My life                       |                |      |
|-------------------------------|----------------|------|
| PostDoc, industrystudent, PhD | 20 years later | Prof |

Never stop learning!

![14_image_2.png](14_image_2.png)

![14_image_4.png](14_image_4.png)

# Signal Processing (Sp) And Machine Learning (Ml)

Common: Process a given input **signal** according to a certain processing **rule** (in SW 
or HW) and return a corresponding output.

![15_image_0.png](15_image_0.png)

Task of the computer: Calculate the output from the input according to the pro-
•
cessing rule.

Your *task*: Design the processing rule.

•

# Different Types Of Output

![16_image_0.png](16_image_0.png)

Depending on the application, the desired output can be different:

From signal to
•
From signal to From signal to class
•
signal a vector/matrix/tensor of numbers in R

•
parameter a few numbers in R, e.g. 12.3m for range a number in N, e.g. male/female

![16_image_1.png](16_image_1.png)

## E1.1: Different Types Of Output

From signal to signal:

a) Digital filter: from time-domain signal x(n) to time-domain signal y(n)

b) Fourier analysis: from time-domain signal x(n) to frequency-domain signal X(jω)

From signal to parameter:

c) Synchronization by correlation: position of x(n) in y(n)
d) Radar: from signal x(n) to distance/velocity/direction of target From signal to class:

e) Radar: from signal x(n) to type of target (car, bike, pedestrian, . . .)
f) Speech recognition, speaker identification, image recognition, . . .

| a), b)     | c), d)         | e), f)   | output   |
|------------|----------------|----------|----------|
| signal     | parameter      | class    |          |
| regression | classification |          |          |

# Signal Processing Vs. Machine Learning

Difference: Two different approaches to design the processing rule "?".

![18_image_0.png](18_image_0.png)

![18_image_1.png](18_image_1.png)

(SP) is traditionally Signal processing model-based:

There is a signal **model**, a mathematical description of the input signal as a function of the desired output. The signal processing rule is derived by humans from the signal model.

## Machine Learning

(ML) is learning-based or **data-driven**:
There is no signal model because the relationship between the input signal and desired output is too complicated or even unknown. The processing rule is learned from examples (supervised **learning**) or actions (reinforcement **learning**).

1 We humans are learning-based. We learn to recognize, walk, run, dance, speak, write, calculate etc. from examples and/or actions.

## E1.2: Model-Based Signal Processing

a) Delay estimation uses the signal model x(t) =
As(t −
τ) between the transmitted signal s(t) and received signal x(t) to estimate the delay τ.

b) Radar distance estimation uses the signal model distance = delay · speed.

c) Radar velocity estimation uses the Doppler equation as signal model to estimate the relative velocity between the transmitter and receiver.

d) Channel estimation in digital communication uses the channel model x(n) =
Pi his(n − i) + z(n) between the transmitted signal s(n) and received signal x(n) to estimate the channel coefficients hi.

e) Edge detection by highpass filtering relies on the knowledge (signal model) that edges are high-frequency components.

![19_image_0.png](19_image_0.png)

1.1 What is machine learning? 1-7 E1.3: Machine learning (1): Classification a) Speech recognition (Siri, Alexa)

# "Good morning"  $$\newcommand{\vecs}[1]{\overset{\rightharpoonup}{\mathbf{#1}}}$$  $$\newcommand{\vecd}[1]{\overset{-\!-\!\rightharpoonup}{\vphantom{a}\smash{#1}}}$$

b) Optical character recognition (OCR)

![20_image_0.png](20_image_0.png)

"Stuttgart"

c) Fish sorting in a fish-packing plant

![20_image_1.png](20_image_1.png)

$$\mathrm{\mathrm{\Large~Salmon~or~seebass?}}$$

d) Semantic image segmentation (pixelwise classification) for autonomous driving

![20_image_2.png](20_image_2.png)

![20_image_3.png](20_image_3.png)

1.1 What is machine learning? 1-8

## E1.3: Machine Learning (2): Regression

e) Age estimation from an MR brain image

![21_image_0.png](21_image_0.png)

For learning from examples, one needs to collect a dataset D
containing input samples and their corresponding desired outputs known as ground truth or **labels**. Supervised learning learns the unknown input-output mapping from examples.

# Three Phases Of Supervised Learning (1)

$$\mathrm{Dataset}\ {\mathcal{D}}={\mathcal{D}}_{\mathrm{train}}\cup{\mathcal{D}}_{\mathrm{test}}\ \Big|$$

![22_image_1.png](22_image_1.png)

 


![22_image_3.png](22_image_3.png)

 

| ∪ Dtest   | training set Dtrain   | test set Dtest   |
|-----------|-----------------------|------------------|

a) **Training** using training **data** from the training set Dtrain

![22_image_0.png](22_image_0.png)

$$\begin{array}{r l r l}{{\frac{\mathrm{st}}{\mathrm{tion}}}}&{{}{\mathrm{ground~truth}}}&{{\Rightarrow}}&{{\begin{array}{l l}{\mathrm{training}}\\ {\mathrm{error}}\end{array}}}\end{array}$$

b) Test or **evaluation** using test **data** from the test set Dtest

input output ground truth ⇒

$$\begin{array}{r l}{{\frac{\mathrm{cost}}{\mathrm{function}}}}\end{array}\rightleftarrows\begin{array}{r l}{\mathrm{ground~truth}}&{{}\implies}\end{array}$$

![22_image_2.png](22_image_2.png)

$$\begin{array}{l}{{\mathrm{test}}}\\ {{\mathrm{err}}\,\mathbf{0}\Gamma}\end{array}$$

c) Deployment on new data

$$\mathrm{\underline{{\mathrm{~input~}}}}\atop{}\atop{}\atop{}\atop{}\atop{}$$

# Three Phases Of Supervised Learning (2)

## Dataset

The dataset is divided into two non-overlapping parts, a training set and a test set.

•
The training set is used for learning the rule and the test set is used for its test.2 Training The actual output is compared against the ground truth via a cost **function**.

•
The rule is adjusted to minimize the cost function, i.e., to make the actual output

•
as close to the ground truth as possible.

Test The learned rule is applied to unseen test data to calculate the test error.

•
If test error ≫
training error, overfitting occurs. Instead of learning the input-
•
output relationship, the computer just memorized the training data. It has a poor generalization to new unseen data. Overfitting has to be *avoided*.

Deployment The learned rule is applied to new data in application.

•

## Different Machine Learning Tasks Depending On The Type Of Output:

Classification: discrete-valued output, e.g. E1.3 a)–d)

•
Regression: continuous-valued output, e.g. E1.3 e)

•

## Different Learning Supervision Schemes Depending On The Availability Of Labels Of Training Data:

Supervised **learning**: Each input x(n) has a corresponding ground truth or label

•
y(n). It guides the learning. The dataset {x(n), y(n)} is **labeled**.

Unsupervised **learning**: There is no ground truth. The dataset {x(n)} is **unlabeled**.

•
Classification and regression are not possible. Possible tasks are clustering and dimension reduction, see course DPR, and self-supervised representation learning, see ch. 10.

Deep learning can solve much more tasks (see ch. 1.3) and has much more supervision schemes (see ch. 10).

# I Offer Four Courses For Sp And Ml In Master

Winter term:

(AM, 6CP),

Advanced mathematics for signal and information processing
•
common mathematical fundamentals for all other courses

(SASP, 6CP)

Statistical and adaptive signal processing
•
Summer term:

Detection and pattern recognition
(DPR, 6CP)

•
Deep learning
(DL, 6CP)

•

![25_image_0.png](25_image_0.png) 

| modelbased            | SASP           | SASP   | DPR     |
|------------|----------------|--------|---------|
| learningbased            | DL             | DL     | DPR, DL |
| signal     | parameter      | class  | output  |
| regression | classification |        |         |

# What Is Deep Learning?

ChatGPT's answer:

"Deep learning is a subfield of machine learning that uses algorithms to model highlevel abstractions in data. It is inspired by the structure and function of the brain, and is used to create artificial neural networks that can learn and make predictions from large sets of data. Deep learning can be used for tasks such as image recognition, natural language processing, and speech recognition."

is a large language **model** (LLM) based on generative pretrained transformer (GPT), see future chapters about "generative", "pretrained", and "transformer".

ChatGPT
Be careful: Not all answers by ChatGPT are *correct*.

My answer:

It is a general and the most successful machine learning approach by leveraging neural network as the computational architecture,

•
a surprisingly simple optimization method (stochastic gradient descent), and
•
enough training data and computation resource.

•

# Ai Vs. Ml Vs. Dl Vs. Dnn

There are many similar buzzwords: Artificial **intelligence** (AI), machine **learning**

(ML), deep learning
(DL), deep neural network
(DNN).

Are they the same?

| AI   | machine intelligence               | also expert systems ("if ... then ...")   |
|------|------------------------------------|-------------------------------------------|
|      | =                                  |                                           |
| ML   | via learning                       | also conventional ML                      |
| =    |                                    |                                           |
| DL   | most successful subarea of ML      | also non-architectural topics             |
| =    |                                    |                                           |
| DNN  | a family of learning architectures |                                           |
| =    |                                    |                                           |

![27_image_0.png](27_image_0.png)

# Conventional Machine Learning Vs. Deep Learning (1)

Conventional machine **learning**: ML before DL, see course DPR.

Conventional machine learning Deep learning

$${\mathrm{e.g.~speech,~fish,~street,~patient}}$$

 

$${\mathrm{Ricrophone}},{\mathrm{came}}$$
$\mathcal{I}$

e.g. microphone, camera, radar, CT
raw data, e.g. speech, image, radar signal, CT image

  **C.g. Spear, Image, radar spring, C.F.**  **e.g. filtering, segmentation**

$$\mathrm{d}\;\mathrm{data}$$

# no theory, needs experiences!  

$${\begin{array}{r l}{{\mathrm{features,~e.g.~f s h~l e n g t h,~f s h a p e,~f s h o l o r~}}}\\ {{}}\end{array}}$$

estimated class ∈ {salmon, sea bass, . . .}

![28_image_1.png](28_image_1.png)

preprocessed data

![28_image_0.png](28_image_0.png)

# Conventional Machine Learning Vs. Deep Learning (2)

Conventional machine learning (see course DPR)

a single ML **model** for a single task

- •
small model, small dataset feature **engineering**: handcrafted feature extraction
•
different supervised classifiers: k nearest neighbor (kNN), Gaussian mixture model
(GMM), neural network (NN), support vector machine (SVM), random forest (RF),

•
. . .

•
different unsupervised clustering algorithms: k-means, mean-shift, DBSCAN, . . .

Deep learning a single DL **model** for a single task or a foundation DL model for many tasks
•
large model, large dataset

•
end-to-end learning with automated feature learning
•
based on DNN
•
much more powerful 
•

# A Short History Of Neural Networks

Birth (1940's–1970's): first idea (mimic brain), simple linear neuron, could solve

•
only simple linear tasks 1970's–1980's: low recognition
•
Neural network era (1980's-1990's): 1. renaissance 
•
feedforward multilayer network with nonlinear neurons 
–
a learning algorithm called backpropagation
–
could solve simple nonlinear tasks (e.g. xor, phoneme recognition)

–
1990's–2012's: stagnation
•
Deep neural network era (2012's-present): 2. renaissance 
•
advanced architectures, advanced learning schemes, new regularizations 
–
huge datasets, significantly increased computing power 
–
can solve challenging real-life problems

–
Future? see last chapter
•

# Conventional Neural Network Vs. Deep Neural Network

| Conventional neural network   | Deep neural network (DNN)          |                                   |
|-------------------------------|------------------------------------|-----------------------------------|
| Depth                         | shallow (1∼2 hidden layers)        | deep (many hidden layers)         |
| Architecture                  | CNN, RNN, GAN, Transformer, . . .  |                                   |
| Activation function           | fully connected                    | ReLU, sigmoid, softmax, . . .     |
| Cost function                 | sigmoidmean squared error (MSE)    | MSE, KLD, CE, categorical, . . .  |
| Learning                      | supervised, self-supervised, . . . |                                   |
| Regularization                | supervised                         | L1, L2, shortcuts, dropout, . . . |
| none                          |                                    |                                   |

# Why Is Deep Better (1)?

Many signals have a native hierarchical representation.

Speech recognition: samples →
phonemes →
words dialog sentences
→
→
•
Image recognition: pixels →
edges →
shapes →
objects →
•
scene local groups →
global strategies Go: stones →
•
eyes →
It is easier to capture this multi-level representation by using a deep neural network.

Each layer learns an easy mapping.

•
Simple local low-level features (e.g. edges, corners) in the (first) shallow **layers** are

•
combined to complex global high-level features (e.g. image composition, semantic meaning) in the (last) deep **layers**.

![32_image_0.png](32_image_0.png)

# Why Is Deep Better (2)?⋆3

A shallow architecture requires a higher complexity than a deep (hierarchical) architecture for the same task.

n-bit analog-digital-converter (ADC) 
•
Deep: Serial ADC, n levels for n bits, determine one bit
–
per level →
comparators n Shallow: Flash ADC, one level for all n bits →
2n −
1

![33_image_0.png](33_image_0.png)

–
comparators Addition of two n-bit numbers 
•
Deep: Ripple-carry adder, calculate and propagate the

–
carry bit for bit →
only full adders (FA) 
n Shallow: Carry-lookahead adder, needs a much more

–
complex logical circuit to predict all carry bits

![33_image_1.png](33_image_1.png)

# Deep Learning Vs. Deep Neural Network

Deep neural network refers only to the network architecture.

Deep learning is more. It consists of architecture **engineering**: design DNN architectures

•
objective **engineering**: design loss or loss combinations and learning schemes

•
prompt **engineering**: design prompts to models (e.g. ChatGPT) to guide the

•
model's answer data **engineering**: collect, label, curate and augment a dataset used to train and
•
validate a DL model

![34_image_0.png](34_image_0.png)

## Purpose Of Various Engineering

| Learning needs   | Learning at school/Uni                                       | Deep learning         |
|------------------|--------------------------------------------------------------|-----------------------|
| guidance         | lecture, exercise, lab, talk, . . . architecture engineering |                       |
| supervision      | exam, grading, feedback, . . .                               | objective engineering |
| material         | books, slides, videos, . . .                                 | data engineering      |
| interaction      | Q&A, discussion forum, . . .                                 | prompt engineering    |

## We Focus On Architecture And Objective Engineering

| Ch.   | Topic                                         | Engineering on           |
|-------|-----------------------------------------------|--------------------------|
| 4     | Dense neural networks                         | architecture             |
| 5     | objective                                     |                          |
| 6     | Model learningOverfitting and regularizations | architecture & objective |
| 7     | Convolutional neural networks                 | architecture             |
| 8     | Recurrent neural networks                     | architecture             |
| 9     | Attention and transformers                    | architecture             |
| 10    | objective                                     |                          |
| 11    | Self-supervised learningGenerative models     | architecture & objective |

## E1.4: Image Classification (1) Imagenet

a large dataset for image recognition (classification)

•
over 14 million labeled images

•
over 20 thousand classes

•
image label often not unique because of multiple objects in an image

![36_image_0.png](36_image_0.png)

![36_image_1.png](36_image_1.png)

•

alp grass snake harvester

![36_image_2.png](36_image_2.png)

![36_image_3.png](36_image_3.png)

![36_image_4.png](36_image_4.png)

![36_image_5.png](36_image_5.png)

Shetland sheepdog Eskimo dog old English sheepdog

## E1.4: Image Classification (2)

ImageNet Large Scale Visual Recognition Challenge (**ILSVRC**)

1,000 classes

•
1.2 million training images and 50,000 test images

•
•
Image **classification**: Assign a class label to one image

•
competition among many research teams from universities and companies

![37_image_0.png](37_image_0.png)

Prob(image from class i), 1 i ≤
1000
≤
Challenges:

high variability of images (color, shape, perspective, illumination, . . .)
- •
image class often not unique. Hence top-5 error **rate**: Classification fails if the true class is not among the top 5
–
results. 

top-1 error **rate**: Classification fails if the true class is not the top result.

–

## E1.4: Image Classification (3)

![38_image_0.png](38_image_0.png)

- 2010, 2011: conventional (shallow) neural networks
- since 2012: deep neural networks
- increasing number of layers (deeper) → increasing accuracy
- ResNet better than average human performance (5%)
Is DL already smarter than human? See last chapter "AI vs. human intelligence".

## E1.5: Semantic Image Segmentation (1)

Semantic image **segmentation**: Assign a class label to each pixel in an image, i.e.

pixelwise classification.

a) Applied to Cityscapes dataset to segment street images for automated driving4

![39_image_0.png](39_image_0.png) 

## E1.5: Semantic Image Segmentation (2)

b) Applied to magnetic resonance images (MRI) to segment organs (spleen, liver) for automated medical diagnosis5

![40_image_0.png](40_image_0.png)

## E1.5: Semantic Image Segmentation (3)

c) Applied to accelerometer and gyroscope time series (each 3 channels) of a smartphone for automated human activity recognition 6

![41_image_0.png](41_image_0.png)

## E1.6: Object Detection

Object **detection**: Localize foreground object instances in terms of bounding boxes and classify their contents.7 This is a combined regression and classification task. It is also called instance **segmentation**.

![42_image_0.png](42_image_0.png)

## E1.7: Panoptic Segmentation

Panoptic **segmentation**: Combine semantic segmentation with instance segmentation.

Each pixel is assigned a class label and an instance identifier.8

![43_image_0.png](43_image_0.png)

## E1.8: Age Estimation

Age estimation from the white matter of an MR brain image. It is a regression problem.9

True age 23 86 

Estimated Age 24.2 88.7

Wiki: "White matter is composed of bundles, which connect various gray matter areas

(the locations of nerve cell bodies) of the brain to each other, and carry nerve impulses between neurons."

![44_image_0.png](44_image_0.png)

![44_image_1.png](44_image_1.png)

## E1.9: Geo Guessing

location10 PlaNet: Guess geo locations from image, i.e. image →
•
a game based on this idea: www.geoguessr.com
•

![45_image_0.png](45_image_0.png)

## E1.10: Text-To-Image Translation

a) 2016: "This bird is white with some black on its head and wings, and has a long orange beak."11 b) 2022, Dall·E 2: "Stuttgart Schlossplatz under snow at sunrise" (Source: me)

c) 2023, Midjourney: Pope (Source: Twitter)

![46_image_0.png](46_image_0.png)

## E1.11: Image-To-Image Translation (1)

image with the same content but in a different style.12 Image →
facade labels →

![47_image_1.png](47_image_1.png)

black/white image →

![47_image_2.png](47_image_2.png)

house photo edges →

![47_image_0.png](47_image_0.png)

color image day photo

## E1.11: Image-To-Image Translation (2) Style Transfer13

analyze the content of a photo
•
analyze the painting style of a picture

•
•
combine the photo content with the painting style

•
available as Android App "Prisma"

![48_image_0.png](48_image_0.png)

## E1.12: Ai Helped To Finish Symphonies

Schubert Symphony No. 8:

Schubert composed only the first 2 movements.

•
A DNN of Huawei AI generated the melody for movement 3 and 4. Composer

•
Lucas Cantor orchestrated the melody.

Premiere in London on Feb 14, 2019: "If I wasn't told the third and fourth move-
•
ments were created by artificial intelligence, I wouldn't have known."

![49_image_0.png](49_image_0.png)

Beethoven Symphony No. 10:

Beethoven started the composition before death. Completed by AI (Telekom).

•
Premiere on Oct 9, 2021.

•

## E1.13: Speech And Language Processing

DNN also becomes the state-of-the-art tool for speech and language processing.

Speech: spoken text, sequence of phonemes 
•
text),14 e.g. Siri, Alexa speech recognition (speech
→
–
speech synthesis (text →
speech) 
–
speech translation, e.g. German English
→
–
speech enhancement (distorted speech clean speech) 
→
–
 . . .

•
Language: written text, sequence of letters language translation, e.g. German English
→
–
text understanding, e.g. news, messages, customer feedbacks 
–
language dialog, e.g. ChatGPT
–
–
 . . .

## E1.14: Alphago (1) Go

oldest board game (over 2,500 years) from China

- •
19x19 board, black and white stones simple rules, complex strategies

•
state-space complexity 319×19 ≈
10172 (1084∼89 atoms in cosmos)

![51_image_1.png](51_image_1.png)

•

![51_image_0.png](51_image_0.png)

## E1.14: Alphago (2)

based on deep learning from DeepMind15 AlphaGo learn from many human Go games (supervised **learning**)

•
calculate a probability map for the next best moves based on the current board
•
situation 2015: AlphaGo Fan beat a human professional Go player (Fan Hui)

•
2016: AlphaGo Lee beat a 9-dan Go player (Lee Sedol) with 4:1
•
2017: AlphaGo Master beat the world No. 1 (Ke Jie) with 3:0
•

![52_image_0.png](52_image_0.png)

## E1.14: Alphago (3) 2017: Alphago Zero16

learn from scratch by practice without knowing any human games: AlphaGo Zero
•
against AlphaGo (deep reinforcement **learning**)

AlphaGo Zero reached the level of AlphaGo Lee in 3 days (100:0 win)

•
AlphaGo Zero reached the level of AlphaGo Master in 21 days

•
unreachable by any humans

•
One month training time beats 2,500 years of human *experiences.*

Now AlphaGo is retired.

## E1.15: Chip Design

Google used deep reinforcement learning for chip floorplanning, designing the physical layout of a computer chip. It is done in a few hours instead of months by humans.

The result is superior or comparable to those produced by humans17.

## E1.16: Ai Discovered Patterns In Pure Mathematics

Mathematicians and AI experts have teamed up to demonstrate how machine learning can support proof of pure mathematical theorems18.

## E1.17: Sora From Openai

A text-to-video generative model based on diffusion transformer19: "Sora can generate videos up to a minute long while maintaining visual quality and adherence to the user's

```
prompt."
       
Prompt: "A stylish woman walks down a Tokyo street filled with warm glowing neon and animated
                                                                             

```

city signage. She wears a black leather jacket, a long red dress, and black boots, and carries a black purse. She wears sunglasses and red lipstick. She walks confidently and casually. The street is damp and reflective, creating a mirror effect of the colorful lights. Many pedestrians walk about."

# Summary Of Examples

| Example         | Task                      | Description                             | Output                              |
|-----------------|---------------------------|-----------------------------------------|-------------------------------------|
| E1.4            | classification            | estimate the class of an image          | class                               |
| E1.5            | semantic segmentation     | estimate the class of each pixel/sample | semantic map bounding box and class |
| E1.6            | object detection          | localize & classify objects             | class label                         |
| and instance ID |                           |                                         |                                     |
| E1.7            | panoptic segmentation     | semantic and instance segmentation      |                                     |
| E1.8            | regression                | estimate real-valued parameter          | number                              |
| E1.9            | classification            | estimate geo location                   | class                               |
| E1.10           | text-to-image translation | image                                   |                                     |
| E1.11           | translation               | image-to-image translation              | image                               |
| E1.12           | translation               |                                         |                                     |
|                 | generative model          | music composition                       | music                               |
| E1.13           | various                   | speech and language processing          | class/language/speech               |
| E1.14           | reinforcement learning    | playing Go                              | game                                |
| E1.15           | reinforcement learning    | chip design                             | chip layout                         |
| E1.16           | supervised learning       | proof                                   | theorem                             |
| E1.17           | generative model          | text-to-video                           | video                               |

# Introduction To Python

easy to learn and powerful, but not developed for ML/DL originally
•
runs on all major operating systems

•
many packages for machine learning, deep learning and scientific computing
•
SciPy/Numpy (basic linear algebra and optimization) 
–
scikit-learn (conventional machine learning) 
–
TensorFlow (deep learning), see "Introduction into TensorFlow" in this course 
–
PyTorch (deep learning)

–
the deep learning community uses Python
•
 you get it for free
•
MATLAB for signal processing, Python for deep learning.

This course also contains an integrated minilab consisting of programming practice: learn the basics of Python
•
assignments: use TensorFlow to solve practical problems

•

# Hardware For Deep Learning

## Research

CPU: Central processing unit

•
GPU: Graphics processing unit

•
enough memory for large datasets

•
GPU is much faster than CPU in computer graphics, image processing and deep learning due to pipelined matrix-vector calculations and parallel structures.

## Embedded Systems

µC: Microcontroller

•
special accelerator like TPU: Tensor processing unit (Google)
•
FPGA: Field programmable gate arrays

•
ASIC: Application-specific integrated circuit

•

## Education

Google Colab: Cloud computing without installation of Python on your computer.

•
You have (limited) access to Google GPUs. This will be used in assignments.

# Mnist Dataset1

a small labeled dataset for handwritten (by many people) digit recognition
•
70,000 clean gray images of pre-segmented handwritten digits

•
60,000 for training and 10,000 for test

•
fixed image size 28x28
•
all popular public datasets are ready for use in TensorFlow/PyTorch
•
(myselph.de/neuralNet.html)

# Fashion Mnist Dataset2

![59_Image_0.Png](59_Image_0.Png)

a dataset of Zalando's

•
article images as a direct replacement of MNIST
10 classes: T-shirt/top,

•
trouser, pullover, dress, coat, sandal, shirt,

```
sneaker, bag, ankle
                       
boot
   

```

70,000 pre-segmented
•
gray images, 60,000 for training and 10,000 for test fixed image size 28x28
•

# Svhn Dataset

a small labeled dataset for street view house number (SVHN) recognition3
•
color images of street view house numbers

•
73,257 for training and 26,032 for test

•
fixed image size 32x32
•

(agi.io/2018/01/31/getting-started-street-view-house-numbers-svhn-dataset/)

large variety of colors and brightness, more challenging than MNIST

![60_image_0.png](60_image_0.png)

•

# Cifar-10 And Cifar-100 Datasets

![61_Image_0.Png](61_Image_0.Png)

small labeled datasets for visual object recognition
•
60,000 color images of size 32x32, 50,000 for training and 10,000 for test

•
CIFAR-10: 10 object classes, 6000 images per class

•
CIFAR-100: 100 object classes, 600 images per class
•

# Imagenet Dataset

a large labeled dataset for visual object recognition
•
over 14 million color images of objects

•
over 20,000 classes

•
varying image sizes (average image size 469x387)

•
www.image-net.org
•

![62_image_0.png](62_image_0.png)

# Tiny Imagenet Dataset

a miniature of ImageNet for low computing power

•
120,000 color images of size 64x64
•
200 classes

•
500/50/50 for training/validation/test images per class
•

# Cityscapes Dataset

a large labeled dataset of street images for semantic image segmentation
•
50 cities and 30 object classes from 8 object groups (flat, human, vehicle, construc-
•
tion, object, nature, sky, void)

5,000/20,000 labeled images with fine/coarse annotations

- •
large image size

•
www.cityscapes-dataset.com

![64_image_0.png](64_image_0.png)

# More Datasets

Computer vision COCO: Common Objects in Context

- •
 . . .

Autonomous driving A2D2: Audi Autonomous Driving Dataset

•
KITTI: KITTI Vision Benchmark Suite

- •
nuScenes Waymo Open Dataset

- •
 . . .

Biomedical UK Biobank: imaging data, blood data, genetics, . . .

•
•
 . . .

# Which Hardware For Which Datasets?

MNIST, Fashion MNIST, SVHN, CIFAR-10, CIFAR-100, tiny ImageNet: CPU is

•
sufficient.

ImageNet, Cityscapes: You need a GPU.

•
The Python code is the same for CPU or CPU+GPU except for an one-line change in a configuration file.

•

# Vector And Matrix

R: **scalar**

∈
•
x RM =
RM×1: column of length M. The notation
[xi] means that x vector
∈
•
x x
=
consists of the elements xi, 1 ≤ i ≤ M.

RM×N:
X
 **matrix** of dimensions M
N. The notation X
[xi j] means that X
∈
×
•
= 
consists of the elements xi j, 1 ≤ i ≤ M, 1 ≤ j ≤ N.

xT or XT :
 transpose of x or X
•
[xi jyi j] is the elementwise multiplication (Hadamard **product**) of two X
Y
⊙
•
=
matrices and of the same dimensions.

X
Y
The vectorization operator vec stacks all columns of a matrix into a single col-
X
•
umn vector:

$$\operatorname{vec}(\mathbf{X})={\left[\begin{array}{l l}{{\mathrm{first~column~of}\ \mathbf{X}}}\\ {\vdots}\\ {{\mathrm{last~column~of}\ \mathbf{X}}}\end{array}\right]}.$$

# Tensor

Vector and matrix are one- or two-dimensional arrays of numbers. An extension to multi-dimensional arrays is called tensor. The dimensionality is called the **order** of the tensor:

x scalar or 0th-order tensor

$$\begin{array}{l}{{\begin{array}{l}{{x=[x_{i}]\in\mathbb{R}^{M}}}\\ {{\mathbf{X}=[x_{i j}]\in\mathbb{R}^{M\times N}}}\\ {{\mathbf{X}=[x_{i j k}]\in\mathbb{R}^{L\times M\times N}}}\end{array}}}\end{array}$$

[xi] ∈ RM vector or 1st-order tensor with the dimension M
[xi j] ∈ RM×N matrix or 2nd-order tensor with the dimensions M, N 
[xi jk] ∈ RL×M×N 3rd-order tensor with the dimensions L, M, N

. . . . . .

Tensors are frequently used in convolutional neural networks, see ch. 7.

## Vector Norms

Given a vector x = [xi] ∈ RM. There are different definitions of the vector norm:
or Euclidean **norm**: kxk2 = qPMi=1 x2i = pxT x or l2-norm 2-norm
•
or l1**-norm**: kxk1 = PMi=1 |xi| 1-norm
•
0-norm or l0**-norm**: kxk0 = number of non-zero elements in x
•
Comments:

kxk22 represents the energy of x.

•
kxk0 measures the sparsity of x.

•
Different vector norms have different unit-norm contour **lines** {x | kxkp = 1}.

•

![69_image_0.png](69_image_0.png)


$p=1:|x_{1}|+|x_{2}|=1\quad p=0:$ one of $x_{1},x_{2}$ non-zero.  

$$p=2:x_{1}^{2}+x_{2}^{2}=1$$
p = 2 : x21 + x22 = 1 p = 1 : |x1| + |x2| = 1 p = 0 : one of x1, x2 non-zero
# Vector Derivative

Let y = [yi] ∈ RM be a vector function of x = [xi] ∈ RN. The derivative of y(x) w.r.t. x is defined as

$$\mathbf{J}={\frac{\partial y}{\partial{\underline{{x}}}}}=\left[{\frac{\partial y_{i}}{\partial x_{j}}}\right]_{i j}={\left[\begin{array}{l l l}{{\frac{\partial y_{1}}{\partial x_{1}}}}&{\cdot\cdot\cdot}&{{\frac{\partial y_{1}}{\partial x_{N}}}}\\ {\vdots}&{\cdot\cdot\cdot}&{\vdots}\\ {{\frac{\partial y_{M}}{\partial x_{1}}}}&{\cdot\cdot\cdot}&{{\frac{\partial y_{M}}{\partial x_{N}}}}\end{array}\right]}\in\mathbb{R}^{M\times N}.$$
It is called the Jacobi **matrix**. Clearly, the derivative of a vector y(x) ∈ RM w.r.t. a


 

 
 
scalar number x is a column vector

$$\frac{\partial y}{\partial x}=\left[\begin{array}{c}{{\frac{\partial y_{1}}{\partial x}}}\\ {{\vdots}}\\ {{\frac{\partial y_{M}}{\partial x}}}\end{array}\right].$$

 y(x) w.r.t. a vector x ∈ RN is a row vector h ∂y ∂x1 But the derivative of a scalar function
 , . . . , ∂y ∂xN i. It is the transpose of the gradient **vector** of y(x)

$$\nabla y=$$

$$\left[\begin{array}{c}{{\frac{\partial y}{\partial x_{1}}}}\\ {{\vdots}}\\ {{\frac{\partial y}{\partial x_{N}}}}\end{array}\right]=\left(\frac{\partial y}{\partial x}\right)^{T}.$$

## Chain Rule And Product Rule Of Derivative

Chain **rule**: Let L(x(y(θ))) ∈ R be a function of x that is a function of y that is a function of θ ∈ R. Then∂L

$$3{-}5$$
$${\frac{\partial L}{\partial\theta}}={\frac{\partial L}{\partial{\underline{{x}}}}}\cdot{\frac{\partial x}{\partial{\underline{{y}}}}}\cdot{\frac{\partial y}{\partial\theta}}$$

It plays an important role in backpropagation for the training of a neural network, see ch. 5.

$$\mathrm{Product~rule}{\mathrm{:}}$$
Product rule:  $$\frac{\partial\left(x(\theta)y(\theta)\right)}{\partial\theta}=\frac{\partial x(\theta)}{\partial\theta}y(\theta)+x(\theta)\frac{\partial y(\theta)}{\partial\theta}$$  This will be used in backpropagation through time, see ch. 8.  

 

 

 
+

## One Random Vector

Let X
[Xi] be a real-valued random vector containing d random variables

(RV).

=
The distribution
(PMF or PDF) describes completely the statistical properties of X.

| discrete-valued           | continuous-valued   |                                   |        |     |      |
|---------------------------|---------------------|-----------------------------------|--------|-----|------|
| value                     | X∈ { x1             | }                                 |        |     |      |
| , x 2, . . .              | X                   | Rd                                |        |     |      |
| ∈                         | )                   |                                   |        |     |      |
| distribution                           | (PMF)               | probability density function (PDF |        |     |      |
| probability mass function | P( x                | X≤                                | + ∆ x) |     |      |
| <                         | x                   |                                   |        |     |      |
| X                         | P(X                 | )                                 |        |     |      |
| xi                        | =P(                 | )                                 |        |     |      |
| xi                        | =pi ≥0              | X                                 | ( x) = | lim | ≥0   |
| ∼                         | =                   | ∼p                                | ∆      | 0   | |∆x| |
| x →                       |                     |                                   |        |     |      |
| normali                           | R p                 |                                   |        |     |      |
| zation                    | P i pi =            | 1                                 | ( x)d  | 1   |      |
| x=                        |                     |                                   |        |     |      |
| notation                  | (·)                 | p (·)                             |        |     |      |
| P                         |                     |                                   |        |     |      |
| P( xi)                    | ( x)                |                                   |        |     |      |
| p                         |                     |                                   |        |     |      |
| x                         | x                   |                                   |        |     |      |

3.2 Random variable and probability distribution 3-7

## E3.1: Pdf And Cumulative Distribution Function (Cdf)

a) Uniform **distribution** in [
a, b]:

![73_image_0.png](73_image_0.png)

$\mathbf{\ddot{a}}\mathbf{\ddot{b}}\mathbf{n}\!\!+\!\!\mathbf{\ddot{a}}\mathbf{\ddot{c}}\mathbf{\ddot{a}}\mathbf{\ddot{c}}\mathbf{\ddot{a}}\mathbf{\ddot{c}}$

$$\begin{array}{l l}{{a\leq x\leq b}}\\ {{\mathrm{otherwise}}}\\ {{\left\{\begin{array}{l l}{{0}}&{{\mathrm{for}}}\\ {{x-a}}&{{\mathrm{for}}}\\ {{\overline{{{b-a}}}}}&{{\mathrm{for}}}\end{array}\right.\;\;a\leq x\leq b}}\\ {{1}}&{{\mathrm{for}}}\end{array}.$$

![73_image_1.png](73_image_1.png)

![73_image_2.png](73_image_2.png)

(µ, σ 2):

b) Normal (Gaussian) **distribution**

N

![73_image_3.png](73_image_3.png)

# Moments Of One Random Vector

of X
•
mean

$$\underline{{{\mu}}}=\mathrm{E}(\underline{{{X}}})=\int\underline{{{x}}}p(\underline{{{x}}})\mathrm{d}\underline{{{x}}}\stackrel{\mathrm{d.-V.}}{=}\sum_{i}\underline{{{x}}}_{i}P(\underline{{{x}}}_{i})\in\mathbb{R}^{d}$$


 


 


 

+
˙
E(

) is the **expectation**, a statistical average over all random realizations.

of correlation matrix X
•

$$\mathbf{R}=\operatorname{E}(X{\underline{{X}}}^{T})=\int{\underline{{x}}}\,{\underline{{x}}}^{T}p({\underline{{x}}})\mathrm{d}{\underline{{x}}}\,{\stackrel{\mathrm{d.-v.}}{=}}\sum_{i}{\underline{{x}}}_{i}{\underline{{x}}}_{i}^{T}P({\underline{{x}}}_{i})\in\mathbb{R}^{d\times d}$$

$$\bullet{\mathrm{~covariance~matrix~of~}}X$$
$$\begin{array}{r c l}{{\mathbf{C}}}&{{=}}&{{\mathrm{E}((\underline{{{X}}}-\underline{{{\mu}}})(\underline{{{X}}}-\underline{{{\mu}}})^{T})=\int(\underline{{{x}}}-\underline{{{\mu}}})(\underline{{{x}}}-\underline{{{\mu}}})^{T}p(\underline{{{x}}})\mathrm{d}\underline{{{x}}}}}\\ {{}}&{{}}&{{\mathrm{d.-v.}}}\\ {{}}&{{}}&{{\sum_{i}(\underline{{{x}}}_{i}-\underline{{{\mu}}})(\underline{{{x}}}_{i}-\underline{{{\mu}}})^{T}P(\underline{{{x}}}_{i})=\mathbf{R}-\underline{{{\mu}}}\underline{{{\mu}}}^{T}\in\mathbb{R}^{d\times d}}}\end{array}$$

In general, **moments** are only a partial description of the statistical properties of X.

# Multivariate Normal (Gaussian) Distribution

PDF

$$\underline{X}\in\mathbb{R}^{d}\ \sim\ N(\underline{\mu},\mathbf{C})$$ $$p(\underline{x})=\frac{1}{(2\pi)^{d/2}|\mathbf{C}|^{1/2}}e^{-\frac{1}{2}(\underline{x}-\underline{\mu})^{T}\mathbf{C}^{-1}(\underline{x}-\underline{\mu})}$$ $$\ln(p(\underline{x}))=-\frac{d}{2}\ln(2\pi)-\frac{1}{2}\ln(|\mathbf{C}|)-\frac{1}{2}(\underline{x}-\underline{\mu})^{T}\mathbf{C}^{-1}(\underline{x}-\underline{\mu})$$  $N(0,\mathbf{I})$ is called the standard normal distribution. $\mathbf{I}$ is the identity matrix.  


1D-Visualization

![75_image_0.png](75_image_0.png) 
Moments

$$\mathrm{E}(\underline{{{X}}})=\underline{{{\mu}}},\qquad\mathrm{Cov}(\underline{{{X}}})=\mathbf{C}$$

# Laplace Distribution

PDF

$$\begin{array}{r l}{X\in\mathbb{R}\;\sim\;\mathrm{Laplace}(\mu,b),}&{b>0}\\ {p(x)\;=\;{\frac{1}{2b}}e^{-{\frac{|x-\mu|}{b}}}}\\ {\ln(p(x))\;=\;-\ln(2b)-{\frac{1}{b}}|x-\mu|}\end{array}$$

Visualization

![76_image_0.png](76_image_0.png) 
Moments

$$\operatorname{E}(X)=\mu,\qquad\operatorname{Var}(X)=2b^{2}$$

# Bernoulli Distribution

for a binary random variable, e.g. flip a coin.

PMF

$X\in\{0,1\}\ \sim\ \mbox{Bernoulli}(p),\quad p=P(X=1)>0$  $$P(x)\ =\ \left\{\begin{array}{ll}1-p&\mbox{for}\quad x=0\\ p&\mbox{for}\quad x=1\end{array}\right.=p^{x}(1-p)^{1-x}$$  $$\ln(P(x))\ =\ x\ln(p)+(1-x)\ln(1-p)$$

 

 

 

 

Visualization

![77_image_0.png](77_image_0.png) 
Moments E(X)

=

p, Var(X)

$$\operatorname{Var}(X)=p(1-p)$$

3.2 Random variable and probability distribution 3-12

Categorical distribution $\mathbf{or}$ multinoulli distribution . 

 

 


 
for a discrete multi-state random variable, e.g. roll a die.

PMF

$$X\in\{1,2,\ldots,c\}\ \sim\ \mathrm{Cat}(\underline{{{p}}}),\quad P(X=i)=P(i)=p_{i},1\leq i\leq c$$

$$\underline{{{p}}}\,=\,[p_{1},p_{2},\ldots,p_{c}]^{T},\quad\sum_{i=1}^{c}p_{i}=\underline{{{1}}}^{T}\underline{{{p}}}=1$$

Visualization

$$\bigwedge^{}P(i)$$

![78_image_0.png](78_image_0.png)

Moments

$$\mathrm{E}(X)=\sum_{i=1}^{c}i p_{i},\qquad\mathrm{Var}(X)=\mathrm{E}(X^{2})-(\mathrm{E}X)^{2}=\sum_{i=1}^{c}i^{2}p_{i}-(\mathrm{E}X)^{2}$$

# One-Hot Coding

One-hot coding is a group of zero bits "0" except for a single "1", e.g. 01000.

For a c-class classification, there are two ways to represent the class label:

Scalar integer coding: The class label is a random variable y ∈ {
1, 2, . . . ,
c
}.

•
Vectorial one-hot coding: The class label is a random vector y ∈ {
e1 c

}

- 
, e 2, . . . ,
e where ei is the i-th unit vector, i.e., the i-th column of the identity matrix
×
c c I 
c]. This means, all elements of y are "0" except for a single "1".

The position of "1" is the class index.

[e1 
, e 2, . . . ,
e
=

## E3.2: One-Hot Coding

For the MNIST digit recognition, there are 10 classes, the digits 0, 1, . . . , 9. The onehot coding of these 10 classes consists of the following 10 unit vectors:

$y=\underline{e}_{1}=[1,0,0,\ldots,0,0]^{T}$ for digit "0" $y=\underline{e}_{2}=[0,1,0,\ldots,0,0]^{T}$ for digit "1" $\vdots$ $\vdots$ $y=\underline{e}_{10}=[0,0,0,\ldots,0,1]^{T}$ for digit "9"

 

 

 

+
The one-hot encoding seems to be more complicated than the integer encoding, but it enables later an elegant expression for the PMF and the categorical cross-entropy loss, see next section.

# Di Fferent Distributions Of X And Y

Continuous-valued RVs, PDF
Discrete-valued RVs, PMF

![81_image_0.png](81_image_0.png)

## Chain Rule Of Probability

The chain rule of probability formulates a joint distribution in terms of a number of conditional distributions. It is the generalization of the product rule:

 
 
$$p(\underline{{{x}}}_{1},\ldots,\underline{{{x}}}_{N})=p(\underline{{{x}}}_{1}|\underline{{{x}}}_{2},\ldots,\underline{{{x}}}_{N})p(\underline{{{x}}}_{2}|\underline{{{x}}}_{3},\ldots,\underline{{{x}}}_{N})\cdots p(\underline{{{x}}}_{N-1}|\underline{{{x}}}_{N})p(\underline{{{x}}}_{N}).$$

2, . . . ,

3, . . . ,

+
Proof:

$$\begin{array}{r l}{p(\underline{{{x}}}_{1},\ldots,\underline{{{x}}}_{N})\;=\;p(\underline{{{x}}}_{1}|\underline{{{x}}}_{2},\ldots,\underline{{{x}}}_{N})p(\underline{{{x}}}_{2},\ldots,\underline{{{x}}}_{N}),}\\ {p(\underline{{{x}}}_{2},\ldots,\underline{{{x}}}_{N})\;=\;p(\underline{{{x}}}_{2}|\underline{{{x}}}_{3},\ldots,\underline{{{x}}}_{N})p(\underline{{{x}}}_{3},\ldots,\underline{{{x}}}_{N}),}\\ {\vdots}\end{array}$$

$$p(\underline{{{x}}}_{N-1},\underline{{{x}}}_{N})\;=\;p(\underline{{{x}}}_{N-1}|\underline{{{x}}}_{N})p(\underline{{{x}}}_{N}).$$

The order of conditioning can be arbitrary, e.g.

$$\begin{array}{l}{{p(\underline{{{x}}}_{1},\ldots,\underline{{{x}}}_{N})\ =\ p(\underline{{{x}}}_{N}|\underline{{{x}}}_{N-1},\ldots,\underline{{{x}}}_{1})p(\underline{{{x}}}_{N-1}|\underline{{{x}}}_{N-2},\ldots,\underline{{{x}}}_{1})\cdots p(\underline{{{x}}}_{2}|\underline{{{x}}}_{1})p(\underline{{{x}}}_{1}),}}\\ {{p(\underline{{{x}}}_{1},\ldots,\underline{{{x}}}_{N})\ =\ p(\underline{{{x}}}_{3}|\underline{{{x}}}_{1},\underline{{{x}}}_{2},\underline{{{x}}}_{4},\ldots,\underline{{{x}}}_{N})\ldots}}\end{array}$$

## E3.3: Estimate Of Pdf And Cdf

(

n
), 1 N
are i.i.d. samples drawn from
(

x
)

∼
N(0, 1). They are used to
≤
≤
x n p 1 N
P
N

```
Fˆ

 (
 
 x

```

calculate the empirical PDF ˆp
(

x
)

=
δ
(

x
(

n)) and the empirical CDF
)

=
x
−

n

=1

```
                                                       
R
 x
 
−∞

```

P
N
1 N

```
pˆ

 (z)dz
   

```

(

x
(

n)).

u
(

x) is the unit step function.

u x
=
−

n

=1

![83_image_1.png](83_image_1.png) 

![83_image_0.png](83_image_0.png)

The PDFs of the true and empirical distribution look quite di fferent, but the CDFs

•
agree well for large N.

We do not need a smooth PDF estimate (using e.g. the Gaussian kernel). Later

•
in this chapter, the empirical distribution is used to approximate the unknown true distribution
(

x).

p

# ⋆ Proof Of Dkl (P|| Q ) 0 ∀P, Q ≥

Let f( x) be a convex **function**. According to the Jensen's **inequality**,

$$f\left(\sum_{i=1}^{N}\lambda_{i}\underline{{{x}}}_{i}\right)\leq\sum_{i=1}^{N}\lambda_{i}f(\underline{{{x}}}_{i})$$


 

+
# The curve of the function $f\left(\sum_{i=1}^{N}\lambda_{i}x_{i}\right)$ is M. 
$\forall\underline{x}_{i}$ and $\lambda_{i}\geq0$ with $\sum_{i=1}^{N}\lambda_{i}=1$. This means, the curve of the function $f\left(\sum_{i=1}^{N}\lambda_{i}\underline{x}_{i}\right)$ below the secant line $\sum_{i=1}^{N}\lambda_{i}f(\underline{x}_{i})$, see course AM.  

- ln($z$) is a convex function. Hence${}^{1}$

$$D_{\rm KL}(P\|Q)=\sum_{i}P(\underline{x}_{i})\ln\left(\frac{P(\underline{x}_{i})}{Q(\underline{x}_{i})}\right)=\sum_{i}\underbrace{P(\underline{x}_{i})}_{\lambda_{i}}\left[-\frac{\ln}{f}\left(\frac{Q(\underline{x}_{i})}{\underbrace{P(\underline{x}_{i})}_{\underline{x}_{i}}}\right)\right]$$ $$\geq-\ln\left(\sum_{i}P(\underline{x}_{i})\frac{Q(\underline{x}_{i})}{P(\underline{x}_{i})}\right)=-\ln(1)=0.$$

| Forward KL divergence DKL(p|| q )                | Backward KL divergence DKL( q|| p )   |                                 |      |
|--------------------------------------------------|---------------------------------------|---------------------------------|------|
| p ( x) !                                         | q ( x) ln  q ( x) ! )                 |                                 |      |
| ( x) ln                                          |                                       |                                 |      |
| p                                                | ( x)                                  | ( x                             |      |
| q                                                | p                                     |                                 |      |
| 0                                                | 0                                     |                                 |      |
| p >q                                             | >                                     | <                               |      |
| →0                                               | → ∞                                   | →0                              |      |
| q                                                | 0                                     | 0                               |      |
| p →                                              | → ∞                                   |                                 |      |
| Minimize→DKL(p|| q )                             | Minimize DKL( q|| p )                 |                                 |      |
| 0: doesn't care about q                          | 0: doesn't care about p               |                                 |      |
| p =                                              | q =                                   |                                 |      |
| 0: make q close to                               | 0: make q close to p                  |                                 |      |
| "zero avoiding" strategy for q:                  | "zero forcing" strategy for q:        |                                 |      |
| p >                                              | p                                     | q >                             |      |
| 0 if p                                           | 0                                     | 0 if p                          | 0    |
| q >                                              | >                                     | q =                             | =    |
| i.e. makes q ( x) broader than                   | ( x)                                  | i.e. makes q ( x) narrower than | ( x) |
| p                                                | p                                     |                                 |      |
| Make denominator in ln(·) broad to minimize DKL. |                                       |                                 |      |

## E3.4: Forward Vs. Backward Kl Divergence

![86_image_0.png](86_image_0.png) 

When minimizing
(p|| q), a) is better than b) because q

(

x) is broad.

DKL
•
When minimizing

```
(

 q||
   p), b) is better than a) because
                                               
                                                  q
                                                  

```

(

x) is narrow.

DKL
•
3.3 Kullback-Leibler divergence and cross entropy 3-21 E3.5: KL divergence between normal and Laplace distributio n

![87_image_0.png](87_image_0.png) 

# Proof Of Additivity

If $\underline{X}=\left[\begin{array}{c}\underline{X}_{1}\\ \underline{X}_{2}\end{array}\right]$ with independent $\underline{X}_{1}$ and $\underline{X}_{2}$, i.e.,  $$p(\underline{x})=p_{1}(\underline{x}_{1})p_{2}(\underline{x}_{2})\quad\mbox{and}\quad q(\underline{x})=q_{1}(\underline{x}_{1})q_{2}(\underline{x}_{2}),$$

 

 


+
then

 p ( x ) !d x Zp ( x) ln DKL (p|| q ) = ( x ) q  p1( x1 ) p 2 ( x 2 ) !d x1 ZZp1( x1 ) p 2 ( x 2) ln d x = 2 q1( x1 ) q 2 ( x 2 )  p1( x1 )  p 2 ( x 2 ) !d x1 !d x1 ZZp1( x1 ZZp1( x1 ) p 2 ( x 2) ln d ) p 2 ( x 2) ln d + x x = 2 2 q1( x1 ) q 2 ( x 2 )  p1( x1 ) !d x1  p 2 ( x 2 ) !d x Zp1( x1) ln Zp 2 ( x 2) ln + = 2 q1( x1 ) q 2 ( x 2 ) DKL (p1|| q1) + DKL (p 2|| q 2 ). =

This implies that for independent samples x
(

n), we can simply add the KLD of x
(

n).

## Entropy And Cross Entropy

In information theory, the **entropy**

H(X) of a random variable X
is a measure for its average information in the sense of uncertainty. The larger the uncertainty, the larger the information content.

and the PMF
If pi is a discrete random variable with the possible values xi

, 1 X
i

≤
M
≤
∼

```
pi, its entropy is
          
           2
           

```

(X
xi)

=
P
=

 

 

$$H(X)=H(P)=-\mathrm{E}(\ln(P(X)))=-\sum_{i=1}^{M}p_{i}\ln(p_{i})\ [\mathrm{nats}]\geq0.$$

 

+
If X
(

x) is a continuous random variable with the PDF
(

x), its (di fferential) enp p

∼
tropy is

$$H(X)=H(p)=-\mathrm{E}(\ln(p(X)))=-\int_{-\infty}^{\infty}p(x)\ln(p(x))\mathrm{d}x\ [\mathrm{nats}]\geq0.$$

```
For two di
        
        fferent distributions
                         

```

(

x) and
(y), the cross **entropy**

(CE) between X
Y
p q

∼
∼
them is defined as

$$H(X,Y)=H(p,q)=-\mathrm{E}_{X\sim p}\ln(q(X))=-\int_{-\infty}^{\infty}p(x)\ln(q(x))\mathrm{d}x\geq0.$$

2In information theory, the base-2 logarithm log2 is used resulting in the entropy unit "bits". In machine learning, the natural logarithm ln
= loge is preferred leading to the unit
"nats".

# Probabilistic Framework Of Supervised Learning

![90_image_0.png](90_image_0.png) 

y: latent variable, ground **truth**, hidden, not measurable, quantity of interest

•
x: observed **variable**, measurement, input for DNN
•

```

 yˆ: output of DNN as estimate for
                          
                           y
                           
•

```

•
real world: how is x generated from
?

y DNN: how to estimate y from
?

•
x

## The Data Generating Distribution

Both y and are modeled as random variables. They are described by the joint **data**

x generating distribution

 

$$p(\underline{{{x}}},\underline{{{y}}})=p(\underline{{{x}}}|\underline{{{y}}})p(\underline{{{y}}})=p(\underline{{{y}}}|\underline{{{x}}})p(\underline{{{x}}}).$$

 


 

 

 
 
+
p

(y) prior PDF of y or **prior**, available before any measurement of x
(

x|y p

) **likelihood**. It describes the real world, the generation of x from y.

It is a kind of channel-sensor model, e.g.,

bit: communication channel

+
receiver
•
microphone

•
speech: speech production system
+
•
digit: handwriting
+
camera p

(

x) prior PDF of x, also called evidence
(y|x) posterior PDF of y after a measurement x or **posterior**

p

# ⋆ Bayesian Decision/Estimation Theory

In the model-based Bayesian decision/estimation **theory**, we assume to know the

```
measurement
          
           x,
            

```

•
likelihood
(

x|y),

•
p

```
prior
    
     p
     

```

(y),

•
loss values

```
l

 (y, yˆ
     
       (
       
        x)) for a correct decision

yˆ or a wrong one
            
             y
             

```

- 
y

=


 
$$\mathrm{one}\;y\neq\hat{y}.$$

 

+
There are di fferent ways to estimate y from x:

a) Maximum **likelihood**

(ML): max
(

x|y

)

p b) Maximum a **posterior**

(MAP): max

$$\begin{array}{c}{{\underline{{{y}}}}}\\ {{\operatorname*{max}_{\underline{{{y}}}}p(\underline{{{y}}}|\underline{{{x}}})=p(\underline{{{x}}}|\underline{{{y}}})\frac{p(\underline{{{y}}})}{p(\underline{{{x}}})}}}\end{array}$$

c) Minimum Bayesian **risk**

(MBR):

$$\operatorname*{min}_{\hat{y}}\operatorname{E}_{\underline{{{X}}},\underline{{{Y}}}}[l(\underline{{{Y}}},\hat{y}(\underline{{{X}}}))]=\int l(\underline{{{y}}},\hat{y}(\underline{{{x}}}))p(\underline{{{x}}},\underline{{{y}}})\mathrm{d}\underline{{{x}}}\mathrm{d}\underline{{{y}}}$$

 # $\bullet$ for more details . 

see course DPR and SASP for more details.

# 1D Digital Filter

![93_image_0.png](93_image_0.png)

A general 1D digital filter IIR(

N, M) is described by the di fference equation

$$y(n)=\sum_{i=0}^{M}b_{i}x(n-i)-\sum_{i=1}^{N}a_{i}y(n-i).$$

Two special cases are relevant for deep learning:

3
: ai =
a) Non-recursive **filter** IIR(0, M) or FIR **filter**

0
∀
i

$y(n)=b_{0}x(n)+b_{1}x(n-1)+\ldots+b_{M}x(n-M)$.  

. . .

is a **convolution** between the input x

```
(

 n) and the impulse response or kernel
                                                             bi
                                                               
                                                                .
                                                                

```

This will be used in convolutional neural networks.

b) Recursive filter or autoregressive filter or IIR **filter** IIR(
N, 0):

bi =
0 1
∀
i

≥

$$y(n)=b_{0}x(n)-[a_{1}y(n-1)+\ldots+a_{N}y(n-N)].$$

. . .

This will be used in recurrent neural networks, momentum method, batch normalization etc.

See my Bachelor course "Digital signal processing".

3FIR: finite (duration) impulse response IIR: infinite (duration) impulse response

# 2D Digital Filter

( n1, n 2 ) x h ( n1, n

$$\overline{{)\left[\begin{array}{l}{y(n_{1},n_{2})}\\ {\end{array}\right]}}$$

 

 

# 211 size $K\times K=3\times3$ ? 

 


![94_image_0.png](94_image_0.png)

Similarly, for a 2D digital filter with the input image

(

n1, n 2) and kernel (or **point**

x spread function in image processing)

h

```
(

 n1, n
     
       2), the output image
                                   
                                     y
                                      

```

(

n1, n 2) is given by the 2D convolution

$$y(n_{1},n_{2})=\sum_{i_{1}=0}^{K-1}\sum_{i_{2}=0}^{K-1}h(i_{1},i_{2})x(n_{1}-i_{1},n_{2}-i_{2}).$$

Typically, the kernel h
(

n1, n 2) has a pretty small size K
K
3 3 or 5 5 or . . .

×
×
×
=
3.5 Some concepts from digital signal processing 3-29

## E3.6: 2D Convolution For Feature Extraction

Di fferent 2D kernels can be used to extract di fferent features.

![95_image_0.png](95_image_0.png) 

This will be the basic operation in convolutional neural networks. The di fference is that the kernels are not human-designed, rather learned from data.

# Down- And Upsampling

In signal processing, a signal x
(

n) by a factor m denotes the process of reducing downsampling N
∈
•
the sampling frequency by m. It is also called and is done by selecting decimation m-the value of x
(

n):

every


 

 

 

$\ldots,x(-m),x(0),x(m),x(2m),\ldots$

), . . .
. . . ,


 

 

 

 

 

 
upsampling a signal x
(

n) by a factor m N
denotes the process of increasing the

∈
•
sampling frequency by m. It is also called interpolation and is done by inserting (

m 1) zeros between each pair of adjacent samples of x
(

n
)

–
−

$$\ldots,x(-1),\underbrace{0,\ldots,0}_{m-1},x(0),\underbrace{0,\ldots,0}_{m-1},x(1),\ldots$$

$${\mathrm{ering~to~smooth~the~zero-inserted~sequence}}$$

and lowpass filtering to smooth the zero-inserted sequence

–
In convolutional neural networks, there are also down- and upsampling layers.

## Correlation

N
Correlation: Similarity between two vectors x
[xi], y

[yi] ∈

R
 measured by
•
=
=
a (scaled) dot product

 


$$\alpha{\underline{{x}}}^{T}{\underline{{y}}}=\alpha\sum_{i=1}^{N}x_{i}y_{i}\quad\mathrm{with}\quad\alpha=1,\quad{\frac{1}{N}},\quad{\overline{{\parallel}}}$$

 

 

+

![97_image_0.png](97_image_0.png)

Cross-correlation: Find a short known signal y

(

n) in a long measurement x
(

n) by
•
correlation, e.g. for synchronization Autocorrelation: Find self-similarity between di fferent segments of the same

(

n
)

•
x cross-correlation autocorrelation

![97_image_1.png](97_image_1.png)

![97_image_2.png](97_image_2.png)

The basic idea will be also used in attention and transformer.

# Why Can We Human Learn So Effectively?

Because we have a powerful brain.

It is a biochemical and electrical network.

•
It consists of a huge number of neurons (about 100 billions).

•
The neurons are massively connected (one to several thousand in average).

•
Each neuron shows a nonlinear input-output behavior: The neuron fires only if the input excitation is beyond a certain level.1
•
Our brain is able to learn, generalize and adapt, though we haven't fully understood its mechanisms yet.

![98_image_0.png](98_image_0.png)

# Why Nonlinear Behavior Of A Neuron?

Linear **tasks**: The desired input-output mapping can be well described by a linear function y = Ax or an affine **function** y = Ax + b.

Regression: The output is an affine function of the input.

•
Classification: The decision boundary is described by an affine function (hyper-
•
plane).

![99_image_0.png](99_image_0.png)

In practice, most tasks are **nonlinear**.

## E4.1: Nonlinear Tasks

a) Nonlinear regression

![100_image_0.png](100_image_0.png) 

b) Nonlinear classification By drawing a straight line (linear decision boundary) in a two-dimensional feature space, it is impossible to separat both classes ω1 and ω2 in the following tasks.

![100_image_1.png](100_image_1.png)

# A Feedforward Multilayer Neural Network

often called multilayer perceptron
(MLP)

![101_image_0.png](101_image_0.png)

The input layer is not really a layer. It just provides the input x0 without any
•
calculations.

is the number of layers. The number of hidden layers is L − 1.

L
•
Each dense layer l is characterized by its weight matrix Wl, bias vector bl and activation function
•
φl, 1 l ≤
L.

≤

## Nonlinear Activation Function And Hidden Layers

A feedforward multilayer neural network is an extension of a linear neuron by using nonlinear activation functions and
•
hidden layers.

•
The joint use of nonlinear activation functions and hidden layers is important:

for all neurons, each layer output xl is an affine function Wlxl−1 + bl of the layer input xl−1. As a consequence, the network output xL
If φ(a) =
•
a is an affine function of the network input x0 and all hidden layers are useless and can be removed.

If there are no hidden layers, the network output is x1 b1). Even if φ(W1x0
+
•
=
 φ(a) is nonlinear, this network is not able to approximate a general nonlinear input-output mapping. Often φ(a) is monotonically increasing and is useless when maximizing the network output in classification tasks.

# Different Activation Functions

1) Linear or **identity**

φ(a) =
a, i.e. no activation function.

It is used in the output layer for regression.

Such a neuron is often called a linear **neuron**.

![103_image_0.png](103_image_0.png)

2) Unit **step**

 

$$\phi(a)=u(a)=\left\{\begin{array}{l l}{{1}}&{{a>0}}\\ {{0}}&{{a<0}}\end{array}\right.$$
The neuron fires or not.


+
$$3)\ \mathrm{Sign~function}$$
3) sign function  $\phi(a)=\mbox{sign}(a)=\left\{\begin{array}{ccc}1&\mbox{for}&a>0\\ -1&\mbox{for}&a<0\end{array}\right.=2u(a)-1.$  As 2), just another output range.  

![103_image_2.png](103_image_2.png)

![103_image_1.png](103_image_1.png)

# Variants Of Relu

7) **Softplus**

φ(a) =
ln(1 ea)

+
A smooth approximation to ReLU.

8) Exponential linear unit **(ELU)**

( 
for

$$\mathbf{\partial}^{\mu}$$

 a ≥ 0 a φ(a) =
1) for a < 0 .

Nonzero gradient for a < 0.

α(ea −

![104_image_0.png](104_image_0.png)


 
 
 
+
9) Leaky ReLU
( 
 a ≥ 0 0.01a for a φ(a) =
 a < 0 .

Nonzero gradient for a < 0.

$\hdots\;0$). 
for 10) Parametric ReLU **(PReLU)**

Like leaky ReLU, but the slope for a < 0 is a learnable parameter (instead of 0.01).

Hence different neurons or layers have individual learned slope parameters.

## E4.2: Softmax Activation Function

The softmax activation function is used in the output layer of a DNN for classification:

φ(aL) =
softmax(aL). It is used to transform Rc, the so called logits, to
∈
xL
aL
c
=
probability values for c classes.

![105_image_0.png](105_image_0.png) 
The definition of softmax is φ(a) =

[1.8, −2.7, 3.1, 0.5]T for a 4-class classification task, then Given the logits aL
=


 

 

 
$\Phi$



+
$$\operatorname{softmax}(\underline{{{a}}}_{L})=[0.2019,0.0022,0.7408,0.0550]^{T}$$
The class i =
3 with the max. probability 74.08% is the predicted class.

## Activation Function: When, Where, Which?

when: regression or classification?

- •
where: hidden or output layer?

•
which: which activation function?

| Regression                        | Classification                  |                   |
|-----------------------------------|---------------------------------|-------------------|
| any nonlinear activation function |                                 |                   |
| Hidden layers                     | like ReLU, sigmoid, softmax etc | as left           |
| Output layer                      | typically identity              | typically softmax |

Nonlinear activation function in the hidden layers is fundamental for both regres-
•
sion and classification.

Softmax in the output layer is necessary for classification in order to transform any
•
logits a ∈ Rc into probability values.

Nonlinear transformation is not necessary in the output layer for regression because

•
we already have enough nonlinearities in the hidden layers.

## Universal Approximation Theorem

The universal approximation theorem states that a feedforward neural network with a linear output layer (φL(a) =
a) and at least one hidden *layer* with

- •
nonlinear activation function a can approximate any continuous (nonlinear) function y(x0) (on compact input sets) to arbitrary accuracy.

## Comments:

arbitrary accuracy: with an increasing number of hidden neurons.

•
valid for a wide range of nonlinear activation functions, but excluding polynomials.

•
minimum requirement for universal approximation: W2φ1(W1x0 b1) +
b2.

+
•
This existence theorem provides only the minimum requirements. It doesn't tell us much about the design and training of a NN. In practice, deep networks are better than shallow ones,

•
some activation functions are better than others.

•

## E4.3: Regression (1)

x2)

true function f0(x) =
sin(1
+
•
N(0, 0.12) for 0 N
100 noisy samples y(n) =
f0(x(n)) +
x(n) ≤
3
≤
•
=
network output f(x) as approximation for f0(x)

•
a) 1 hidden layer with 1 and 10 hidden neurons M0 M2 M1
=
=
=

![108_image_0.png](108_image_0.png) 

E4.3: Regression (2)

b) 1 hidden layer with M1 100 neurons

=

![109_image_0.png](109_image_0.png) 

$$4.5\quad\mathrm{Un}$$

4.5 Universal approximation 4-13

## E4.3: Regression (3)

c) 2 hidden layers with M0 1, M1 100, M2 5, M3 1
=
=
=
=

![110_image_0.png](110_image_0.png) 

# Case C: Laplace Distribution And L1-Loss

 
y = f(x; θ) + z, z = [z1, . . . ,zc]T . Case C: zi i.i.d. Laplace(0, b), i.e., yi i.i.d. Laplace(fi(x; θ), b) Y c Y c 1 2be−1b|yi−fi(x;θ)|, q(y|x; θ) = q(yi|xi; θ) = i=1 i=1 c 1 1 X l(x, y; θ) = ln(q(y|x; θ)) = |yi − fi(x; θ)| = bky − f(x; θ)k1, const + const + − b i=1 N N 1 1 X X L(θ) = l(x(n), y(n); θ) = ky(n) − f(x(n); θ)k1. N N n=1 n=1 This is the mean absolute error (MAE) loss or l1-loss.


 

 

 

 



+
Note:

The cost function L(θ) depends on the assumed distribution of y − f(x; θ).

•
Each distribution assumption implies a certain approximation of p(y|x) by q(y|x; θ).

•
In all cases, y − f(x; θ) is minimized in a certain sense.

•

## Output Layer, Loss And Cost Function

The loss l(x, y; θ) measures the quality of prediction for one pair (x, y).

•
1 PNn=1 The L(θ) =
l(x(n), y(n); θ) is the average loss over all N
traincost function
•
N
ing samples.1

| Regression                     | Classification   |                     |
|--------------------------------|------------------|---------------------|
| ♯Output neurons ML             | length of y      | number of classes c |
| Activation functionφL          | identity         | softmax             |
| l2-loss or MSE-loss            | categorical loss |                     |
| Lossl(x, y; θ)                 | ky − f(x; θ)k2   | −yT lnf(x; θ)       |
| weightedl2-loss                | focal loss       |                     |
| (y − f(x; θ))TC−1(y − f(x; θ)) | −yT [(1          | f(x; θ)]            |
| l1-loss or MAE-loss            | f(x; θ))γ ⊙ln    |                     |
| −                              |                  |                     |
| ky − f(x; θ)k1                 |                  |                     |

They are called distribution-based **loss** because they are derived from the distribution.

## The Probabilistic Way Toward The Cost Functions

 p(x, y): true but unknown data generating distribution, application specific
•
{x(n), y(n), 1 N}: training set, i.i.d. samples of p(x, y)

Dtrain
≤
≤
•
n
=
 p(y|x): true posterior describing the desired inference x →
•
y q(y|x; θ): a parametric model (DNN) to approximate p(y|x)
•
min. forward KL divergence DKL(p(x, y)||q(x, y; θ))
↓
ch. 3: p fixed


 
 # 5: $q(\nu|\chi;\ell)$


min. cross entropy H(p, q) =

−EX,Y∼p(x,y) ln(q(Y|X; θ))
↓
ch. 3: use empirical distribution ˆp(x, y)

min. cross entropy H( ˆp, q) =

$\square$         1. 
−EX,Y∼pˆ(x,y) ln(q(Y|X; θ))
↓
ignore constant term 1 PNn=1[−
min. cost function L(θ) =
ln(q(y(n)|x(n); θ))]

N
↓
ch. 5: q(y|x; θ) for regression and classification l2-loss or l1-loss or categorical loss

## Other Losses For Classification⋆

Categorial CE loss is the standard loss for classification, but not the only possibility.

a) l2-loss ky − f(x; θ)k2 for classification?

Theoretically possible, but no probabilistic interpretation
–
fi(x)| (outliers) penalized due to (·)2 large deviations |yi −
–
slow convergence during training
–
b) 0/1-loss for classification?

The 0/**1-loss** is defined as


 
( 0 for y = estimated class label f(x) 1 for y , estimated class label f(x)
$$l(\underline{{{x}}},\underline{{{y}}})=\left\{\begin{array}{l l}{{0}}&{{\mathrm{for~}y=\mathrm{ess}}}\\ {{1}}&{{\mathrm{for~}y\neq\mathrm{ess}}}\end{array}\right.$$
 
 
+
L(θ) is the error rate of the classifier, see DPR. This is a better, more objective performance metric for classification, but is not differentiable for training.

Then the cost function

# Semantic Image Segmentation

Assign each input pixel xhw to one of c object classes.

2D color input image of size H
W, i.e.

×
•
∈ {0, . . . , 255}H×W×3.

3D input tensor X
[xhw]hw
=
[yhw]hw ∈ {0, 1}H×W×c, one-hot coding class label tensor Y
•
=
∈ {e1, . . . , ec} for pixel (h, w)
 yhw ˆ
f(X; θ) =
Output tensor Y
- 
=
c-class softmax output ˆyhw


 

 

 

$$\begin{array}{l}{{0)=[\hat{\mathcal{V}}_{h w}]_{h w}\in\mathbb{R}^{H\times W\times c},}}\\ {{\quad\in\mathbb{R}^{c}\mathrm{~for~pixel~}(h,w)}}\end{array}$$

 
 



+
How to define the loss l(X, Y; θ) for image segmentation? 

ˆ 
ˆ 
) ∈ {0, 1}H×W×c by replacing each ˆyhw Final segmentation result: Hard decision of Y 
with HD(Y
•
Rc by a one-hot vector HD(ˆyhw) ∈ {0, 1}c indicating the class with the max. probability.

∈

![115_image_0.png](115_image_0.png) 

## Iou Loss And Dice Loss (1)

Given two sets A and B, e.g. the sets of true and estimated object pixels in a binary segmentation, i.e. object vs. background. The Jaccard or IoU **loss** (intersection-overunion) is defined as

$$0\leq J={\frac{|A\cap B|}{|A\cup B|}}={\frac{|A\cap B|}{|A|+|B|-|A\cap B|}}\leq1.$$

denote the **intersection** and **union** of A and B, respectively. |A| ∈ N

is the **cardinality** of A, i.e. the number of elements in A.2 and A
B
A
B
∩
∪
The Dice **loss** has a slightly different definition

$$0\leq D={\frac{2|A\cap B|}{|A|+|B|}}\leq1.$$

They are two similar region overlap **metrics** related by J =

D
D. If A
B
Ø, J = D = 0. If A = B, J = D = 1.

≤
∩
= 
2−D

![116_image_0.png](116_image_0.png) 
In contrast to the categorical CE loss, there is no problem of imbalanced pixel classes.

J
and D
are independent of the region size of the background.

## Iou Loss And Dice Loss (2)

The previous IoU and Dice loss are defined for one object class only. In a multi-class segmentation, they can be calculated for each of the c classes. Let Ai and Bi denote the sets of true and estimated pixels for class 1
(e.g. human, vehicle, street, ...),

i ≤
≤
c respectively. Then



$$0\leq J_{i}=\frac{|A_{i}\cap B_{i}|}{|A_{i}\cup B_{i}|}\leq1$$

 

 


 

 
+
$${\mathrm{~is~the~IoU~loss~for~class~}}i{\mathrm{~and~}}$$
is the IoU loss for class $i$ and  $$J=\frac{1}{c}\sum_{i=1}^{c}J_{i}$$  is the mean IoU loss. The mean Dice loss is defined in a similar way.  

The above metrics are used for the evaluation of a semantic segmentation, i.e. comparison of the true and estimated class labels. They are not suitable for training because they count pixels and are not differentiable.

## Soft Iou And Dice Loss

[yhw]hw and Y

ˆ
∈ {e1, . . . , ec} is the one-hot coding and y ˆ
Let again Y
[ˆyhw]hw. yhw
= 
=
Rc is the c-class softmax output for the pixel (h, w). The soft IoU **loss** to be

∈

hw minimized in training for image segmentation is


PHh=1 PWw=1 [αi] ∈ Rc, yˆhw ⊙ α yhw = = PHh=1 PWw=1(yhw β yˆhw) = [βi] ∈ Rc, + = αi + ǫ Ji = ǫ, βi − αi + 1 Pci=1 l(X, Y; θ) = 1 Ji − c

 
 


 

 


is the elementwise multiplication. αi and βi represent arithmetic calculations of |Ai ∩
⊙
Bi| and |Bi| for class i, respectively. Ji is the soft IoU loss for class i where |Ai| +
0 is a suitable number to avoid 0/0 if αi =
0. l(X, Y; θ) is the soft IoU
βi =
>
ǫ loss differentiable w.r.t. θ. The soft Dice **loss** is defined in a similar way. For 3D 
segmentation, α and are calculated by three-dimensional sums over all pixels.

β Soft IoU/Dice loss for training; IoU/Dice loss for evaluation.

## Combination Of Losses

Different losses have different impacts to the training. Very often, they are combined to achieve a compromise.

Let LCE(θ) and LIoU(θ) be the cost functions based on the CE and IoU loss for semantic segmentation, respectively. One can minimize L(θ) =
LCE(θ) +
λLIoU(θ).

The parameter λ 0 controls the balance between LCE(θ) and LIoU(θ) and has to be

>
chosen suitably. Instead of the CE loss, also the focal loss can be used.

The final grade of a student thesis at ISS is a weighted average of three subgrades for the work, thesis, and presentation.

•
Work-life balance is also a kind of compromise.

•

# Signal Chain In Dnn

$$\bullet\operatorname*{min}_{\underline{{\theta}}}\;L(\underline{{\theta}})=\frac{1}{N}\sum_{n=1}^{N}l(\underline{{x}}(n),\underline{{y}}(n);\underline{{\theta}})$$


 

 

+
$$\mathrm{NN~output}\;\underline{{{x}}}_{L}=f(\underline{{{x}}};\underline{{{0}}})$$
$$\bullet\;l(\underline{{{x}}},\underline{{{y}}};\underline{{{\theta}}})\;\mathrm{depen}$$

l(x, y; θ) depends on the DNN output xL
f(x; θ) with x = x0.

•
contains all parameters of Wl =
[wl,i j] and bl = [bl,i], 1 ≤ l ≤ L.

θ
•
L(θ) is a cascade of functions of the elements of θ.

$$\mathrm{\Large~Layer}\;L\colon\frac{\partial L(\theta)}{\partial w_{L,i j}}\colon$$

$$\underline{{{x}}}_{0}\rightarrow\ldots\to\underline{{{x}}}_{L-2}\stackrel{\mathrm{W}_{L-1}}{\underset{\underline{{{b}}}_{L-1}}{\longrightarrow}}\underline{{{a}}}_{L-1}\stackrel{\mathrm{W}_{L}}{\underset{\phi_{L-1}}{\longrightarrow}}\underline{{{x}}}_{L-1}\stackrel{\mathrm{W}_{L}}{\underset{\underline{{{b}}}_{L}}{\longrightarrow}}\underline{{{a}}}_{L}\stackrel{\mathrm{X}}{\underset{\phi_{L}}{\longrightarrow}}\underline{{{x}}}_{L}\to\mathcal{L}(\underline{{{\theta}}})$$

$$\mathrm{\Large~Layer}\,L-1\colon\frac{\partial L(\theta)}{\partial w_{L-1,i j}}\colon$$

Forward pass vs. backward pass in each update step Forward **pass** to calculate xL
from x0:

![121_image_0.png](121_image_0.png)

$$\operatorname{tors}\underline{{{\partial}}}_{l}^{T}:=\mathbf{J}_{L}(\underline{{{a}}}_{l})=\frac{\partial L(\underline{{{\theta}}})}{\partial\underline{{{a}}}_{l}}$$

 
 
$$\longrightarrow L(\underline{{{\theta}}})$$

Backward pass or **backpropagation** of so-called error **vectors** δTl := JL(al) = ∂L(θ) ∂al ∈
R1×Ml:

![121_image_1.png](121_image_1.png)

$$-\,\underline{{{\delta}}}_{L}$$
For $l=L-1,\ldots,1$  $$\delta_{l}^{T}=\delta_{l+1}^{T}\cdot\mathsf{J}_{\underline{a}_{l+1}}(\underline{x}_{l})\mathsf{J}_{\underline{x}_{l}}(\underline{a}_{l})=\overline{\phantom{\underline{a}_{l}}}\cdot\left[\right]$$ $$\frac{\partial L(\theta)}{\partial w_{l,ij}}=\delta_{l}^{T}\cdot\mathsf{J}_{\underline{a}_{l}}(w_{l,ij})=\overline{\phantom{\underline{a}_{l}}}\cdot\left[\right]$$ $$\frac{\partial L(\theta)}{\partial b_{l,i}}=\delta_{l}^{T}\cdot\mathsf{J}_{\underline{a}_{l}}(b_{l,i})=\overline{\phantom{\underline{a}_{l}}}\cdot\left[\right]$$

# Jacobi Vectors/Matrices During Backpropagation (1)

a) Loss: xL cost function L(θ) → "  # ∈ R1×ML, ∂L(θ) ∂L ∂L JL(xL) = = · · ·   ∂xL,1  ∂xL,ML ∂xL N 1 X L(θ) = l(x(n), y(n); θ), N n=1 X N ∂l(x(n), y(n); θ) ∂L(θ) 1 = ∂xL,i , ∂xL,i N n=1  ky − xLk2  for l2-loss l(x, y; θ) = −yT ln(xL) for categorical loss ,  −2(yi − xL,i) for  ∂l(x, y; θ) l2-loss = yi xL,i for categorical loss . ∂xL,i −


 

 

 


 

 
# Jacobi Vectors/Matrices During Backpropagation (2)

$${\bf b})\;\mathrm{Activation~function}\colon\underline{{{a}}}_{l}\to\underline{{{x}}}_{l}=\phi_{l}(\underline{{{a}}}_{l}),\;1\leq l\leq L$$

 


$$\mathbf{J}_{\underline{{{x}}}_{l}}(\underline{{{a}}}_{l})={\frac{\partial x_{l}}{\partial\underline{{{a}}}_{l}}}=\left[\mathbf{\theta}\right]$$

∂xl,1 · · ·  ∂al,1 ... ... ... ∂xl,Ml  ∂al,1
$$\left.\begin{array}{c}{{\partial x_{l,1}}}\\ {{\overline{{{\partial a_{l,M_{l}}}}}}}\\ {{\vdots}}\\ {{\partial x_{l,M_{l}}}}\\ {{\overline{{{\partial a_{l,M_{l}}}}}}}\end{array}\right|\in\mathbb{R}^{M_{l}\times M_{l}}.$$

According to the derivative calculation of activation functions in ch. 4, Hidden layer l, ReLU
φl(a) =
φ(a) =

$$(0,a)\;\mathrm{with}\;\phi^{\prime}(a)$$

max(0, a) with φ′(a) =
u(a):

•

$\mathbf{J}_{\underline{x}_{l}}(\underline{a}_{l})=\text{diag}(\phi^{\prime}(a_{l,1}),\ldots,\phi^{\prime}(a_{l,M_{l}}))=\text{diag}(u(\underline{a}_{l}))$.  
$\bullet$ Hidden layer $l$, sigmoid $\phi_{l}(a)=\sigma(a)$ with $\sigma^{\prime}(a)=\sigma(a)(1-\sigma(a))$:
$$\mathbf{J}_{\underline{{{x}}}}(\underline{{{a}}}_{l})=\mathrm{diag}(\sigma^{\prime}(a_{l,1}),\ldots,\sigma^{\prime}(a_{l,M_{l}}))=\mathrm{diag}(\underline{{{x}}}_{l}-\underline{{{x}}}_{l}\odot\underline{{{x}}}_{l}).$$
Output layer L, identity activation function with xL = aL:
•

$$\mathbf{J}_{\underline{{{x}}}_{L}}(\underline{{{a}}}_{L})=\mathbf{I}.$$

5.2 Training and validation 5-14

# Jacobi Vectors/Matrices During Backpropagation (3)

$$\bullet\mathrm{{Output~layer}}\,L,\,\mathrm{softmax}\,\phi_{L}(\underline{{{a}}}_{L})=\mathrm{softmax}(\underline{{{a}}}_{L})\mathrm{:}$$

 

 


 
$${\bf J}_{\underline{{{x}}}_{L}}(\underline{{{a}}}_{L})=\mathrm{diag}(\underline{{{x}}}_{L})-\underline{{{x}}}_{L}\underline{{{x}}}_{L}^{T}.$$
$${\mathrm{c)}}\ {\mathrm{Layer~transition}}\ \underline{{{x}}}_{l-1}\to\underline{{{a}}}_{l}={\mathrm{W}}_{l}\underline{{{x}}}_{l-1}+\underline{{{b}}}_{l},\ 1\leq l\leq L\colon$$
$$\mathbf{J}_{{\underline{{{a}}}}_{l}}({\underline{{{x}}}}_{l-1})={\frac{\partial a_{l}}{\partial{\underline{{{x}}}}_{l-1}}}=\mathbf{W}_{l}\in\mathbb{R}^{M_{l}\times M_{l-1}}.$$

d) **Weight $w_{l,ij}\rightarrow\underline{a}_{l}=\mathbf{W}_{l}\underline{x}_{l-1}+\underline{b}_{l},\ \ 1\leq l\leq L$:**  $$\mathbf{J}_{\underline{a}_{l}}(w_{l,ij})=\frac{\partial\underline{a}_{l}}{\partial w_{l,ij}}=[0,\ldots,0,\underbrace{x_{l-1,j}}_{i-th\ \text{position}},0,\ldots,0]^{T}=x_{l-1,l}\underline{u}_{l}\in\mathbb{R}^{M_{l}\times1}.$$

  **c) Bias $b_{l,i}\to\underline{a}_{l}=\mathbf{W}_{l}\underline{x}_{l-1}+\underline{b}_{l},\ 1\leq l\leq L$:**  $$\mathbf{J}_{\underline{a}_{l}}(b_{l,i})=\frac{\partial\underline{a}_{l}}{\partial b_{l,i}}=[0,\ldots,0,\underbrace{1}_{i-th\ \mathrm{position}},0,\ldots,0]^{T}=\underline{u}_{l}\in\mathbb{R}^{M\times1}.$$

# Jacobi Vectors/Matrices During Backpropagation (4)

$$=\delta_{l+1}^{T}\cdot{\bf J}_{\underline{{{a}}}_{l+1}}(\underline{{{x}}}_{l}){\bf J}_{\underline{{{x}}}_{l}}(\underline{{{a}}}_{l})=\delta_{l+1}^{T}\cdot{\bf W}_{l+1}{\bf J}_{\underline{{{x}}}_{l}}(\underline{{{a}}}_{l}),\quad\epsilon$$
δTl = δTl+1 · Jal+1(xl)Jxl(al) = δTl+1 · Wl+1Jxl(al), l = L − 1, . . . , 1

 
 

![125_image_0.png](125_image_0.png)

$$l=L-1,\ldots,1$$

All quantities of all layers of both forward and backward pass must be stored in the GPU memory for fast training Need large enough GPU memory!

→

# Four Components For Machine Learning

| optimizer        |       |               |
|------------------|-------|---------------|
| dataset          | model | cost function |
| machine learning |       |               |

| Description                                 | Deep learning             | Other possibilities   |
|---------------------------------------------|---------------------------|-----------------------|
| Dataset                                     | for training and test     |                       |
| A                                           | parametric                | model                 |
| q(y|x; θ) as an approxi                                             | DNN with the parame                           |                       |
| Model                                       | SVM, GMM                  |                       |
| mation for the posterior                    | ter vector θ              |                       |
| p(y|x) L(θ) measures the devi                                             |                           |                       |
| Cost                                        | mostlyl2, categorical or  |                       |
| ation of the model out                                             | . . .                     |                       |
| function                                    | soft IoU loss             |                       |
| put from ground truthoptimization algorithm | gradient descent or vari- |                       |
| Optimizer                                   | Newton, . . .             |                       |
|                                             |                           |                       |
| to minimize L(θ)                            | ants                      |                       |

## Epoch And Minibatch

For a convergence, one needs to pass the training set Dtrain multiple times to the DNN.

One epoch denotes one pass of Dtrain through the DNN. It consists of N/B minibatches of size B and corresponds to N/B update **steps** of θ.3

![127_image_0.png](127_image_0.png) 
0, 1, . . . , NB − 1 is the minibatch index.

•
mod(t, NB) =
For each epoch, t′ =
•
For each minibatch t′, t′B
(t′ +
1)B
is the sample index for x(n), y(n).

1
≤
≤
+
n

## E5.1: Mnistnet1 - Baseline Network (1)

on MNIST. It serves as a baseline network for further studies.

The first neural network in this course for digit recognition

## Dataset: Mnist

•
gray images for 10 digits 0, 1, . . . , 9
•
60,000 training images and 10,000 test images of size 28x28

## Network: Mnistnet1

1D dense neural network
•
vectorize each 28x28 gray image into a column vector x0(n) ∈ R784
•
input layer with M0 784 neurons

•
=
1 hidden layer with M1 128 neurons and ReLU activation function
•
=
10 neurons in the output layer with softmax activation function M2
•
=
Np = M1(M0 + 1) + M2(M1 + 1) = 101, 770 parameters
•
101, 632 multiplications N×
M1M0 M2M1
+
•
=
=
5.3 Implementation in Python 5-19 E5.1: MNISTnet1 - Baseline network (2)

Import TensorFlow and **packages**

import tensorflow as tf import tensorflow.keras as keras

## Load Dataset And Scale Input

(x_train, y_train), (x_test, y_test)
= keras.datasets.mnist.load_data()
x_train 
= x_train / 255.0 x_test 
= x_test / 255.0 E5.1: MNISTnet1 - Baseline network (3)

Define the network

$$\begin{array}{l l l}{{\underline{{{x}}}_{0}\;=\;\underline{{{x}}}}}\\ {{\underline{{{a}}}_{l}\;=\;\mathbf{W}_{l}\underline{{{x}}}_{l-1}+\underline{{{b}}}_{l},}}&{{l=1,2}}\\ {{\underline{{{x}}}_{1}\;=\;\mathrm{ReLU}(\underline{{{a}}}_{1})}}\\ {{\underline{{{x}}}_{2}\;=\;\mathrm{softmax}(\underline{{{a}}}_{2})}}\end{array}$$


 
 
$$\begin{array}{l}{{\mathrm{ut\_shape=(28,~28))}\,,}}\\ {{\mathrm{activation='relu')}\,,}}\\ {{\mathrm{citation='softmax')]})}}\end{array}$$

model 
= keras.Sequential([
keras.layers.Flatten(input_shape=(28, 28)), keras.layers.Dense(128, activation='relu'),
keras.layers.Dense(10, activation='softmax')])
R28×28 is reshaped to a In the so-called flatten layer (see ch. 7), an input image X
∈
vector x = vec(X) ∈ R784.

## E5.1: Mnistnet1 - Baseline Network (4)

Define the **loss**, the categorical cross entropy, and choose the **optimizer**, the stochastic gradient descent (SGD) with the learning rate lr.

$$\mathrm{\boldmath~\hbar~}\mathrm{\boldmath~size}=32\;.\quad\mathrm{\boldmath~\epsilon~}$$
 
$$\underline{{{\theta}}}^{+1}=\underline{{{\theta}}}^{\prime}-\frac{\gamma^{\prime}}{B}\sum_{n=t^{\prime}B+1}^{(t^{\prime}+1)B}\underline{{{\nabla}}}\;l(\underline{{{x}}}(n),\underline{{{y}}}(n);\underline{{{\theta}}}))\Big|_{\underline{{{\theta}}}=\underline{{{\theta}}}^{\prime}}\,,\qquad t^{\prime}=\mathrm{mod}(t,N/B)$$


$$\mathrm{{\bf~ers\,.\,SGD}\,(\mathrm{\bf~l}r=0\,.\,1)}$$

= keras.optimizers.SGD(lr=0.1)
model.compile(optimizer=opt, opt loss='sparse_categorical_crossentropy', metrics=['accuracy'])
The sparse categorical CE is just an efficient implementation of the categorical CE.

Training and **test** using a minibatch size B = 32 and 30 epochs history 
= model.fit(x_train, y_train, batch_size=32, epochs=30, validation_data=(x_test, y_test))
All training and test results (cost function, accuracy) are contained in the object history. Alle optimized model parameters θ are contained in the object model.

## E5.1: Mnistnet1 - Baseline Network (5) Results

technical metric: training/test cost function L(θ) after each epoch
•
objective metric: training/test error rate (1−
accuracy) after each epoch
•
•
almost zero training error rate

•
but large gap between training and test error rate →
overfitting

![132_image_0.png](132_image_0.png)

## E5.1: Mnistnet1 - Baseline Network (6)

R128×784 after training Visualization of W1
∈
take the first 100 rows of W1: weight vectors of the first 100 hidden neurons

•
normalize each row vector of length 784 by its l2-norm and reshape it to a 28x28 weight matrix
•
arrange 100 weight matrices in a 10x10 grid, resulting in a 280x280 image

- •
the learned weight vectors seem to match to some digit-like patterns

![133_image_0.png](133_image_0.png)

# Challenges In Optimization

Training a NN, in particular a DNN, is a challenging task and suffers from a number of optimization difficulties. Some of these difficulties are common to all optimization 
(OPT) problems, while some others are special to the training of a DNN.

| Difficulty                 | OPT      | DNN                          | Solutions      |
|----------------------------|----------|------------------------------|----------------|
| D1) stochastic gradient    | ×        | larger minibatch size B, 5.5 |                |
| ×                          |          |                              |                |
| D2) ill conditioning       | 5.5, 5.7 |                              |                |
| ×                          | ×        |                              |                |
| D3) saddle point / plateau | ×        | ×                            | noisy gradient |
| D4) sensitive to step size | ×        | ×                            | 5.6, 5.7       |
| D5) local minimum          | ×        | ×                            | 5.8, 5.9       |
| D6) vanishing gradient     | ×        | 5.9                          |                |

# Basics Of Optimization

f(x) : x = [xi] ∈ RN →
Given a differentiable function f ∈
R.

gradient **vector** ∇f(x) and Hessian **matrix** H(x):
•

$${\underline{{\nabla}}}f(x)={\left[\begin{array}{l}{{\frac{\partial f}{\partial x_{1}}}}\\ {\vdots}\\ {{\frac{\partial f}{\partial x_{N}}}}\end{array}\right]}\in\mathbb{R}^{N},\qquad\mathbf{H}(x)={\left[\begin{array}{l l l}{{\frac{\partial^{2}f}{\partial x_{1}^{2}}}}&{\cdots}&{{\frac{\partial^{2}f}{\partial x_{1}\partial x_{N}}}}\\ {\vdots}&{\ddots}&{\vdots}\\ {{\frac{\partial^{2}f}{\partial x_{N}\partial x_{1}}}}&{\cdots}&{{\frac{\partial^{2}f}{\partial x_{N}^{2}}}}\end{array}\right]}\in\mathbb{R}^{N\times N}$$

x∗ is a local minimum of f(x) if there is no smaller value than f(x∗) in a local
•
neighborhood of x∗, i.e. f(x) ≥
f(x∗) ∀kx − x∗k ≤ δ.

 f(x) if there is no smaller value than f(x∗) in RN, i.e.

f(x) ≥
x∗ is a global minimum of
•
f(x∗) ∀x ∈ RN.

x∗ is a local maximum of f(x) if f(x) ≤
f(x∗) ∀kx − x∗k ≤ δ.

•
x∗ is a stationary **point** of f(x) if ∇f(x∗) = 0.

•
x∗ is a saddle **point** of f(x) if it is a stationary point, but neither a local minimum
•
nor a local maximum.

# 1D Visualization

![136_image_0.png](136_image_0.png)

![136_image_1.png](136_image_1.png)

+
desired global minimum

```
                                                  
1 local minima
                                  
2 local maxima stationary points
                                                                                                      
3 saddle points
                                  
4

```

plateau, region of constant f(x)

# Conditions On A Local Minimum

Necessary condition for x∗ being a local minimum:
•
x∗ is a stationary point, i.e. ∇f(x∗) =
0 If f(x) is convex: The above condition is also sufficient for x∗ being a global mini-
•
mum.

 f(x) is non-convex: The above condition is only necessary. The Hessian matrix H(x∗) is required to distinguish between different types of stationary point:

If
•
H(x∗) non-negative definite: x∗ local minimum
⋄
H(x∗) non-positive definite: x∗ local maximum
⋄
H(x∗) indefinite: x∗ saddle point

⋄
Even if x∗ is a local minimum, there is no guarantee that it is a global minimum.

•
See course AM for more details.

The cost function L(θ) of a DNN with nonlinear activation functions is non-convex.

# Gradient Descent Minimization

Task: minx f(x)
Gradient descent minimization

![138_image_1.png](138_image_1.png)

Properties:

It is a local search.

•
−∇f(x) points to the direction of steepest descent at the current position x. This is,

•
however, often not the direction pointing to the local minimum.

−∇f(x) orthogonal to the tangent of the contour line.

•
H−1(x) and hence much simpler than No need to calculate Hessian matrix H(x) and
•
second-order methods like Newton method.

It is like hiking downhill in fog without GPS and map.

•

![138_image_0.png](138_image_0.png)

# D1) Stochastic Gradient

The gradient vector of a batch cost function for a NN

$$\underline{{{\nabla}}}L(\underline{{{\theta}}})=\frac{1}{N}\sum_{n=1}^{N}\underline{{{\nabla}}}l(\underline{{{x}}}(n),\underline{{{y}}}(n);\underline{{{\theta}}})$$

is calculated over the complete set of N

# $\alpha$ samples. If $N$ is large, $\nabla I(\theta)$ is a. 
training samples. If N
is large, ∇L(θ) is almost deterministic.

When using minibatches, the gradient vector of the minibatch cost function

$$\frac{\nabla L(t;\underline{\varrho})}{B}=\frac{1}{B}\sum_{n=t}^{(t^{\prime}+1)B}\frac{\nabla l(\underline{x}(n),\underline{y}(n);\underline{\varrho})}{B},\qquad t^{\prime}=\mathrm{mod}(t,N/B)$$  is averaged over a smaller number of $B\ll$

N
samples. It is a stochastic gradient with a noisy update direction. This leads to a worse convergence per update step.

![139_image_0.png](139_image_0.png)

## Choice Of The Minibatch Size B

It is a nontrivial choice depending on many factors: size of the dataset, DNN model, availability of GPU, amount of GPU memory, desired convergence rate, etc. It is often a trade-off between accuracy and speed.

| Advantages of a large B            | Advantages of a small B                |                                     |
|------------------------------------|----------------------------------------|-------------------------------------|
| math.                              | •less noisy gradient vectors           | - more minibatches and update steps |
| better convergence per update step | faster convergence per epoch           |                                     |
| -                                  | -                                      |                                     |
| GPU                                | more parallel processing per minibatch | minibatch fits into GPU RAM even    |
| -                                  | -                                      |                                     |
| reduced GPU time per epoch         | for large DNN and data tensors         |                                     |
| -                                  |                                        |                                     |

60, 000 training images on a GPU,

B

| B   | ⌈N/B⌉   | GPU time [ms] per minibatch   | GPU time [s] per epoch   |
|-----|---------|-------------------------------|--------------------------|
| 4   | 15,000  | 2                             | 30                       |
| 32  | 1,875   | 2                             | 3.75                     |
| 128 | 469     | 2                             | 0.94                     |

# D2) Ill Conditioning

well conditioned
−∇L(t; θ) points to the local minimum
(circular) contour lines ill **conditioned** (narrow) contour lines equal curvatures strongly different curvatures

−∇L(t; θ) mostly points to wrong directions fast convergence slow convergence (oscillation)

![141_image_0.png](141_image_0.png)

# D4) Sensitive To The Choice Of The Step Size

too small step size γt too large step size γt slow convergence oscillation and no convergence

![142_image_0.png](142_image_0.png)

![142_image_1.png](142_image_1.png)

The optimum step size γt depends on the cost function L(t; θ) and the current position
•
θt and is unknown in advance.

# D5) Local Minimum

![143_image_0.png](143_image_0.png)

The gradient descent method often converges to a local minimum. There is no guarantee to find the global minimum. But a local minimum is not always bad.

L(t; θ) comparable to that of the global minimum: Acceptable

•
Local minimum with
•
L(t; θ) much larger than that of the global minimum: Bad

•
Local minimum with You do not know whether it is a good or bad local minimum.

# Dnns Make D1-D5 Even More Serious

The difficulties D1–D5 are not new. They occur in all gradient descent minimizations.

In deep learning, however, they are even more serious because a DNN
contains a very large number of parameters in θ,

•
consists of a deep cascade of nonlinear functions,

•
is often over-designed (more parameters than necessary) in order to guarantee a

•
satisfied performance for a given task.

As a result, the cost function L(θ) of a DNN has a huge number of local minima and saddle points,

•
plateaus (some parameters have a very small influence on L(θ)),

•
many equivalent minima due to non-unique solutions (e.g. weight symmetry).

•
This makes the gradient descent training of a DNN much more difficult.

# Recursive Filter For Smoothing

The following recursive filter of first order with the coefficient 0 1
< β 
<

 

 

 
$$y(n)=\beta y(n-1)+x(n)=x(n)+\beta x(n-1)+\beta^{2}x(n-2)+\beta^{3}x(n-3)+\ldots$$
. . .

![145_image_1.png](145_image_1.png)


 


+
performs an exponentially weighted **average** of the input signal x(n) and returns a smoothed output y(n). The larger β, the stronger the smoothing effect.4

![145_image_0.png](145_image_0.png) 

![145_image_2.png](145_image_2.png)

# Stochastic Gradient Descent Vs. Momentum

![146_image_0.png](146_image_0.png)

![146_image_1.png](146_image_1.png)

| SGD                                        | SGD with momentum                            |
|--------------------------------------------|----------------------------------------------|
| strong oscillation in undesired directions | reduced oscillation                          |
| slow convergence in desired direction      | accelerated convergence in desired direction |

Momentum performs a recursive smoothing of the stochastic gradients.

accumulate the gradient component along the desired (horizontal) direction
•
reduce the oscillation along the undesired (vertical) direction
•
This accelerates the convergence.

# Momentum Vs. Nesterov Momentum

![147_image_1.png](147_image_1.png) 

![147_image_0.png](147_image_0.png) 

Nesterov momentum is like the momentum method except that the gradient vector

∇L(t; θ) is not calculated at the current position θ = θt, but rather at the "lookahead" position β∆θt−1.

θt +
θ
=

# Static Schedules

Static **schedules**: {γ0, γ1, . . .} fixed and independent of θt

$$\mathrm{\mathrm{\large~lules}}\colon\{\gamma^{0},\gamma^{1},\ldots$$

![148_image_0.png](148_image_0.png)

## Adaptive Schedules And Adam Optimizer

Popular adaptive schedules are **RMSprop** (root mean square propagation), Adam 
(adaptive moment estimation) and AdaGrad
(adaptive gradient algorithm). Below the Adam optimizer5 for updating the element θi of θ at iteration t is shown:


θ=θt , ∂L(t; θ)  gti =  ∂θi  mti = β1mt−1 i + (1 − β1)gti, vti = β2vt−1 i + (1 − β2)(gti)2, t i = mti/(1 − β1), ˆ m ˆt i = vti/(1 − β2), v t i ˆ m θt+1 i = θti − γ  qvˆti + ǫ

 



+
It is a refined version of the momentum method. gti is the stochastic gradient. mti and vti denote the exponentially weighted 1. and 2. order moment of gti. Often β1 = 0.9 and β2 0 is a small number to avoid division-by-zero. γ is a fixed learning rate.

0.99. ǫ 
>
=

## Covariate Shift

Different channels of the input of a DNN may have different means and variances, see a) below. Moreover, the distribution of the input samples may change from time to time, e.g. from training set to test set, see b). This phenomenon is called covariate shift in deep learning.

## E5.4: Covariate Shift

a) Heterogenous input channels acoustic measurement of a microphone force measurement of a force sensor x3, x4, x5 3D components of an acceleration sensor x6, x7, x8 3D components of a gyroscope sensor (angular velocity)
Different sensors may have different offsets and gains. This may lead to ill conditioning (D2).

b) Cat recognition in images training set: black cats 
–
test set: white cats
–
x1 x2 5.7 Normalization 5-41

## E5.5: A Linear Neuron

Input x ∈ Rd
$\underline{x}\in\mathbb{R}^{d}$  $f(w)=\underline{w}^{T}\underline{x}\in\mathbb{R}$, i.e. no hidden layer. This is called Wiener filter in SASP.  
Output f(w) =
 
 
$$y\in\mathbb{R}$$


 

 
 



+
R, i.e. no hidden layers and a single linear neuron.

| Ground truth   |
|----------------|

 y ∈ R

| Cost function   |
|-----------------|

Training set x(n), y(n), 1 ≤ n ≤ N

$\underline{x}(n),y(n),\ 1\leq n\leq N$  $L(\underline{w})=\frac{1}{N}\sum\nolimits_{n=1}^{N}(y(n)-\underline{w}^{T}\underline{x}(n))^{2}=\ldots=\underline{w}^{T}\mathbf{R}\underline{w}-2\underline{c}^{T}\underline{w}+\sigma_{y}^{2},$  $\mathbf{R}=\frac{1}{N}\sum\nolimits_{n=1}^{N}\underline{x}(n)\underline{x}^{T}(n)$ correlation matrix of $\underline{x}(n)$  $\underline{c}=\frac{1}{N}\sum\nolimits_{n=1}^{N}\underline{x}(n)y(n)$ cross-correlation vector between $\underline{x}(n)$ and $y(n)$  $\sigma_{y}^{2}=\frac{1}{N}\sum\nolimits_{n=1}^{N}y^{2}(n)$ power of $y(n)$

The contour lines {w|L(w) =
const} are ellipses in this case. Their shape and orientation are determined by the Hessian matrix
∇T L
H
∇
2R.

=
=

## Batch Normalization During Training (1)

The al(n) at layer l for one minibatch 1 batch normalization
(BN) of the activation B
is defined as

≤
≤
n

$$\left[\underline{{{a}}}(n)\right]_{i}\leftarrow\gamma_{i,i}\frac{\left[\underline{{{a}}}(n)\right]_{i}-\mu_{i,i}}{\sqrt{\sigma_{l,i}^{2}+\epsilon}}+\beta_{l,i},\quad1\leq n\leq B,\ 1\leq i\leq M_{l}.$$

It consists of two steps for each element hal(n)ii of al(n):
a) A
zero-mean unit-variance normalization like input normalization.

$$\mu_{l,i}=\frac{1}{B}\sum_{n=1}^{B}\left[\underline{{{a}}}_{l}(n)\right]_{i}\quad\mathrm{and}\quad\sigma_{l,i}^{2}=\frac{1}{B-1}\sum_{n=1}^{B}\left(\left[\underline{{{a}}}_{l}(n)\right]_{i}-\mu_{l,i}\right)^{2}$$

are the sample mean and variance of hal(n)ii of this minibatch. ǫ is a small positive number (e.g. 10−5) to avoid division-by-zero. In contrast to the complete dataset, σ2l,i ≈ 0 may happen for a small minibatch.

b) Scale-and-offset γl,i βl,i with two *learnable* parameters γl,i and βl,i per neuron
+

## Batch Normalization During Training (2)

a) Why zero-mean unit-variance normalization? Decoupling of the layers. 

During training, the gradient in layer l depends on the surrounding layers. Since the weights of the surrounding layers have not been stabilized, the learning of

–
layer l is neither stable. All layers are coupled which makes the training difficult. 

The zero-mean unit-variance normalization fixes the dynamic range of one layer,

–
makes it less dependent of the other layers. This decouples the layers and makes the optimization landscape smoother and training easier.

b) Why the second step γl,i βl,i? Allows a flexible data dynamic range for each
+
neuron and does not reduce the expressiveness of the network. 

 γl = σl and βl = µl, batch normalization is effectless. Typically, γl , σl and βl , µl. 

$\phi$
$\mathbf{M}$
If
–


 


bl is redundant due to βl and can be omitted, i.e. al(n) = Wlxl−1(n). 

–
γl and βl are learned from training data like Wl. Each neuron adjusts its indi-
–
vidual optimum dynamic range of hal(n)ii.

# Batch Normalization During Inference

The sample mean µl,i and sample standard deviation (std) σl,i previously change from minibatch to minibatch. They are used for batch normalization of the corresponding minibatch during training.

Let µtl,i and σtl,i be the sample mean and std of minibatch t. During training, exponentially weighted averages of µtl,i and σtl,i are calculated like in the momentum method.

This leads to sample mean and std of the whole training set:

 

$$\begin{array}{l}{{\overline{{{\mu}}}_{l,i}^{t}\;=\;\beta\overline{{{\mu}}}_{l,i}^{t-1}+(1-\beta)\mu_{l,i}^{t},}}\\ {{\overline{{{\sigma}}}_{l,i}^{t}\;=\;\beta\overline{{{\sigma}}}_{l,i}^{t-1}+(1-\beta)\sigma_{l,i}^{t}.}}\end{array}$$
+
µ¯tl,i and ¯σtl,i are used for batch normalization during inference.

# Parameter Initialization

1) Zero **initialization**, e.g. θ0 =
0.

Immediately after the zero initialization,

 
$$\begin{array}{l}{{\underline{{{a}}}_{l}\;=\;{\bf W}_{l}\underline{{{x}}}_{l-1}+\underline{{{b}}}_{l}=\underline{{{0}}},}}\\ {{\underline{{{x}}}_{l}\;=\;\phi_{l}(\underline{{{a}}}_{l})=\phi_{l}(\underline{{{0}}})}}\end{array}$$


+
holds for all layers l. All neurons of a layer do the same calculation. The functionality of a layer is reduced to a single neuron. This is bad.

In general, a symmetry in Wl and bl will lead to a symmetry of al and xl and is not desired, because effectively the number of neurons is reduced. Symmetry-breaking is thus an important requirement for initialization.

In practice, all bias vectors bl are initialized with zero,
•
all weight matrices Wl are randomly initialized to break the symmetry.

•

## He Initialization

Consider layer l after random initialization. Due to zero initialization of bias bl = 0,

al = Wlxl−1 + bl = Wlxl−1. We denote the elements of al,Wl, xl−1 by al,i, wl,i j, xl−1, j.
$a_{l}=\mathbf{W}_{l}\mathbf{x}_{l-1}+b_{l}=\mathbf{W}_{l}\mathbf{x}_{l-1}$. We der Hence, $a_{l,i}=\sum_{j=1}^{M_{l}-1}\mathbf{W}_{l,i}\mathbf{x}_{l-1,j}$. Since

 

 

+
xl−1, j are assumed to be i.i.d. with zero mean and variance σ2x,l−1,
•
wl,i j are i.i.d. with zero mean and variance σ2w,l,
•
xl−1, j and wl,i j are independent,
•
we obtain

PMl−1 j=1 wl,i jxl−1, j = PMl−1 j=1 E(wl,i j)E(xl−1, j) = 0, E(al,i) = E E(a2l,i) = E PMl−1 j=1 wl,i jxl−1, j2 = E Pj Pk wl,i jwl,ik xl−1, jxl−1,k Var(al,i) = Pj Pk E(wl,i jwl,ik)E(xl−1, jxl−1,k) = PMl−1 j=1 σ2w,lσ2x,l−1 = Ml−1σ2w,lσ2x,l−1. He proposed = constant activation flow in the forward pass, i.e., constant Var(al,i) and
constant σ2x,l−1 σ2x for all neurons i and layers l. Thus it yields

=

$$\sigma_{w,l}\sim\frac{1}{\sqrt{M_{l-1}}}.$$  The number of input neurons $M_{l-1}$ is called fan-in in this context.  

## How To Prevent Vanishing And Exploding Gradient?

```
There are di
         
          fferent ways:
                     

```

Use a better activation function with non-vanishing derivative, e.g., ReLU instead
•
of sigmoid, leaky ReLU instead of ReLU
Use batch normalization to normalize the activations 
•
 gradient **clipping** to limit the value of gradients to prevent exploding gradien Use t

•
Use a better optimization algorithm like Adam which is less sensitive to the scale

•
of gradients Use an architecture with skip-connections (or **shortcuts**), e.g. **ResNet**
•
Fore easier hiking, you can buy good hiking shoes, but you can also make the mountain smoother.

It is not necessary to implement all above techniques. Experiment with combinations of them to see what works best for your problem and architecture.

## E5.6: Mnistnet1 - Advanced Optimizations (1)

We use the same baseline network, cost function and optimizer settings as in E5.1.

Below we vary each time one optimizer parameter and study its impact.

a) Step **size**

γ ∈ {
0.001, 0.01, 0.1, 1
}

![158_image_0.png](158_image_0.png) 

A too small step size leads to a slow convergence.

•
A too large step size leads to a non-convergence.

•

## E5.6: Mnistnet1 - Advanced Optimizations (2)

b) Minibatch **size**

B
}

∈ {
4, 32, 128

![159_image_0.png](159_image_0.png) 

A too large minibatch size reduces the number of update steps per epoch.

•

## E5.6: Mnistnet1 - Advanced Optimizations (3) Observations

A good choice of optimizer settings can accelerate the convergence of training.

•
It does not, however, help to prevent overfitting. In all experiments, the best test

•
error rate is roughly 2% while the corresponding training error rate is almost zero.

The problem of overfitting is not solved yet.

•

## Model Capacity, Underfitting And Overfitting

Goal of machine learning: Learn a **model** f(x; θ) with a low test (generalization)

error, i.e., the model performs well for new unseen data x.

Model **capacity**: Ability of the model to learn a mapping y from training data.

It is determined by the model function x
→
f(x; θ), i.e. the DNN architecture and the number of parameters. A simple function f means a low model capacity and a complex function f with a large number of parameters implies a high model capacity (a large solution space).

Underfitting: Large dataset and low model capacity, model fails to learn the mapping, high training error.

Overfitting: Small dataset and high model capacity, model tends to the memorize training data without learning the underlying mapping y, very low training error, x
→
but high test error, model performs poorly for new unseen data.

6.1 Model capacity and overfitting 6-2

## E6.1: Underfitting And Overfitting In Univariate Polynomial Regression

x(n)2 +
Training set: x(n), y(n) =
N(0, 1), 1 N
5
≤
≤
n
=
a) a too simple model: f(x) =
underfitting a1x + a0 →
P
d aixi, d b) a too complex model: f(x) =
4 overfitting
→
=
i=0 a2x2 +
c) a reasonable model: f(x) =
+

![162_image_0.png](162_image_0.png)

Overfitting is visible on large coefficients. They lead to an erratic curve.

# How To Find The Right Model?

For multivariate regression and classification, wT x linear neuron b: a (too) simpel model

+
•
shallow NN: a quite powerful model due to the universal approximation theorem
•
deep NN: even more powerful

•
can solve more challenging problems than shallow NN
+
tend more to overfitting
−
Challenge: Find the right model complex enough to solve a given problem
•
simple enough to avoid overfitting
•
But: No theory to predict the right model for a given problem!

Solution:

use a powerful enough model (large DNN)

•
and regularization
•

## Regularization

In machine learning, all techniques to prevent overfitting are known collectively as regularization.

A model with a higher capacity than necessary for a given task has a large solution space, i.e. many different solutions with a comparable training error. Some of the solutions are overfitted to the training data while others not, see E6.1. Regularization reduces the solution space and prefers these solutions with a good generalization.

There are different methods for regularization. Some of them are applicable to general optimization (OPT) problems, some others to ML, and some of them to DNN only.

| Ch.   | Methods             | Change on                             | OPT   | ML   | DNN   |
|-------|---------------------|---------------------------------------|-------|------|-------|
| 6.2   | weight norm penalty | cost function                         | ×     | ×    | ×     |
| 6.3   | early stopping      | optimizer                             | ×     | ×    |       |
| 6.4   | data augmentation   | dataset                               | ×     | ×    |       |
| 6.5   | ensemble learning   | dataset/model/cost function/optimizer | ×     | ×    | ×     |
| 6.6   | dropout             | model                                 | ×     |      |       |

# Weight Norm Penalty

Why does this work?

A model with large weights tends to overfitting.

•
Large weights mean large changes in output for small changes in input. Thus, a model with large weights can fit perfectly to training samples, but will behave erratically between the training samples, see E6.1.

Most practical problems have a smooth input-output relationship. Hence models

•
with small weights are better because they behave smoothly between training samples (i.e. on test samples), not just right at the training samples.

Careful choice of the regularization parameters λl:
A too small value will lead to no effective regularization.

•
A too large value will seriously change the solution of the original problem min L(θ).

•
see hyperparameter optimization in the last section of this chapter.

# Early Stopping

If a model has a larger capacity than necessary, it tends to overfitting. In this case, the training error decreases continuously with the number of epochs. The longer

•
the training, the smaller the training error.

the test error, however, decreases first and then increases due to overfitting.

•
Early **stopping**: Stops the training before the test error increases.

For this purpose, the learned model needs to be applied to the validation set after
•
each epoch in order to calculate the validation error. The validation set is a small subset of the training set, see last section of this chapter.

This is a simple regularization method without any changes on dataset, model, cost

•
function and without tuning of any hyperparameters.

## Data Augmentation

Data **augmentation**: Generate synthetic but realistic training samples to increase the diversity of the training samples, to cover a larger part of the data distribution, and to prevent overfitting.

This is easy for image classification by slightly modifying
•
the images without changing their class labels, e.g., 
translation/rotation/scaling/flip
–
add noise 
–
modify colors, textures etc. 

–
use image patches

–
However, depending on the application, not all modifications are allowed. e.g.,

horizontal flip in character recognition due to confusion between "b" and "d".

Data augmentation is more difficult for image regression due to continuous-valued
•
label, e.g. age estimation from photo Data augmentation is even more difficult for non-image data like speech, audio,

•
radar, medical data.

![167_image_0.png](167_image_0.png)

# Ensemble Learning

Ensemble **learning**: A model averaging method to reduce the test error by combining an ensemble of models:

train different independent models for the same task
•
combine these models to reduce the test error 
•
regression: average of the model outputs 
–
classification: voting of the model outputs, e.g. 3×
cat and 1×
dog
–
It is unlikely that all independent models will make the same errors on the test set.

The different models can be trained by using different subsets of the training set or

•
different model architectures or

•
different cost functions or

•
different optimizers or

•
combinations of them.

•
In measurement, collect multiple (noisy) samples and compute their average.

![169_image_0.png](169_image_0.png) 
Training For each minibatch, dropout1 randomly removes some neurons in layer l of a base network with a

•
probability dl, the dropout **rate**.

this leads to a randomly thinned subnetwork for solving the same task
•
the dropped weights do not contribute to the activation on the forward pass and are

•
not updated on the backward pass

## Inference

no dropout, use the base network
•
all outgoing weights of neurons in layer l are weighted by 1−dl to correct the large fan-in to the neurons of the next layer.

•

## Dropout (2)

Why does dropout work?

When we train a single network with a large capacity, it often happens that the

•
weights of some neurons are sensitive to the weights of other neurons. They are coadapted. By dropout, the weights become more independent to the other weights.

This makes the model more robust. If a hidden neuron has to work well in different combinations with other hidden neurons, it's more likely that this hidden neuron does something individually useful.

The final DNN for inference can be viewed as an average of a huge number of
•
thinned networks. Each thinned network gets poorly trained
(underfitting) due to a smaller number of neurons. It will never be used alone. But the ensemble averaging of these weak models results in a stronger model. This is the basic idea of ensemble learning.

Dropout is simple to implement and does not boost the computational complexity.

•
Yet it is effective to prevent overfitting and is widely used.

Different groups of students in an apartment don't like each other.

Dropout: Random move out of groups of students.

## E6.2: Overfitting And Regularization In Regression By A Dnn (1)

x2)

true function f0(x) =
sin(1
+
•
N(0, 0.22) for 0 noisy samples y(n) =
f0(x(n)) +
x(n) ≤
N
50 3
≤
•
=
2 hidden layers with 100, M2 50 M1
•
=
=
i.e., a large model capacity but a small training dataset →
overfitting
•
a) No regularization

![171_image_0.png](171_image_0.png) 

Network learns a too complicated function f(x; θ). It is overfitted to training data.

## E6.2: Overfitting And Regularization In Regression By A Dnn (2)

b) l2-regularization to both hidden layers with λ = 0.001

![172_image_0.png](172_image_0.png) 

## E6.2: Overfitting And Regularization In Regression By A Dnn (3)

c) l1-regularization to both hidden layers with λ = 0.001

![173_image_0.png](173_image_0.png) 

## E6.3: Mnistnet2 (1)

The second dense neural network for MNIST.

architecture 784x128x128x10, i.e. 2 hidden layers with each 128 neurons

•
Np = 118, 282 parameters and N× = 118, 016 multiplications
•
minibatch size B = 32, 30 epochs and γt = 0.1 as in MNISTnet1
•
l2-regularization with 0.0002 λ
•
=
a) no dropout

•
b) dropout rate 0.2 in both hidden layers keras.layers.Flatten(input_shape=(28, 28)), keras.layers.Dense(128, activation='relu',
kernel_regularizer=keras.regularizers.l2(0.0002)),
keras.layers.Dropout(0.2),

```
                         
keras.layers.Dense(128, activation='relu',
    kernel_regularizer=keras.regularizers.l2(0.0002)),

```

keras.layers.Dropout(0.2), 
keras.layers.Dense(10, activation='softmax')])

## E6.3: Mnistnet2 (2)

![175_image_0.png](175_image_0.png) 

dropout is very effective to reduce overfitting
•
test error rate 2%, not really better than MNISTnet1
•

## E6.3: Mnistnet2 (2) R128×784

![176_Image_1.Png](176_Image_1.Png) Visualization Of W1 ∈

As in E5.1, the weight vectors of the first 100 neurons in the first hidden layer are

•
normalized and reshaped to a 10x10 grid containing each a 28x28 weight matrix.

![176_image_0.png](176_image_0.png)

no dropout dropout

Each of the neurons seems to match to a digit-like input pattern.

•
W1 with dropout shows a more clear pattern than without dropout, i.e., dropout

•
forces each neuron to learn something more useful.

## Hyperparameters What Are They?

In contrast to the model parameters θ (weights and biases) to be learned from training data, **hyperparameters** η are configuration parameters of a machine learning model which are not adapted during training.

•
η is chosen before learning and remains fixed. It controls, together with θ, the behavior of the model f(x; θ, η). They also need an optimization.

•
Why are they not learned together with during training?

θ a) Hyperparameters are often discrete valued (e.g. number of layers/neurons, type of activation function). Gradient descent is not applicable to them.

b) The training cost function is a monotone function of some hyperparameters which control the model capacity and the number of epochs. An optimization of these hyperparameters would always maximize the model capacity (e.g. more layers, more neurons) resulting in overfitting. The training set alone is not suitable for hyperparameter optimization.

# Hyperparameters Of A Dnn

| Model type of neural network        | dense network/CNN/RNN/transformer/. . .   |         |
|-------------------------------------|-------------------------------------------|---------|
| number of layers                    | L                                         |         |
| number of neurons                   | Ml, 1                                     | ≤ l ≤ L |
| type of activation function         | φl( ), 1                                  | l ≤L    |
| ≤                                   |                                           |         |
| dropout rate                        | dl, 1                                     | ≤ l ≤ L |
| Cost functiontype of regularization | or l1                                     |         |
| regularization parameters           | λl, 1                                     | ≤ l ≤ L |
| Optimizer                           | l2                                        |         |
| minibatch size                      | B                                         |         |
| number of epochs                    | Nepoch                                    |         |
| learning schedule and step size     | γt                                        |         |
| momentum factor                     | β                                         |         |
| Nesterov momentum                   | yes/no                                    |         |
| initial value of θ                  | θ0                                        |         |

## Training Set, Validation Set And Test Set

Training set Dtrain: It is used for training the model, i.e., learning the model parameters θ
(weights and biases) for a fixed hyperparameter vector η.

Validation set Dval: It is never used in training. It is reserved for tuning the hyperparameters η.

Test set Dtest: It is never used in training and hyperparameter optimization. It is used to calculate the test error of the trained (θ) and tuned (η) model f(x; θ, η) to exam its generalization capability. The motivations of using the test set are to avoid overfitting of θ to the training set and

- •
to avoid overfitting of η to the validation set.

## E6.4: Early Stopping

Early stopping is a hyperparameter optimization to determine the optimum number of epochs. This is only possible by using a validation set because the training error often decreases continuously as the number of epochs increases.

## Hyperparameter Optimization Approaches

Grid search A blind search on a human-selected grid in the hyperparameter

•
space, e.g. M1
∈ {100, 150, 200}, φl ∈ {sigmoid, ReLU}, dl ∈ {10%, 30%, 50%}, . . .

+) simple, −) time-consuming, especially for many hyperparameters and a fine grid
−) suboptimal performance for a coarse grid Treat hyperparameter tuning as an optimization problem Bayesian optimization
•
compute a probabilistic model for the posterior of the cost function p(L|η) based
–
on Bayes' rule and Gaussian processes iterative refinement of the model after each sample of (η, L) 
–
posterior: which regions of η are uncertain and worth exploring 
–
choose next value of η based on exploration and exploitation

–
- target function, - - - predicted function, samples, confidence interval

•
https://github.com/fmfn/BayesianOptimization:

A Python package ready for use.

![180_image_0.png](180_image_0.png)

## Drawbacks Of Dense Networks

Huge number of parameters in θ. The number of parameters of a dense layer Wl ∈

RMl×Ml−1 increases quadratically with the number of neurons for Ml =
•
Ml−1. 

a tremendous computational and memory complexity and
–
a higher overfitting risk due to a large model capacity.

–
Not suitable to learn local patterns/features of input due to the full connectivity of

•
neurons. Dense layers are good for decision making (classification/regression), but not for hierarchical feature learning.

## E7.1: Huge Number Of Parameters Of A Dense Network

a) tiny MNISTnet1 in E5.1:

282 =
Np = 101.770 parameters b) ImageNet, reduced image size 224x224:

M0 784, M1 128, M2 10
→
=
=
=
2242 =
the first layer has M1(M0 + 1) = 51.381.248 parameters!

M0 50176, M1 1024
→
=
=
c) Full HD image 1920x1080: M0 = 1920 · 1080 = 2.073.600, . . .

## 1D Convolution

see last section of ch. 3 "Digital signal processing"

$$\frac{y(n)}{n}$$

![182_image_2.png](182_image_2.png)

 
 
between $\chi(n)$ and $\hbar(n)$


$$\begin{array}{c c c}{{x(n)}}&{{}}&{{}}&{{x(n)}}\\ {{}}&{{}}&{{}}&{{}}\\ {{\prod_{i=1}^{n}x(n)^{i+1}(n)^{i+1}}}&{{}}&{{}}&{{}}\end{array}$$

![182_image_0.png](182_image_0.png)

![182_image_1.png](182_image_1.png)

The output y(n) is the **convolution** of the input x(n) with the **kernel** h(n)
•

$$y(n)=\sum_{i}h(i)x(n-i).$$

A similar operation is the **correlation** between x(n) and h(n)
•

$$\sum_{i}h(i)x(n+i)^{k=-i}\sum_{k}h(-k)x(n-k).$$

Clearly, convolution is equivalent to correlation of x(n) with h(−n).

Such a filter is **time-invariant**, namely a shifted input signal results in a shifted output signal: x(n − n0) →
•
y(n n0).

−

# 2D Convolution

![183_image_0.png](183_image_0.png)


 

 

+
m1, n2 − m2) is now called **shift-invariant**.

![183_image_1.png](183_image_1.png)

Using different kernels, different features can be extracted from the image.

The property x(n1 m1, n2 − m2) →
y(n1
−
−
The same idea is used in CNN except for the kernels are learned from data instead of human-designed automated feature learning
•
→
the output of each convolution passes a nonlinear activation function
•

# 2D Convolutional Layer L

The input Xl−1 contains Cl−1 channels (2D feature maps) of size Hl−1 × Wl−1.

•
The layer contains Cl 3D filters to calculate Cl output channels of size Hl × Wl.

•
Each 3D filter contains Cl−1 2D kernels of size Kl × Kl, each performing a **sliding**
window
•
convolution/correlation and then combined over all Cl−1 input channels.

![184_image_0.png](184_image_0.png) 

# 1D Convolutional Layer

RHl−1×Cl−1:
 Cl−1 input vectors of length Hl−1

•
input matrix Xl−1
∈
•
φl(Al) ∈ RHl×Cl: Cl output vectors of length Hl output matrix Xl =
kernel tensor Wl ∈ RKl×Cl−1×Cl: Cl filters, each of Cl−1 kernels of kernel size Kl
•
bias vector bl ∈ RCl: one bias value for one output vector
•
Al ∈ RHl×Cl:
activation matrix
•

$$[\mathbf{A}_{l}]_{h o}=\mathbf{\nabla}\mathbf{\nabla}$$


$$\sum_{i=1}^{K_{l}}\sum_{c=1}^{C_{l-1}}[\mathbf{W}_{l}]_{i c o}[\mathbf{X}_{i}]_{h+i-1,c}+[\underline{{{b}}}_{l}]_{o},\quad1\leq h\leq H_{l}=H_{l-1}-K_{l}+1,1\leq o\leq C_{l}.$$

![185_image_0.png](185_image_0.png) 

# 3D Convolutional Layer

In some applications like computer tomography (CT) and magnetic resonance tomography (MRT), the input data is a 3D volume image. In this case, often a 3D CNN 
is required with 5D kernel tensors Wl ∈ RKl×Kl×Kl×Cl−1×Cl. The typical kernel size is 3 3 3 or 5 5 5.

×
×
×
×

## E7.2: Convolution As A Matrix Multiplication

The convolution operation in a convolutional layer can be rewritten as a matrix multiplication as in a dense layer. For simplicity, we consider a 1D convolutional layer with the input vector [x1, . . . , xH]T , the kernel [w1, . . . ,wK]T ,Cl−1 = Cl = 1 and zero bias.

The activation is then

$$a_{h}=\sum_{i=1}^{K}w_{i}x_{h+i-1},\quad1\leq h\leq H-K+1$$


 

 

 


+
or in matrix notation

   . . . wK  x1 x2...xH .  =  w1  a1 . . . wK w1 a2  . . . wK ... ...  ...  aH−K+1 w1
The corresponding (H
1) ×
weight matrix is a Toeplitz band matrix with K
H
+
−
obvious parameter **sharing**. It contains only K parameters instead of (H − K + 1)H

for a dense layer.

# Variants Of Convolution: Padded Convolution

Zero-padding the input with P ∈ N zeros at each side before convolution.

$$\mathrm{~No~padding~}(K_{l}=3)$$
$$\mathrm{~\mathrm{~\mathrm{Padding~with~}}}P=1\;(K_{l}=3)$$
l - 1 o o o o o C o o O o O C l o o o o o o o o

![188_image_0.png](188_image_0.png)

Effect: Larger output size. In particular, if P = K = 1 for odd K i , the output has the same size H 1 = H 1 –1 + 2P − K 1 + 1 = H 1 –1 as the input. This is desired in some applications.

# Variants Of Convolution: Strided Convolution

Move the kernel each time by S  ∈ N positions instead of one position.

$$\mathrm{{\bf~Stride\;1}}$$
$$\mathrm{Stride\2}$$

![189_image_1.png](189_image_1.png)

![189_image_0.png](189_image_0.png)

![189_image_3.png](189_image_3.png)

![189_image_2.png](189_image_2.png)

Effect: Downsampling or subsampling or decimation of the output of a normal convolution with stride 1 by factor S , i.e. keep every S -th value of the output. This reduces the output size.

# Variants Of Convolution: Dilated Convolution

Apply convolution to input samples with a dilation distance D ∈ N instead of D = 1.

$$\mathrm{Dilation~distance~1}$$
$$\mathrm{\ddot{D}l i a t i o n\ o t i a t i o n\ o t i s t i a n c e\ 2}$$

![190_image_1.png](190_image_1.png)

$$\begin{array}{l l l}{{\begin{array}{l l}{{\bigcirc}}&{{}}&{{\bigcirc}}\\ {{}}&{{}}&{{}}\end{array}}}\\ {{\begin{array}{l l}{{\bigcirc}}&{{}}&{{}}\end{array}}}\\ {{\begin{array}{l l}{{\bigcirc}}&{{}}&{{}}\end{array}}}\\ {{\begin{array}{l l}{{\bigcirc}}&{{}}\end{array}}}\end{array}}}\end{array}$$

![190_image_0.png](190_image_0.png)

Effect: (Polyphase) downsampling of the input by factor D . This increases the receptive field without increasing the kernel width.

All modifications padding, stride and dilated convolution can be used in combination.

a

$$\begin{array}{c c c}{{l-1}}&{{\mathrm{O}}}\\ {{}}&{{}}&{{}}\\ {{}}&{{}}&{{\underline{{l}}}}\end{array}$$

# 1 1 Convolution ×

1 1 convolution, pointwise **convolution**: kernel size K = 1
×

$$H\times W\times C*1\times1\times C\times\bar{C}\to H\times W\times\bar{C}$$

no spatial processing
•
combine input channels at every pixel

•

![191_image_0.png](191_image_0.png) 
Purposes:

combine channels

•
reduce the number of input channels for the next layer (depth shortening)

•
higher nonlinearity of DNN due to one additional φl()
•

## Convolution And Approximations 1) Standard Convolution

3D convolution: H
W
Cl−1 K
K
Cl−1 Cl →
H
W
Cl

×
×
×
×
×
×
×
•
∗
HWK2Cl−1Cl N×,1
•
=
2) spatial and depthwise separable **convolution**

depthwise (1D) convolution: H
W
Cl−1 K
1 1 Cl−1 H
W
Cl−1
×
×
×
×
×
×
×
•
∗
→
depthwise (1D) convolution: H
W
Cl−1 1 K
1 Cl−1 H
W
Cl−1
×
×
×
×
×
×
×
•
∗
→
pointwise (1D) convolution: H
W
Cl−1 1 1 Cl−1 Cl →
H
W
Cl

×
×
×
×
×
×
×
•
∗
HWCl−1Cl, N×,2 N×,1 = 2KCl + 1K2 3) depthwise separable **convolution**

N×,2 2*HWKC*l−1
+
•
=
depthwise (2D) convolution: H
Cl−1 1 Cl−1 Cl−1 W
K
K
H
W
×
×
×
×
×
×
×
•
∗
→
pointwise (1D) convolution: H
W
Cl−1 1 1 Cl−1 Cl →
H
W
Cl

×
×
×
×
×
×
×
•
∗
HWCl−1Cl, N×,3 N×,1 = 1Cl + 1K2 4) depth **shortening**

HWK2Cl−1 N×,3
+
•
=
C¯l−1 C¯l−1 pointwise (1D) convolution: H
W
Cl−1 1 1 Cl−1 H
W
×
×
×
×
×
×
×
•
∗
→
C¯l−1 C¯l−1 Cl, C¯l−1 3D convolution: H
Cl →
Cl−1 W
K
K
H
W
×
×
×
×
×
×
×
<
•
∗
HWK2C¯l−1Cl, N×,4 N×,1 = C¯l−1 K2Cl + C¯l−1 Cl−1 ≈ C¯l−1 Cl−1

+
HWCl−1C¯l−1 N×,4
+
•
=
7.3 Downsampling layers 7-13

# Max Pooling

Before max pooling After 2 2 max pooling
×

![193_image_0.png](193_image_0.png) 

![193_image_1.png](193_image_1.png)

E7.3: Max pooling

![193_image_2.png](193_image_2.png)

Vertical edges

![193_image_5.png](193_image_5.png)

2x2 max pooling

![193_image_4.png](193_image_4.png)

![193_image_3.png](193_image_3.png)

# Effects Of Pooling

Reduced spatial size of the feature maps

•
reduced computational and memory complexity
→
One pooling layer is often invariant to a small translation (<
stride) of the input.

•

![194_image_0.png](194_image_0.png) 

2 

![194_image_1.png](194_image_1.png) 

A stack of convolutional and pooling layers is **invariant** to a large translation of the input. This means, the classification output of a CNN is independent of the position of the object in the input image as desired.

Translation-invariant is different from *translation-equivariant.*

Hierarchical feature learning
•
reduce spatial resolution to derive abstract high-level representations of the in-
–
put image (e.g. class, image content, semantic meaning) from its low-level detailed features (e.g. edges, corners, shapes) 
learn image structure instead of pixels, useful to avoid overfitting
–

# Upsampling In Signal Processing

Upsampling in signal processing is an interpolation applied to a signal to enhance its sampling rate (resolution) without changing its waveform/shape. For an 1D signal x(n)

and an integer upsampling **factor** S ,
insert S − 1 zeros between each pair of adjacent samples of x(n):
•

 
 
$$\ldots,x(n-1),\underbrace{0,\ldots,0}_{S^{-1}}x(n),\ldots$$  $\bullet$ smooth out the discontinuities with a fixed lowpass filter (convolution).  

 


![195_image_0.png](195_image_0.png)


 


+

## E7.4: Upsampling Vs. Unpooling

| lowpass    |           |                |    |    |    |    |
|------------|-----------|----------------|----|----|----|----|
| 28original | unpooling | zero insertion | +  |    |    |    |
| 28         | 56        | 56             | 56 | 56 | 56 | 56 |
| ×          | ×         | ×              | ×  |    |    |    |

# Deconvolution In Cnn

```
                                               
upsampling with a learnable kernel
                             

```

## Deconvolution In Signal Processing

In signal processing, deconvolution is defined in a different way. It is the process to reverse a previous convolution.

 
x(n) y(n) z(n) h(n) g(n) 



Given an input signal x(n), a digital filter with the impulse response h(n) and frequency response H(ω) changes x(n) by a convolution

$$y(n)=\sum_{i}h(i)x(n-i).$$

In the frequency domain, Y(ω) =
H(ω)X(ω). A second filter in cascade with h(n)

has the impulse response g(n) and frequency response G(ω). It computes z(n) and Z(ω) =
G(ω)Y(ω) =
G(ω)H(ω)X(ω). If G(ω) =
1/H(ω), z(n) =
x(n). In this case, g(n) is called the deconvolution or inverse filter of h(n).

## Different Names In Dl And Signal Processing

The DL community reinvented many new names for known concepts from signal processing, in particular for CNN. Unfortunately, some new names are not consistent to the old ones. This often causes confusion.

| in deep learning    | in signal processing             |                       |                         |                      |
|---------------------|----------------------------------|-----------------------|-------------------------|----------------------|
| Xl                  | feature maps/channels            | signals               |                         |                      |
| Wl                  | kernel                           | impulse response      |                         |                      |
| Pi Pj wi jxh+i,w+j  | convolution                      | correlation           |                         |                      |
| x(n                 | n0) →y(n                         | n0)                   | translation-equivariant | time/shift-invariant |
| −                   | −                                |                       |                         |                      |
| x(n                 | n0) →y(n)                        | translation-invariant | -                       |                      |
| −                   |                                  |                       |                         |                      |
| padding             | zero initialization              |                       |                         |                      |
| stride              | downsampling of output           |                       |                         |                      |
| dilated convolution | polyphase downsampling of input  |                       |                         |                      |
| deconvolution       | learnable upsamlingdeconvolution |                       |                         |                      |
| -                   |                                  |                       |                         |                      |

# Different Normalizations Of Convolutional Layers (1)

There are two possibilities to normalize a layer:

to normalize the weight tensor W,
weight normalization
•
to normalize the activation tensor A.

activation normalization
•
We focus on the widely used activation normalization.

[a*nhwc*] ∈ RB×H×W×C be the activation tensor of a convolutional layer with Let A
=
minibatch size B, height H, width W
and channel depth C. Zero-mean unit-variance a*nhwc*−µ σ normalization
, with and being the mean and std, see ch. 5.

means anhwc ←
µ σ Different normalization schemes estimate µ and σ in different ways:
batch **normalization**: across minibatch, height and
•
weight 1 BHW
PBn=1 PHh=1 PWw=1 for each channel c instance **normalization**: across height and weight 1 HW
•
PHh=1 PWw=1 for each sample n and each channel c layer **normalization**: across channel depth, height and
•
PCc=1 PHh=1 PWw=1 weight 1 CHW
for each minibatch sample n B

![198_image_0.png](198_image_0.png)

# Different Normalizations Of Convolutional Layers (2)

## Comments:

A good estimate of the activation statistics µ and σ requires a large number of i.i.d.

samples of a*nhwc*.

•
[anm] ∈ RB×M consists of B minibatch samples, In BN for dense layers (ch. 5), A
•
=
each being a vector a ∈ RM. Different elements of a are calculated from different row vectors of the weight matrix W
and have different distributions. Hence µ and 1 PBn=1 are estimated across the minibatch only for each element of a.

σ B
In convolutional layers, the same filter kernel is used at all pixels (h, w). Hence all

•
pixels in a channel are assumed to be i.i.d. This justifies the estimation of µ and σ across minibatch and all pixels 1 BHW
PBn=1 PHh=1 PWw=1 in BN.

Sometimes the minibatch size B is very small (even B = 1). Average across minibatch is then less effective. Due to memory limitation, different GPUs use different

•
minibatch sizes for training the same model. This leads to different results if normalized over minibatch. In these cases, instance and layer normalization are used instead of BN.

## Application-Dependent Cnn Architecture

encoder-head architecture From image to number From image to image

(e.g. classification, regression) (e.g. segmentation, translation)

encoder-decoder architecture

![200_image_0.png](200_image_0.png)

or **backbone**: Hierarchical feature learning from high-dimensional input.

Encoder
•
It consists of convolutional, downsampling, normalization, and dropout layers.

Head: Decision making based on the learned high-level features. It consists of

•
reshaping, dense, normalization, and dropout layers.

Decoder: Prediction of high-dimensional output based on learned high-level fea-
•
tures. It consists of convolutional, upsampling, normalization, and dropout layers.

## E7.5: Mnistnet3 (1)

The third neural network, a CNN, for digit recognition on MNIST.

Architecture
(Encoder-head)

![201_image_0.png](201_image_0.png)

$$\mathbf{\tau}\mathbf{n}\mathbf{a}\mathbf{x}\ \mathbf{p}\mathbf{o}\mathbf{o}\mathbf{i}\mathbf{n}\mathbf{g}\ \mathbf{l}\mathbf{a}\mathbf{y}$$

4 convolutional layers (CL) - 2 max pooling layers
- •



$$\mathbf{m}\mathbf{j}=\mathbf{m}\mathbf{a}\mathbf{j}$$
$${\bullet\ \ 2\ \mathrm{dense\layers\left(D L\right)}}$$
1 flatten layer - 2 dense layers (DL)
From shallow to deep layers, the number of channels Cl is increased in parallel to the reduction of their spatial size Hl × Wl in order to avoid too much information loss.

7.7 Architecture of CNNs 7-22 E7.5: MNISTnet3 (2)

TensorFlow Keras **code** for the architecture model 
= keras.Sequential([
keras.layers.Conv2D(32, (3, 3), activation='relu',
input_shape=(28, 28, 1)),
keras.layers.Conv2D(32, (3, 3), activation='relu'),
keras.layers.Dropout(0.25), 
keras.layers.MaxPooling2D((3, 3)), keras.layers.Conv2D(64, (3, 3), activation='relu'), keras.layers.Conv2D(64, (3, 3), activation='relu'),
keras.layers.Dropout(0.25), 
keras.layers.MaxPooling2D((2, 2)),
keras.layers.Flatten(), 
keras.layers.Dense(256, activation='relu'),
keras.layers.Dropout(0.5), 
keras.layers.Dense(10, activation='softmax')])

## E7.5: Mnistnet3 (3) Complexity

Convolutional layer with kernel tensor Wl ∈ RKl×Kl×Cl−1×Cl and output tensor Xl ∈

RHl×Wl×Cl:
•
 Np = K2l Cl−1Cl + Cl, N× = HlWlK2l Cl−1Cl Wl ∈ RMl×Ml−1 and output vector xl ∈ RMl: Np =

Ml(Ml−1 Dense layer with weight matrix
•
1), N×
MlMl−1
+
=

| Weight tensor   | ♯parameters Np   | Output tensor   | ♯multiplications N×   |           |
|-----------------|------------------|-----------------|-----------------------|-----------|
| CL 1            | 3x3x 1x32        | 320             | 26x26x32              | 194,688   |
| CL 2            | 3x3x32x32        | 9,248           | 24x24x32              | 5,308,416 |
| CL 3            | 3x3x32x64        | 18,496          | 6x 6x64               | 663,552   |
| CL 4            | 3x3x64x64        | 36,928          | 4x 4x64               | 589,824   |
| DL 1            | 256x256          | 65,792          | 256                   | 65,536    |
| DL 2            | 10x256           | 2,570           | 10                    | 2,560     |
| MNISTnet3       | 133,354          | 6,824,576       |                       |           |
| MNISTnet1       | E5.1             | 101,770         | 101,632               |           |
| MNISTnet2       | E6.3             | 118,282         | 118,016               |           |

## E7.5: Mnistnet3 (4) Cost Function

categorical cross entropy loss

•

## Optimizer

minibatch size B = 32
•
50 epochs

•
SGD with fixed step size γt = 0.02
•

## Regularization

dropout rate 0.25 for 2 convolutional layers before max pooling
•
dropout rate 0.5 for the first dense layer
•

## E7.5: Mnistnet3 (5)

Results

![205_image_0.png](205_image_0.png) 

no overfitting due to dropout.

•
at the end, the test error rate is 0.48%, i.e., only 48 from 10,000 test images are

•
wrong classified:

![205_image_1.png](205_image_1.png)

E7.5: MNISTnet3 (6)

Visualization of the output Xi of all 4 convolutional layers and 2 dense layers for the

![206_image_0.png](206_image_0.png)

![206_image_1.png](206_image_1.png)

X3, 6x6x64 X4, 4x4x64

![206_image_2.png](206_image_2.png) 

# Mnist Experiments

| Example           | E5.1         | E5.6          | E6.3            | E7.5         |    |
|-------------------|--------------|---------------|-----------------|--------------|----|
| Network           | 1D MNISTnet1 | 1D MNISTnet1  | 1D MNISTnet2    | 2D MNISTnet3 |    |
| Hl                | 784/128/10   | 784/128/10    | 784/128/128/10  | 4 CL, 2 DL   |    |
| φl                | ReLU/softmax | ReLU/softmax  | ReLU/softmax    | ReLU/softmax |    |
| Purpose           | baseline     | optimizations | regularizations | solution     |    |
| minibatchB        | 32           | varying       | 32              | 32           |    |
| epochs            | Wepoch       | 30            | 30              | 30           | 50 |
| learning schedule | fixed        | fixed         | fixed           | fixed        |    |
| step size γ       | 0.1          | varying       | 0.1             | 0.02         |    |
| l2-penaltyλ       | 0.0002       |               |                 |              |    |
| -                 | -            | -             |                 |              |    |
| dropout rate      | -            | 0.2           | 0.25/0.5        |              |    |
| -                 |              |               |                 |              |    |
| overfitting       | yes          | yes           | no              | no           |    |
| test error rate   | 2%           | 2%            | 2%              | 0.48%        |    |

There are better results than 0.48% in the literature.

## E7.6: Fmnistnet

Exactly the same CNN and TensorFlow code of MNIST can be applied to the Fashion MNIST dataset by just changing one line:

(x_train, y_train), (x_test, y_test) 
= keras.datasets.fashion_mnist.

![208_image_0.png](208_image_0.png)

$$\mathrm{\Large{-train)\,,\,\,\,(x_{-}t e s t,\,\,y_{-}t e s t)\,=}}$$

## E7.7: Cifar10Net (1)

A CNN for image recognition on CIFAR-10: 32x32 color images →
10 classes

![209_image_0.png](209_image_0.png)

4 convolutional layers (CL)

•
2 max pooling layers

•
1 flatten layer

•
2 dense layers (DL)

•
The same architecture as MNISTnet3 except for kernel 5x5 in the first two CL.

Architecture
(Encoder-head)

## E7.7: Cifar10Net (2) Data Augmentation "On The Fly"

random rotation of images up to
±10o

•
±10% of the image size

•
random translation of images up to
•
random horizontal flip of images

![210_image_0.png](210_image_0.png)

Data augmentation significantly reduces overfitting.

•
Image classification on CIFAR10 is much more challenging than MNIST.

•

## E7.8: Semantic Image Segmentation (1)

Image **segmentation**: Pixelwise classification, see E1.5

## Dataset

Cityscapes (Daimler)

•
5000 images: 2975/500/1525 for training/validation/test

•
8 classes: road/construction/object/nature/sky/human/vehicle/global

•
image size downscaled to 224x224 to reduce complexity
•
48h training time on a GPU
•
Based on a Master thesis at ISS (2016)1

## E7.8: Semantic Image Segmentation (2)

Architecture
(encoder-decoder)

fully convolutional network consisting of 30 convolutional layers only
•
a each layer with padding, BN and ELU activation function
•
first 15 (encoding) CL: smaller image size by max pooling, more feature maps

•
last 15 (decoding) CLs: larger image size by unpooling, less feature maps

•
no flatten and dense layers

•
shortcuts to avoid vanishing gradient

![212_image_0.png](212_image_0.png)

•

## E7.8: Semantic Image Segmentation (3) Network Parameters

| Layer         | Kernel      | Output       |               |          |
|---------------|-------------|--------------|---------------|----------|
| Layer         | Kernel      | Output       | KlxKlxCl−1xCl | HlxWlxCl |
| KlxKlxCl−1xCl | HlxWlxCl    |              |               |          |
| Input         | 224x224x 3  |              |               |          |
| 3×            | . . .       | . . .        | . . .         |          |
| 3×            | 14x 14x 512 |              |               |          |
| Unpooling     | 2x2         | 28x 28x 512  |               |          |
| 3×CL          | 3x3         |              |               |          |
| 3x3x...       | 224x224x 64 |              |               |          |
| Max Pooling   | 2x2         | 112x112x 64  |               |          |
| 3×CL          | 28x 28x 256 |              |               |          |
| Unpooling     | 2x2         | 56x 56x 256  |               |          |
| 3×CL          | 3x3         |              |               |          |
| 112x112x 128  |             |              |               |          |
| Max Pooling   | 2x2         | 56x 56x 128  |               |          |
| 3×CL          | 3x3         | 56x 56x 128  |               |          |
| Unpooling     | 2x2         | 112x112x 128 |               |          |
| 3×CL          | 3x3         |              |               |          |
| 56x 56x 256   |             |              |               |          |
| Max Pooling   | 2x2         | 28x 28x 256  |               |          |
| 3×CL          | 3x3         | 112x112x 64  |               |          |
| Unpooling     | 2x2         | 224x224x 64  |               |          |
| 2×CL          | 3x3         |              |               |          |
| 28x 28x 512   |             |              |               |          |
| Max Pooling   | 2x2         | 14x 14x 512  |               |          |
| 3×CL          | 3x3         | 224x224x 64  |               |          |
| CLCL          | 3x3         |              |               |          |
| CL            | 3x3         | 14x 14x1024  | 224x224x 8    |          |
| Softmax       | 3x3         |              |               |          |
| . . .         | . . .       | . . .        | 224x224x 8    |          |

44 million parameters

•
≈
34 billion of multiplications for one image
•
≈

![214_image_0.png](214_image_0.png)

E7.8: Semantic image segmentation (4)

Results

```
 Test
    
images
     

```

CNN

```
   
output
     

```

Ground truth

# Fir Filter Vs. Cnn

| FIR filter                      | CNN                                  |                     |    |
|---------------------------------|--------------------------------------|---------------------|----|
| Architecturebasic operation     | convolution                          | convolution         |    |
| activation function             | no                                   | yes                 |    |
| system behavior                 | linear                               | nonlinear           |    |
| cascaded stages                 | no                                   | yes                 |    |
| input feature maps              | 1                                    | ≥                   | 1  |
| small                           | large                                |                     |    |
| Designcomplexity                | calculated from filter specification | learned from data   |    |
| Purpose coefficientsapplication | linear filtering                     | nonlinear filtering |    |

CNN can be roughly interpreted as a nonlinear generalization of FIR filter whose coefficients are learned from examples instead of human designed.

# Classical Networks And Modules

- For image classification

![216_image_0.png](216_image_0.png)

Inception-v4, MobileNet
- For image segmentation: U-Net, V-Net
- For object detection: Faster R-CNN, YOLO

# Lenet⋆

LeNet2: Yann LeCun3 et al (1998) for MNIST digit recognition
•
60k parameters, 0.96% error rate

•
5 weight layers (3 CL
2 dense), ≈
+
•
great impact to today's CNN architecture consisting of convolutional layers 
–
sub-sampling layers 
–
dense layers

–
Lesson: First successful CNN
•

![217_image_0.png](217_image_0.png)

# Alexnet (1)★

- AlexNet: Alex Krizhevsky et al from University of Toronto for ILSVRC 4
- 8 weight layers (5 CL + 3 dense), ≈ 60M parameters, 16.4% top-5 error rate, won ILSVRC 2012
- first time ReLU instead of sigmoid, heavy data augmentation, dropout in CL
- Lesson 1: First large-scale CNN, beginn of DNN
- Lesson 2: Large dense layers cause heavy memory usage

![218_image_0.png](218_image_0.png)

# Alexnet (2)⋆

| Layer   | Kernel      | Output             | Np        |
|---------|-------------|--------------------|-----------|
| 0       | input       | 224x224x3          |           |
| 1       | CL          | 11x11x3x96, S=4    | 54x54x96  |
| 2       | max-pooling | 2x2                | 27x27x96  |
| 3       | CL          | 5x5x48x256, P=2    | 27x27x256 |
| 4       | max-pooling | 2x2                | 13x13x256 |
| 5       | CL          | 3x3x256x384, P=1   | 13x13x384 |
| 6       | CL          | 3x3x192x384, P=1   | 13x13x384 |
| 7       | CL          | 3x3x192x256, P=1   | 13x13x256 |
| 8       | max-pooling | 3x3                | 4x4x256   |
| 9       | flatten     | 4096               |           |
| 10      | dense       | 4096               | 16M       |
| 11      | dense       | 4096               | 16M       |
| 12      | output      | 1000-class softmax | 4M        |

## Vggnet (1)⋆

VGGNet: Karen Simonyan et al from the Visual Geometry Group at University of

•
Oxford for ILSVRC5 6 different architectures evaluated. The most famous ones are 
•
VGG16: 16 weight layers, 138M parameters 
–
VGG19: 19 weight layers, 144M parameters, 7.3% top-5 error rate

•
–
simple architecture: 3x3 kernels, 2x2 max-pooling doubled channel depth
+
huge Np due to large dense layers
•
2. place in ILSVRC 2014
•
Lesson: Deep narrow kernels more *memory* efficient than shallow wide *kernels* For example, a stack of four 3x3 kernels has the same receptive field as a single 9x9 kernel, but has 
•
a higher nonlinearity
–
less parameters 
–
better performance than AlexNet

–

# Vggnet (2)★

| ConvNet Configuration       |             |           |           |           |           |
|-----------------------------|-------------|-----------|-----------|-----------|-----------|
| A-LRN                       | B           | D         | E         |           |           |
| A                           | C           |           |           |           |           |
| 11 weight                   | 11 weight   | 13 weight | 16 weight | 16 weight | 19 weight |
| layers                      | layers      | layers    | layers    | layers    | layers    |
| input (224 × 224 RGB image) |             |           |           |           |           |
| conv3-64                    | conv3-64    | conv3-64  | conv3-64  | conv3-64  | conv3-64  |
| LRN                         | conv3-64    | conv3-64  | conv3-64  | conv3-64  |           |
| maxpool                     |             |           |           |           |           |
| conv3-128                   | conv3-128   | conv3-128 | conv3-128 | conv3-128 | conv3-128 |
| conv3-128                   | conv3-128   | conv3-128 | conv3-128 |           |           |
| maxpool                     |             |           |           |           |           |
| conv3-256                   | conv3-256   | conv3-256 | conv3-256 | conv3-256 | conv3-256 |
| conv3-256                   | conv3-256   | conv3-256 | conv3-256 | conv3-256 | conv3-256 |
| conv1-256                   | conv3-256   | conv3-256 |           |           |           |
| conv3-256                   |             |           |           |           |           |
| maxpool                     |             |           |           |           |           |
| conv3-512                   | conv3-512   | conv3-512 | conv3-512 | conv3-512 | conv3-512 |
| conv3-512                   | conv3-512   | conv3-512 | conv3-512 | conv3-512 | conv3-512 |
| conv1-512                   | conv3-512   | conv3-512 |           |           |           |
| conv3-512                   |             |           |           |           |           |
| maxpool                     |             |           |           |           |           |
| conv3-512                   | conv3-512   | conv3-512 | conv3-512 | conv3-512 | conv3-512 |
| conv3-512                   | conv3-512   | conv3-512 | conv3-512 | conv3-512 | conv3-512 |
| conv1-512                   | conv3-512   | conv3-512 |           |           |           |
| conv3-512                   |             |           |           |           |           |
| maxpool                     |             |           |           |           |           |
| FC-4096                     |             |           |           |           |           |
| FC-4096                     | VGG16 VGG19 |           |           |           |           |
| FC-1000                     |             |           |           |           |           |
| soft-max                    |             |           |           |           |           |

# Googlenet (1)⋆

GoogLeNet or **Inception-v1**: Christian Szegedy et al from Google for ILSVRC6
•
22 weight layers, won ILSVRC 2014, 6.7% top-5 error rate

•
•
only 5M parameters because of global average pooling layer instead of flatten layer 
–
many 1 1 convolutions

×
–
stack of 9 inception modules

•

![222_image_0.png](222_image_0.png)

## Googlenet (2)⋆

Inception **module** (movie Inception: dream in dream in dream, i.e. deep dream)
•
a carefully designed ready-to-use module

•
Lesson 1: Parallel feature extraction with different kernel *widths* (1x1, 3x3, 5x5)
Lesson 2: 1x1 convolution helps to reduce channel depth and complexity of the

•
next *layer*

![223_image_0.png](223_image_0.png)

# Resnet (1)⋆

ResNet: Kaiming He et al from Microsoft for ILSVRC7
•
5 different architectures RetNet-18/34/50/101/152 evaluated
•
RetNet-152: 152 weight layers, 60M parameters, won ILSVRC 2015
•
3.57% top-5 error rate, better than human
(5%)

•
stack of residual blocks

•
Lesson: Residual connections mandatory for deep *networks*

•

![224_image_0.png](224_image_0.png)

# Resnet (2)

## Residual Block:

Let x →
H(x) be the desired mapping to be learned. F(x) =
H(x) −
is the residual x mapping. It is non-trivial to realize the identity mapping H(x) =
by a stack of x nonlinear layers, but it is trivial to realize the corresponding residual mapping F(x) = 
H(x) −
0.

x
=
Forward pass: Provide low-level features to high-level layers

•
Backward pass: Shortcut connections help to avoid vanishing gradients.

•

![225_image_0.png](225_image_0.png)

# Inception-V4★

- Inception-v4 or Inception-ResNet: Christian Szegedy et al (2016) from Google for ILSVRC8
- combine Inception module and residual block: Inception-ResNet-A/B module - 3.08% top-5 error rate
- Lesson: Combine existing good ideas

![226_image_0.png](226_image_0.png)

# Comparison Of Dnns For Image Classification★

- A. Canziani et al (2016) 9
- ILSVRC top-1 accuracy vs. memory usage and number of operations for inference

![227_image_0.png](227_image_0.png)

# Mobilenet⋆

MobileNet: A. G. Howard Woo et al (2017)10
•
depthwise separable convolution consisting of 
•
use depthwise convolution: 2D spatial convolution for each feature map individu-
–
ally pointwise convolution 1x1 convolution
=
–
less parameters and operations at the price of a small performance degradation
•
applicable for all feature extractors in image segmentation, object detection and
•
image segmentation Lesson: For low-cost mobile/embedded *systems*

•

# U-Net

- U-Net: Olaf Ronneberger et al (2015) for 2D medical image segmentation¹¹
- Lesson: symmetric encoder-decoder + residual connections

![229_image_0.png](229_image_0.png)

# V-Net⋆

V-Net: Fausto Milletari et al (2016) for 3D medical image segmentation12
•
Lesson: 3D *U-Net*

![230_image_0.png](230_image_0.png)

•
+

# Object Detection

localize and classify objects in the input image, multi-classes, multi-instances

•
bounding box regression and box content classification
•
use feature extractors from image classification net (e.g. ResNet)

•
two-stage **detector** (e.g. faster R-CNN): a) generate regions of interest (RoI), b)
localization and classification for each RoI

•
one-stage **detector** (e.g. YOLO): no region proposal, divide the image into square

•
grids and do bounding box regression and classification for each grid

![231_image_0.png](231_image_0.png)

# Faster R-Cnn⋆

Faster **R-CNN**: Shaoqing Ren et al (2016)13
•
calculate feature maps from the input image once

•
•
a region proposal network (RPN) generates regions of interest (RoI) from the feature maps bounding box estimation and box classification for each RoI

![232_image_0.png](232_image_0.png)

•

# Yolo⋆

You only look once (**YOLO**): Joseph Redmon et al (2016)14
•
divides each input image into grids

•
bounding box estimation and box classification in each grid
•
delete boxes without objects and merge highly overlapping boxes

•
faster than Faster R-CNN
•
YOLOv2, YOLOv3 and YOLO9000 available

![233_image_0.png](233_image_0.png) 
•

## Memory Of Non-Recursive And Recursive Filter (1)

A general digital filter IIR(N, M) with the input x(n) and output y(n) is described by the difference equation

$$y(n)=\sum_{i=0}^{M}b_{i}x(n-i)-\sum_{i=1}^{N}a_{i}y(n-i),$$

see last section of ch. 3. The simplest recursive filter or IIR **filter** IIR(1, 0) of first order is given by y(n) = b0x(n) − a1y(n − 1). By using β = −a1,

$$y(n)=\beta y(n-1)+b_{0}x(n)=\ldots=b_{0}[x(n)+\beta x(n-1)+\ldots+\beta^{n-1}x(1)]+\beta^{n}y(0).$$

This means, y(n) depends on all input samples x(n), . . . , x(1) from the beginning of the measurement till the current time instance n.

In comparison, a non-recursive filter or FIR **filter** IIR(0, M) of order M

$$y(n)=\sum_{i=0}^{M}b_{i}x(n-i)$$

depends only on the last M
1 input samples.

+

# Memory Of Non-Recursive And Recursive Filter (2)

![235_image_0.png](235_image_0.png) 

Due to the feedback, IIR filter is the natural choice to model autoregressive systems

•
with a temporal correlation.

Roughly spoken, CNN is a nonlinear extension of FIR filter,

•
RNN is a nonlinear extension of IIR filter.

•

## Feedforward Vs. Feedback Neural Networks

| Feedforward neural networks   | Feedback neural networks             |                                      |
|-------------------------------|--------------------------------------|--------------------------------------|
| architecture                  | dense (ch. 4), CNN (ch. 7)           | RNN (ch. 8)                          |
| feedback                      | no                                   | yes                                  |
| memory of neuron              | no                                   | yes                                  |
| memory of network             | short, depend on kernel size         | long                                 |
| input of network              | vector/matrix/tensor                 | sequence of vectors/matrices/tensors |
| output of network             | probabilities or                     | probabilities or                     |
| vector/matrix/tensor          | sequence of vectors/matrices/tensors |                                      |
| temporal correlation          | not considered                       | exploited                            |

RNN processes sequential data with temporal correlation:

speech (spoken language) and text (written language),

•
e.g., "Prof Yang at Uni of Stuttgart teaches deep learning."

•
audio video (sequence of images)

- •
 . . .

## E8.1: Applications Of Rnn

a) Speech recognition: Translate a spoken speech (a sequence of speech samples) to a text (a sequence of letters and words)

b) Translation: Translate a speech/text from one language to another c) Optical character recognition (OCR)/Handwriting recognition: Translate an image of PDF/handwriting to a text d) Image caption: Return a short text description for a given image. It typically uses a CNN to analyze the image and a RNN to generate natural language description e) Music composition: Generate a music, a sequence of notes f) . . .

In all applications above, the input and/or output are sequential data (speech, text, music) with a strong temporal correlation described by phonetics, grammar, melody etc.

# Unfolding Of Recurrent Layer L

Network graph Unfolded computational graph

Wl,s Wl,s Wl,s Wl,s s l ( n ) s l(0) s l(1) s l(2) s l ( B ) - . . . Wl,x Wl,x Wl,x ( n ) xl−1(1) xl−1(2) xl−1 ( B ) xl−1 . . .


 

+
Unfolding a time recursion

```
s

 l
 
 (
 
  n

```

)

=
f( xl−1 
(

n

```
), s
  
    l
    
     (
     
      n

```

1)) along the time axis n
•
=
− 
1, 2, . . . ,
B
yields a computational graph with repetitive structure. It is a convenient way to visualize the calculations in a RNN.

The unfolded graph contains no feedback (cycles) and can be handled in the same

•
way as feedforward networks.

The same parameters Wl,x, Wl,s

, bl are used in the unfolded graph for all time in-
•
stances.

State initialization s l(0): a) zero, b) random, c) last state of previous minibatch

•

# Single-Neuron Vs. Cross-Neuron Feedback

Recurrent layer l:

![239_image_0.png](239_image_0.png) 

```
feedforward:
           
             Wl,s
                

```

0
•
=
single-neuron feedback:

![239_image_1.png](239_image_1.png)

•
cross-neuron feedback:

•

# Unfolded Recurrent Layer In Details

with Ml−1 3, Ml =
4 and single-neuron feedback
=

![240_image_0.png](240_image_0.png) 

## E8.2: A Simple Rnn

one input layer x
(

n
)

•

```
one recurrent layer
               
                s
                
                 (
                 
                  n

```

)

=
φ1( W
(

n
)

+

```
s

 s
 
  (
  
    n

```

1)

+
b1 
)

W
•
x x
−
one output layer ˆy

(

n
)

=
2

(

W

```
2

 s
 
  (
  
   n

```

)

+

```

 2
 
  ), φ
          2()
            
               =

```

b softmax() for classification φ
•
loss l

(

n; θ) between RNN output ˆy

(

n) and ground truth
(

n
)

- 
y 1 B
P
B
cost function
(

θ
)

=
l

(

n; θ) for one minibatch 1 L
B
≤
≤
•
n n

=1 contains all parameters from s

, b1 2, b θ W
x, W 
, W
- 
2

| l ( n; θ )    | l(1; θ)          | l(2; θ)   | l ( B; θ )   |       |        |
|---------------|------------------|-----------|--------------|-------|--------|
| . . .         |                  |           |              |       |        |
| yˆ ( n)       | yˆ(1)            | ˆy(2)     | ˆy ( B)      |       |        |
| decode output | . . .            |           |              |       |        |
| W2            | capture temporal |           |              |       |        |
| ( s n)        | W                | s(0)      | s(1)         | s(2)  | ( Bs ) |
| -             | s                | . . .     |              |       |        |
| correlation   |                  |           |              |       |        |
| Wx            | encode input     |           |              |       |        |
| ( n)          | x(1)             | x(2)      | . . .        | x( B) |        |
| x             |                  |           |              |       |        |

In practice, an RNN may contain multiple recurrent layers.

# Organisation Of Input Data

![242_image_0.png](242_image_0.png)

d contains 1 the original input data ˜x

(

n
)

∈
R
d time series

•
T
(

n T
(

n T
(

n

```
−1)]
  
   T
   
     ∈

```

form a new sequence of input vectors x
(

n
)

=
[˜x
), x˜

+1), . . . ,
x˜
P
+
•
RPd by using a sliding window of length P
1. This corresponds to the non-
≥
recursive part of a recursive filter.

the choice of the window length P
depends on the application
•
divide
(

n) into non-overlapping minibatches of length B
•
x
(tB
), 1 B; t

=
0, 1, . . . is the t-th minibatch of input for the RNN
≤
≤
+
•
x n n

![243_image_0.png](243_image_0.png)

B
paths from W
L
(

θ) through

```
l

 (
 
  B; θ), namely those over
                                      
                                        s(1), . . . ,
                                                      s
                                                      
                                                        (
                                                        
                                                         B),
                                                             

```

s to

•
−1 paths from
(

θ) through l

(

B

```
−1;
  
   θ), namely those over
                  
                    s(1), . . . ,
                           s
                           
                            (
                            
                            B

```

−1),

B
W
L
s to

•

# How Many Paths From W L ( Θ ) In E8.2? To X

![244_image_0.png](244_image_0.png)

Similarly, in

```
s

 (
 
  n

```

)

=
φ1( W
(

n
)

+

```
s

 s
 
  (
  
    n

```

1)

+
b1), both x and

```
s

 (
 
  n

```

1) depend on W
W
x x
−
− 
W
x. Hence the backpropagation carries over both layers and time.

## Motivation Of Bidirectional Recurrent Neural Network

RNN: Recursive calculation s(n 1) →
s(n) in the forward time direction. This

−
corresponds to a causal behavior: Presence depends on past.

In some applications, however, a non-causal behavior is desired because the output y ˆ(n) may depend on both past and future of input x(n), e.g.,

Text recognition: Linguistic dependency of current word on past and future words.

•
Speech recognition: Co-articulation, i.e., the current phoneme also depends on the

•
next few phonemes.

## (Brnn)2 Bidirectional Recurrent Neural Network

presents one minibatch of training data in the forward- and backward-time direc-
•
tion to two separate recurrent layers and concatenation of both outputs.

•
This provides the next layer with complete past and future information of the input sequence at each time instance.

![246_image_1.png](246_image_1.png)

![246_image_0.png](246_image_0.png)

![246_image_2.png](246_image_2.png)

# A Lstm Cell

A
long short-term memory
(LSTM) cell/neuron/unit/block replaces a normal recurrent neuron in RNN. It contains3
•
a memory storing the state s(n) at time n like a recurrent neuron and

•
three multiplicative gates, the input/forget/output **gate**, which control the write/reset
/read operation of the memory state.

## A Lstm Layer L (1)

containing Ml LSTM cells:
xl−1(n) ∈ RMl−1 layer input at time n xl(n) ∈ RMl layer output at time n sl(n) ∈ RMl memory state at time n, sl(n) , xl(n) in contrast to RNN

il(n) ∈ RMl input gate **signal** at time n f l(n) ∈ RMl forget gate **signal** at time n ol(n) ∈ RMl output gate **signal** at time n

# A Lstm Layer L (2)

![248_image_0.png](248_image_0.png) 

•
memory gate: elementwise multiplication
⊙
σ

![248_image_1.png](248_image_1.png)

neuron with any

![248_image_2.png](248_image_2.png)

activation function
A LSTM layer l (3)
involving 8 weight matrices.

Gate **signals**

$$\begin{array}{r c l}{{}}&{{\underline{{{i}}}_{l}(n)\;=\;\sigma({\bf W}_{l,i x}\underline{{{x}}}_{l-1}(n)+{\bf W}_{l,i o}\underline{{{x}}}_{l}(n-1)+\underline{{{b}}}_{l,i}),}}\\ {{}}&{{f_{l}(n)\;=\;\sigma({\bf W}_{l,f x}\underline{{{x}}}_{l-1}(n)+{\bf W}_{l,f o}\underline{{{x}}}_{l}(n-1)+\underline{{{b}}}_{l,f}),}}\\ {{}}&{{\underline{{{o}}}_{l}(n)\;=\;\sigma({\bf W}_{l,o x}\underline{{{x}}}_{l-1}(n)+{\bf W}_{l,o o}\underline{{{x}}}_{l}(n-1)+\underline{{{b}}}_{l,o}).}}\end{array}$$


+
They all have the range (0, 1). Typically, Wl,∗o are diagonal, i.e., each gate signal is affected by the output of the same LSTM cell.

## Update Memory State

$$\underline{{{s}}}_{l}(n)=\underline{{{\int_{-l}^{\cdot}}}}(n)\odot\underline{{{s}}}_{l}(n-1)+\underline{{{i}}}_{l}(n)\odot\phi_{l}(\mathbf{W}_{l,s s}\underline{{{x}}}_{l-1}(n)+\mathbf{W}_{l,s o}\underline{{{x}}}_{l}(n-1)+\underline{{{b}}}_{l,s})$$

## Layer Output

$$\underline{{{x}}}_{l}(n)=\O_{\underline{{{o}}}_{l}}(n)\odot\phi_{l}(\underline{{{s}}}_{l}(n))$$

In contrast to RNN which always processes each state information, a LSTM neuron is able to selectively remember/forget things which are relevant/irrelevant.

# Function Of Gates (1)

LSTM is the most successful type of RNN.

In the forward pass:

The gating signals il(n), f l(n), ol(n) are time-varying. This enables a dynamic and flexible short/long-term storage and access of information.

•

| input gate            | forget gate                | output gate      | memory state   |
|-----------------------|----------------------------|------------------|----------------|
| open                  | write into memory          |                  |                |
| memory protected from |                            |                  |                |
| close                 | overwriting and forgetting |                  |                |
| open                  | open                       | read from memory |                |
| close                 | clear memory               |                  |                |

Instead of making manual decisions for opening/closing gates, an LSTM learns

•
to open/close the gates automatically, conditioned on the input xl−1(n) and output xl(n − 1).

# Function Of Gates (2)

In the backward pass:

Conventional RNN:

•

 
$$\underline{{{s}}}_{l}(n)=\phi_{l}(\mathbf{W}_{l,x}\underline{{{x}}}_{l-1}(n)+\mathbf{W}_{l,s}\underline{{{s}}}_{l}(n-1)+\underline{{{b}}}_{l})$$


→
 

$\underline{s}_{i}(n-1)$ is inside the activation function $\phi_{i}(\cdot)$. Hence vanishing gradient happens during the backpropagation $\frac{\partial L}{\partial\underline{s}_{i}(n)}\rightarrow\frac{\partial L}{\partial\underline{s}_{i}(n-1)}$ if $\left|\frac{\partial\phi_{i}(a)}{\partial a}\right|<1$.  
LSTM:

•

$\underline{s}_{i}(n)=\underline{f}_{i}(n)\odot\underline{s}_{i}(n-1)+\underline{i}_{i}(n)\odot\phi_{i}(\mathbf{W}_{i,s,\mathbf{x}\underline{x}_{i-1}}(n)+\mathbf{W}_{i,s\odot\underline{x}_{i}}(n-1)+\underline{b}_{i,s})$  $\underline{s}_{i}(n-1)$ is outside the activation function $\phi_{i}(\cdot)$. If the forget gate is open, i.e. $\underline{f}_{i}(n)$
close to one, there is no gradient attenuation during the backpropagation
 ∂sl(n) 
∂L 
∂sl(n − 1). Hence an LSTM can have a long memory without vanishing gradients.

## Modifications And Extensions Of Lstm (1)

Peephole-LSTM: The gate signals also depends on the previous state sl(n − 1).

This leads to a more complex LSTM layer involving 11 weight matrices.

•
Gate **signals**

 
σ(Wl,ix xl−1(n) + Wl,issl(n − 1) + Wl,ioxl(n − 1) + bl,i), f l(n) = σ(Wl, f x xl−1(n) + Wl, f ssl(n − 1) + Wl, f oxl(n − 1) + bl, f), il(n) = ol(n) = σ(Wl,ox xl−1(n) + Wl,ossl(n − 1) + Wl,ooxl(n − 1) + bl,o).

## Update Memory State

$${\underline{{s}}}_{l}(n)={\underline{{f}}}_{l}(n)\odot{\underline{{s}}}_{l}(n-1)+i_{l}(n)\odot\phi_{l}({\bf W}_{l,s,x}{\underline{{x}}}_{l-1}(n)+{\bf W}_{l,s o}{\underline{{x}}}_{l}(n-1)+{\underline{{b}}}_{l,s})$$

Layer **output**

$$\underline{{{x}}}_{l}(n)=\underline{{{o}}}_{l}(n)\odot\phi_{l}(\underline{{{s}}}_{l}(n))$$

## Modifications And Extensions Of Lstm (2)

Gated recurrent unit **(GRU**)4 
•
a simplified version of LSTM using only two gates, reset and update gate 
–
less parameters, lower complexity and lower model capacity compared to LSTM
–
comparable performance to LSTM in many applications

 –
bidirectional LSTM
•
ConvLSTM5: Combine LSTM with convolutional layers for video processing, i.e.,

•
replace matrix-vector multiplications Wx by convolutions W
X.

∗
Disadvantages of RNN and LSTM:

more difficult to train due to recurrence, need a larger number of training steps

•
sequential calculation, hard to parallelize for GPU
•

## E8.3: Midi Composition (1)

Musical instrument digital interface (**MIDI**): A digital standard for describing and exchanging music information between electronic musical instruments

## Training Set

single-channel (d 1) MIDI data ˜x(n)

•
=
integer coded MIDI data ˜x(n) ∈ {1, 2, . . . , 375} for C = 375 possible single tones
(pressing one key) and accords (pressing multiple keys)

•
188 MIDI files of classical piano (Bach, Beethoven, Brahms, Chopin, . . .)
•
•
tone duration roughly 1/4s in total 222407 training tones (total duration 15.4h)

•
[ ˜x(n), x˜(n+1), . . . , *x˜(n*+P−1)]T ∈ N100
•
sliding window length 100, i.e. x(n) =
P
•
=
minibatch size B = 128 Task: MIDI "composition". Given the first P
100 MIDI tones, the RNN should
=
continue to "compose" in a sliding-window way:


 
 

$${\tilde{x}}(n),{\tilde{x}}(n+1),\ldots,{\tilde{x}}(n+P-1)\to{\tilde{x}}(n+P),\quad n=1,2,\ldots$$
1), . . . , x˜(n + P − 1) → *x˜(n* + P), n = 1, 2, . . .

## E8.3: Midi Composition (2) Rnn Architecture

input layer of P
100 neurons, corresponding to 25s

•
=
1. recurrent layer of 512 LSTM units and 30% dropout

•
2. recurrent layer of 512 LSTM units and 30% dropout

•
3. recurrent layer of 512 LSTM units and 30% dropout

•
1. dense layer of 256 neurons and 30% dropout

•
2. dense layer of 375 neurons

•
output layer with softmax for 375 possible tones/accords at the next time instant

•
categorical cross-entropy loss (classification of 375 classes)
•

## E8.3: Midi Composition (3) Hearing Experiments

a) an original MP3 file from Chopin (Prelude Op. 28 No. 15)

b) MP3 converted to MIDI with the original time-varying tone duration c) MP3 converted to MIDI with a fixed ton duration d) MIDI "composed" by the LSTM. 

The first P/4 = 25s contain the initialization ˜x(1), . . . , x˜(P) from c) –
–
2), . . . is generated by the LSTM-RNN.

e) MIDI containing randomly selected tones/accords The rest ˜x(P
1), x˜(P
+
+

## E8.4: Human Activity Recognition

The Deep Learning Lab offered by ISS consists of two projects:
- Diabetic retinopathy detection in color fundus image
- Human activity (walking, sitting, standing, laying, . . .) recognition using smartphone accelerometer (3 channels) and gyroscope (3 channels) signals.  Since the input is a sequential data with temporal correlation, LSTM or GRU are recommended architectures.

![257_image_0.png](257_image_0.png)

# Inductive Bias

It's a word used by advanced users.

Bias: inclination/tendency towards something
•
Inductive (math. induction): generalization from concrete observations to a general

•
statement, not related to inductive coils in electrical engineering Inductive **bias**: a bias that shapes the learning and inference in a desired way and
•
thus facilitates induction (learning)

ChatGPT: Inductive bias in deep learning refers to the set of assumptions that a machine learning model inherently incorporates, guiding it to prefer one solution over another. These biases influence how a model learns from data and generalizes to unseen examples. In essence, they shape the model's learning process by constraining the space of solutions. Different neural network architectures embody different inductive biases tailored to different types of problems to be solved.

Building streets is a bias enabling faster travel.

## E9.1: Examples Of Inductive Bias

Different approaches and architectures imply different inductive biases:

Model-based signal processing: The signal model is a strong inductive bias. It

•
enables the development of signal processing algorithms according to the signal model.

Dense layer: The dense connections enable the collection of information from all
•
input neurons for decision making, summary of input etc.

Convolutional layer: The kernels enable the learning of local features. Thus the feature learning from a whole image requires a cascade of convolutional layers.

•
Recurrent layer: The feedback enables the learning of temporal correlation of se-
•
quential data.

Transformer: Attention enables a feature transform based on range-independent

•
similarity.

# Human Attention When Looking At An Image

![260_image_0.png](260_image_0.png)

The query-key-value **principle**:
 you look for something (**query**) in an image, e.g

•
scan the whole image quickly and compare the query with all image parts (**keys**)

•
return the most similar image part (**value**)
•

# Correlation In Signal Processing

Attention is an extension of correlation in signal processing, see ch. 3.

Cross-correlation ˆ=
attention find a short known signal y(n) in a long measurement x(n), e.g. for synchronization

•
•
 y(n) (query) and each segment of x(n) of the same length (key) and returns that segment (value) of maximum similarity compute the similarity between cross-correlation autocorrelation

![261_image_1.png](261_image_1.png)

![261_image_0.png](261_image_0.png)

Autocorrelation ˆ=
self-attention find self-similarity between different segments of the same x(n). Queries and keys
•
are from the same signal x(n).

## Attention In General

Attention: Allow some elements to selectively attend to other elements.

weight different parts of input →
refined input with captured semantic information
•
large/small weights for more/less important parts of input →
inductive bias

•
Difference to normal weighting Wx and W
X:

⊗
normal: static weighting, i.e. W
fixed for all inputs after training
•
attention: *dynamic* weighting, i.e. input-specific weights W(x)
•

If query

![262_image_0.png](262_image_0.png)

# Channel And Spatial Attention

Convolutional block attention module (**CBAM**)1
•
channel **attention**: compute weights ∈ RC to scale (emphasize) individual chan-
•
nels (feature maps)

spatial **attention**: compute weights ∈ RH×W to scale (emphasize) individual pixels
•
and spatial regions each feature tensor can be refined in this way at a low additional cost

•
this early version of attention did not follow strictly the query-key-value principle

•

![263_image_0.png](263_image_0.png)

## Single-Query Attention

proposed in "Attention is all you need" (2017),2 the first paper about transformer

•
αi are importance weights of vi. They describe how strongly q attends to ki.

•
αi depend on the current input (q, ki) and are dynamically calculated.

•
s(q, k) calculates the similarity between q and k.

•

![264_image_0.png](264_image_0.png)

# Similarity Metrics

$${\mathrm{for~}}q,k\in\mathbb{R}^{d}$$
 


+

| Name               | Similarity metric s(q, k)   |
|--------------------|-----------------------------|
| dot product        | kT q                        |
| scaled dot product | kT q/ √d kT q               |
| cosine similarity  | kkk · kqk                   |
| . . .              | . . .                       |

scaled dot product: 1/ √d to limit the growth of |kT q| for large d to avoid numerical
•
es(q,k)
instability in cosine similarity is normalized: |s(q, k)| ≤ 1
•
for simple calculations, the scaled dot product is widely used
•

# Multi-Query Attention

Extends the single-query attention to multiple queries:

Rdv×N as above

•
Rd×N and keys in values in N
K
N
V
∈
∈
•
queries qj ∈ Rd, 1 ≤ j ≤ M. Let Q = [q1, . . . , qM] ∈ Rd×M.

•
M
calculate attention vectors for M
queries M
[attention(q1, K, V), . . . , attention(qM, K, V)] ∈ Rdv×M
attention(Q, K, V) =



$\mathbf{a}\cdot\mathbf{b}=\mathbf{a}\cdot\mathbf{b}$. 


+
Assuming the scaled dot product similarity,3 we get

•
· softmax(KTQ/ √d).

attention(Q, K, V) =
V
softmax() is applied to each column of its matrix argument.

•
also known as scaled dot-product **attention**

•
"correlate" all queries with all keys a simple computational block without any learnable parameters

•
no recurrence, all calculations can be parallelized on a GPU
•

# Multi-Head Attention (Mha) (1)

the core module in transformer

•
h heads doing h scaled dot-product attentions

•
each head 1 h does this in an individual projection of Q, K, V
determined by i ≤
≤
•

```
                                 
WQi ,WKi , WVi
     
•

```

concatenated results of all scaled dot-product attentions weighted by W0

•
all weight matrices WQi , WKi , WVi (1 ≤ i ≤ h) and W0 learned from data

![267_image_0.png](267_image_0.png)

# Multi-Head Attention (2)

Given:

embedding dimension N, see embedding below D
∈
•
RD×M, K
RD×N, V
RD×N

query, key, value matrices Q 
•
∈
∈
∈
number of attention heads h ∈ N. Let d = D/h ∈ N.

Steps:

•
for each attention head 1 h, project Q, K, V
into the column spaces of i ≤
≤
•
WQi ,WKi , WVi ∈ Rd×D and calculate h scaled dot-product attentions attention(WQi Q, WKi K, WVi V) ∈ Rd×M, 1 ≤ i ≤ h hi =
concatenate all attentions along the embedding dimension
•


$$\mathbf{h}={\left[\begin{array}{l}{\mathbf{h}_{1}}\\ {\vdots}\\ {\mathbf{h}_{h}}\end{array}\right]}\in\mathbb{R}^{D\times M},\quad D=h d$$
 



+
RD×M with RD×D

calculate the final attention W0h W0
∈
∈
•
like multiple kernels in CNN, multiple attention heads allow to jointly attend to
- 
information in different representation subspaces.

# Rnn/Lstm Vs. Transformer For Seq2Seq Models

At the beginning, RNN/LSTM was used in seq2seq models. Today, transformer4 is the standard choice.

RNN and LSTM: sequential in nature

+) simple 
−) process word for word recursively, focus on local dependencies, hard to capture long-range dependencies 
−) no parallel computation on GPU due to recurrence Transformer: parallel in nature

+) process all words in a sequence jointly, "correlate" all queries with all keys, capture both short- and long-range dependencies

+) no recurrence, easy parallel computation on GPU
N2 for a sequence length
−) higher quadratic complexity N
∼
−) require much more training data

## Tokenization

Tokenization: Divide a text into shorter units known as **tokens**.

Tokens can be characters (including punctuation marks and space), subwords, words, sentences etc. The choice of tokens determines the trade-off between the size of token **vocabulary**: how many different tokens in the vocabulary?

•
•
the token **length**: how many tokens in the text to be analyzed?

A shorter token leads to a smaller token vocabulary but a larger token length and vice verse. The choice of tokens is language-dependent and system-dependent.

## Chatgpt:

subwords as token,

•
e.g. "ChatGPT is great!" →
6 tokens ["Chat", "G", "PT", " is", " great", "!"]
ChatGPT 3.5 has a maximum token length of 4096 for a single interaction5. A
•

```
                                                                            
longer text will be truncated.
                          

```

This slide contains roughly 130 words, corresponding to several hundreds of tokens in

```
                                                                      
ChatGPT.
       
 5Based on answer of ChatGPT 3.5

```

## Token Embedding

We need a numerical representation of each token for calculation. Both ASCII coding 
(one byte per character) and one-hot coding (one one-hot vector for each token in the vocabulary) do not reflect the similarity of tokens.

Token **embedding**: Represent each token as a numerical vector or **embedding** x(n) ∈

RD with the embedding **dimension** D ∈ N such that the embeddings of tokens with similar meanings stay close in the embedding space RD (e.g., "apple" and "orange" vs. "Germany" and "France").

Token embedding is not manually designed. It is learned by a NN during the training process. **Word2Vec** is one popular word embedding method (token = word). It maps each token to its embedding and is trained on word prediction.6 3072.7 This high-dimensional ChatGPT 3.5 uses a token embedding dimension of D
=
embedding space is required later to capture semantic information about each token (its relationship to other tokens) during training.

## Transformer

First **transformer** in "Attention is all you need" (2017) for translation:

a seq2seq model relying fully on at-
•
tention encoder-decoder architecture

- •
encoder: learn a representation of the input sequence

•
decoder: learn a representation of the output sequence + relate the output to the input (translation)

3 multi-head attentions 
•
2 self-attentions with Q
K
for
=
–
input and output 1 attention with Q
K
between
,
–

```
                                
input and output
               

```

![272_image_0.png](272_image_0.png)

# Self-Attention

Self-attention: Learn the relationship between different tokens of an input sequence

[x(1), . . . , x(N)] ∈ RD×N
Use a multi-head attention (MHA) with Q
K
V
=
=
=
look at all N2 pairs of tokens x(m), x(n) and learn their relationship
•


 

 


 
 
ˆ(n) captures x(n) and other related tokens x(m)
•
x
{xˆ(n)} is a refined version of {x(n)}, capturing more semantic information
•
stacking Nx 6 self-attentions to get more hierar-
•
=
chical semantic information 1. MHA: look at pairs of tokens 
–
2. MHA: look at pairs of pairs of tokens 
–
3. MHA: . . .

–

$${\underline{{{\hat{x}}}}}(1),\ldots,{\underline{{{\hat{x}}}}}(N)\in\mathbb{R}^{D}$$

![273_image_0.png](273_image_0.png)

![273_image_1.png](273_image_1.png)

## More Details About The Transformer⋆

"Input/Output Embedding": tokenization and token embedding
•
"Positional **Encoding**": inject positional information into the embeddings 
•
MHA is permutation-invariant; a change of the token order does not change the

–
attention result. 

In NLP, the sequential order of the tokens is of course relevant. 

–
Thus add a position-dependent embedding vector ∈ RD to the input/output embeddings to capture the token positions (no details)

–
"Add & Norm": residual connection and layer normalization
•
"Feed Forward": dense layers

•
Causal autoregressive decoder: generate the current output token based on past

•
output tokens and input tokens 
"shifted right": current output token not fed into the decoder 
–
"masked multi-head attention": all future tokens are zeroed out, prevent the decoder from seeing into the future 
–
"Output Probabilities": softmax output for different tokens at the current posi-
–
tion

# Inductive Biases Of The Transformer

Different special designs for different purposes:

Positional encoding
•
provide positional information about the tokens 
–
allow the transformer to consider the sequential order of the input

–
Self-attention
•
assume that all tokens in a sequence are potentially related to each other 
–
weight different tokens in the sequence based on their similarities to queries 
–
learn relationship between tokens across short and long ranges

–
Attention stack and residual connection
•
attention stack to learn hierarchical representations 
–
residual connection to enable information flow across different stacks

–
Feedforward dense layers 
•
combine information from all tokens 
–
learn complex relationships within the sequence, capturing non-linear patterns
–

# Evolutions Of Chatgpt⋆

GPT-3: Generative pretrained transformer 
•
seq2seq language model, Np ≈ 175B 
–
self-supervised pretraining using pretext tasks, e.g.

–
masked language model: understand relationships between words

*
next sentence prediction: understand relationships between sentences

*
GPT-3.5: ChatGPT
•
supervised fine-tuning of GPT-3
–
reinforcement learning from human feedback (RLHF) to improve the training
–
use free of charge

–

## Gpt-4: •

multimodal input (text and image), text output 
–
training cost $100M
–
you have to pay
 –
Future ?

•

# Vision Transformer

Vision transformer (ViT, 2021) for image classification: 8
- patch tokenization + token embedding as in NLP
- "transformer encoder" much like the encoder in the language transformer
- no convolutional layers - "MLP Head": classification head

![277_image_1.png](277_image_1.png)

![277_image_0.png](277_image_0.png)

# About Transformer

+
general purpose

+
range-independent attention, excellent scalability to input size

+
parallel computation, excellent scalability to network size

+
state-of-the-art performance

+
widely used today in NLP, computer vision, . . . 

need huge datasets 
−
very high computation and memory complexity
−
Current research: more efficient transformers, e.g. by restricting attention to local neighbors Shifted Window Vision Transformer (SWinViT)9
•

# A List Of The Most Important Layers In Deep Learning

Layer: A building block of a neural network performing a specific action.10

| Layer type                    | Ch.                                    | Purpose & Examples               |
|-------------------------------|----------------------------------------|----------------------------------|
| Main feature transform layers |                                        |                                  |
| dense                         | 4                                      | summary, decision making         |
| convolutional                 | 7                                      | learn local features             |
| recurrent                     | 8                                      | learn temporal correlation       |
| attention                     | 9                                      | self-attention, cross-attention  |
| Supporting layers             |                                        |                                  |
| activation                    | 4                                      | apply nonlinearity               |
| normalization                 | 5, 7batch/instance/layer normalization |                                  |
| regularization                | 6                                      | dropout                          |
| downsampling                  | 7                                      | max pooling, strided convolution |
| upsampling                    | unpooling, deconvolution               |                                  |
| 7                             |                                        |                                  |
| reshaping                     | 7                                      | flatten, global average pooling  |
| residual                      | 7                                      | avoid vanishing gradient         |

# The Major Bottlenecks Of Deep Learning

Addressed in this chapter:

Data-inefficiency in contrast to human brains 
•
a large model needs a large amount of training data 
–
even more serious: needs corresponding labels for supervised learning
- 
This is time-consuming, expensive, or hardly possible in some cases (e.g. privacy).

It is highly desirable to reduce the required amount of labeled training *data.*

Poor generalization: A trained DL model underperforms or fails if 
•
the task changes 
–
the task doesn't change but the test data distribution changes We need to adapt the trained DL model. Again data-efficiency is the key issue: We

–
won't use a large labeled dataset to adapt the model.

Addressed in the last chapter:

lack of explainability
•
sensitive to adversarial attacks

•
high complexity
•

![281_image_0.png](281_image_0.png)

## Different Learning Supervisions (1) Supervised Learning •

labeled dataset {x(n), y(n)} 
–
e.g. classification, regression, segmentation, object detection
–
excellent performance due to precise learning supervision by labels 
–
but expensive data collection and labeling
–

## Unsupervised Learning •

huge amount of unlabeled data {x(n)} on internet 
–
impossible: classification, regression, segmentation, object detection
–
course DPR: clustering, feature dimension reduction
–
recently: self-supervised representation learning
–
Supervised learning is task-oriented, self-supervised learning is representation-oriented.

# Different Learning Supervisions (2)

Weakly supervised learning
•
a subarea of supervised learning
–
use coarse-grained/inaccurate but easy-to-collect labels 
–
e.g., bounding box or labeled point instead of object shape in semantic image segmentation
–
Self-supervised representation learning
(SSL) 

![282_image_0.png](282_image_0.png)

•
a subarea of unsupervised learning
–
supervised learning with unlabeled dataset {x(n)}. Possible? 

–
mostly for pretraining 
–
Y. LeCun: "Self-supervised learning: The dark matter of intelligence"

–
Semi-supervised learning
•
use a small labeled dataset to reduce the labeling effort and simultaneously a large unlabeled dataset to reduce performance degradation
–

# When Visiting A Zoo

Supervised learning: learn each animal by reading the animal identification sign

![283_image_0.png](283_image_0.png)

•
Self-supervised learning: don't read any animal signs, go around and learn to com-
•
pare (contrast) animals in size, shape, color, texture etc. without knowing their names

![283_image_1.png](283_image_1.png)

# Pretraining And Finetuning

A simple, effective, and widely used method to enhance the data-efficiency.

Downstream task
(DT)

the task you want to solve, e.g. classification, regression, segmentation, object

•
detection described by a labeled dataset

•
wish: this labeled dataset as small as possible

•
Basic idea of pretraining and finetuning:

Don't train a model for the downstream task from scratch with random parameter

•
initializations and a huge labeled dataset.

Use a pretrained model, followed by a finetuning step.

•
pretraining: Trained on other datasets. After that, model parameters pretty close to the desired solution (better initialization).

•
finetuning: Finetune the pretrained model using a small labeled dataset of the downstream task. This is a conventional supervised learning step.

•

# Backbone And Head

![285_image_0.png](285_image_0.png)

Backbone, feature extractor, **encoder**: first layers

 x z = fB(x) learn features/representations z from the raw input x

![285_image_1.png](285_image_1.png)

 

 

 

 
 
Head, **decoder**: last layers

![285_image_2.png](285_image_2.png)

$$\begin{array}{l l}{{\hat{\underline{{{y}}}}=f_{H}(\underline{{{z}}})}}&{{\mathrm{~classification~head}}}\\ {{}}&{{}}\\ {{\hat{\underline{{{y}}}}=f_{H}(\underline{{{z}}})}}&{{\mathrm{~segmentation~head}}}\end{array}$$

$$\mathrm{\mathrm{\normalsize~learn~features/represent}}$$

Observations cross different datasets/tasks: small change in fB, big change in fH.

Hence pretraining of backbone.

# Different Types Of Pretraining

a) Supervised pretraining use a supervised loss lsup(x, y) based a labeled dataset from p(x, y), often done by others 
–
in general p(x, y) not related to the downstream task, backbone not optimum
–
many backbones from VGGNet, ResNet, transformers etc. pretrained on Ima-
–
geNet (or other datasets) are ready for use in TensorFlow/PyTorch b) Self-supervised **pretraining** 
use a self-supervised loss lunsup(x) based on an unlabeled dataset from p(x) 
–
p(x) may be from the same downstream task, this is good. 

–
methods: autoencoder, pretext task, contrastive learning, . . .

–
c) Semi-supervised **pretraining**: a) +
b) 
combine a self-supervised cost function Lunsup based on a *large* unlabeled dataset
–
Lsup based on a *small* labeled dataset:
L
and a supervised cost function Lunsup + λLsup 
=
better than b) because more tailored to the downstream task
–

## E10.1: Representations (1)

Any measured signal (speech, image, . . .) may have different representations. Its native representation, the raw data of measurement, is not always optimum for solving a certain task. Finding a suitable representation or **embedding** can simplify the task.

a) Time- vs. frequency-domain representation in signal processing b) The arabic notation is much easier for calculation than the Roman one.

![287_image_0.png](287_image_0.png) 

## E10.1: Representations (2) D) Mnist

There are more information contained in the MNIST images than just the digit labels:

•
appearance

•
orientation identity of writer

•

•
 . . .

$\begin{array}{c}\mathbf{-}\mathbf{\hat{\ell}}\\ \mathbf{\hat{\ell}}\end{array}$


 - $\epsilon$ ... 



+
The native representation of MNIST, the pixels, is not optimum for digit recognition because a small distortion to one image can change the digit:

$$1\leftrightarrow7,\qquad4\leftrightarrow9,\qquad0\leftrightarrow6,\qquad\ldots$$
$\frac{1}{2}$
$\epsilon$

 $\blacksquare$
6, . . .
This is the reason why MNISTnet3 needs multiple convolutional layers to first extract a more discriminative representation of the MNIST images before classification.

## E10.2: Latent Variable

The latent variable z is a hidden and compressed representation/code for the input x because

```
                                       z with only small differences. z may contain
                                                                                
information about:
                 

```

can be well reconstructed from x a) MNIST:

![289_image_0.png](289_image_0.png) 
b) Face:

![289_image_1.png](289_image_1.png) 

## Comments On Autoencoder

A perfect self-copy ˆx = x is never the purpose of an autoencoder. Actually, it is designed to be unable of that by using an
•
undercomplete **autoencoder** with c ≪
d.

An autoencoder is restricted to reconstruct the input x approximately.

In an autoencoder, we are not interested in the output ˆx. Instead, the latent variable

•
z as a compressed representation of the input x is of interest.

An autoencoder is trained in the conventional supervised way with the ground truth
•
y = x, i.e. with unlabeled data.

An autoencoder always uses an encoder-decoder structure, but not each encoder-
•
decoder is an autoencoder, e.g. the semantic segmentation network from E7.8. An autoencoder requires the self-reconstruction label y = x.

The principal component analysis (PCA) from signal processing is a special case

•
WT x, a linear decoder ˆx = Wz and the MSE loss E(kx − xˆk2), see course DPR. In other words, autoencoder is a nonlinear extension of PCA and is more powerful than PCA.

of autoencoder with a linear encoder z =

## E10.3: Mnistnet4 - Denoising Autoencoder (1) Task

denoise a noisy digit image

•

## Dataset

MNIST image + Gaussian pixel noise N(0, 0.52) as input for the autoencoder
•
corresponding clean MNIST image as ground truth
•

## Training

l2-loss

•
minibatch size 32
•
20 epochs

•
Adam optimizer

•
learning rate γ = 0.001
•

## E10.3: Mnistnet4 - Denoising Autoencoder (2)

Autoencoder: Using a latent space of R32 for a 28x28 image

![292_image_0.png](292_image_0.png)

Convolutional and deconvolutional layers with stride S = 2 instead of MaxPooling and Unpooling
Padding for all convolutional/deconvolutional layers to control the image size
•

## E10.3: Mnistnet4 - Denoising Autoencoder (3) Results

From top to bottom in each block: original MNIST, noisy/corrupted MNIST, de-
•
noised MNIST
a) From training set of noisy MNIST
•
b) From test set of noisy MNIST
•
c) middle rows and columns in each MNIST image set to 0 (dark)

•
d) middle rows and columns in each MNIST image set to 255 (bright)

•

![293_image_0.png](293_image_0.png)

The denoising autoencoder performs well for learned noise in b), but bad for unseen corruption in d).

# Downstream Vs. Pretext Task

## Downstream Task (Dt)

the task you want to solve, e.g. classification, regression, segmentation, object

•
detection described by a labeled dataset for supervised learning
•
wish: this labeled dataset as small as possible

•
(PT) or surrogate **task**

Pretext task not the task you want to solve, an artificially generated supervised task
•
for any given unlabeled data {x(n)}, we generate one or multiple PTs with the cor-
•
responding surrogate labels y(n) for each x(n)
learn the representation of x(n) by solving PTs in a supervised way

•
•
the representation is used later for solving downstream tasks definition of PT depends on the type of input

•
Autoencoder: self-reconstruction y = x is the simplest pretext task

## Self-Supervised Representation Learning On Pretext Task (1)

Supervised learning Self-supervised

![295_image_1.png](295_image_1.png)

a) self-supervised representation learning on PT 
b) supervised learning on DT with pretrained fB
representation learning

![295_image_0.png](295_image_0.png)

Backbone or feature extractor fB: often task-agnostic and can be reused

•
•
Task heads fPT and fDT for PT and DT: task-specific and must be retrained

## Self-Supervised Representation Learning On Pretext Task (2)

 given unlabeled input

PT pretext task

|    | modified input for PT                                                                                                                                            |
|----|------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| y  | surrogate label for PT                                                                                                                                           |
| z  | learned representation of input backbone or feature extractor for learning the representationz PT head for solving PT based onz DT head for solving DT based onz |

 estimate for y

Explanations:

x Pretraining:

fB
is pretrained on PT. After this initialization, it is finetuned on DT.

•
fPT
not used in DT. fDT
is trained from scratch using a small labeled DT dataset.

- 
˜
x fPT
y ˆ
fDT

# Examples Of Pretext Task (1)

Spatial domain:1

| Pretext task           |                |
|------------------------|----------------|
| image rotation         | discriminative |
| patch rotation         | discriminative |
| Jigsaw patch puzzle    | discriminative |
| image colorization     | generative     |
| image masking          | generative     |
| image super-resolution | generative     |

![297_image_0.png](297_image_0.png) 
# Examples Of Pretext Task (2)

Temporal domain:

Language (e.g. ChatGPT) 
•
masked language model: predict 15% masked words in a sentence

"I attend the course Deep Learning at University of Stuttgart." 
–
next sentence prediction: Given two sentences A and B.

–
Are they semantically equal?

*
A?

Is B
an answer to
*
Is B
the next sentence to A?

*
*
. . .

•
Speech and audio masked latent speech representation2
–
Video
•
frame order estimation: estimate the true order of video frames 
–
video prediction: predict the next video frame based on the previous ones

–

## E10.4: Simclr (1) - Concept

A simple (but seminal) framework for contrastive learning of visual representations.3 x: input instance (sample)

- 
j: two different "views" of x after 
˜ 
i, x

˜
•
x applying two different data augmentations t, t′ to x

![299_image_0.png](299_image_0.png)

f(·): feature extractor

•
hi, hj: representations of ˜xi, x˜j

•
 g(·): projector (a shallow NN)
•
zi,zj: projections of hi, hj for similarity calculation
•
Linear **classification**: A linear classification head (one dense layer with softmax)

trained on top of the pretrained backbone f(·) with only 1% of ImageNet labels (fewshot **learning**) achieves a better top-5 accuracy than AlexNet trained with all labels.

Contrastive learning is the main technique in self-supervised pretraining for images.

## E10.4: Simclr (2) - Data Augmentations

![300_Image_0.Png](300_Image_0.Png)

Rich data augmentations play a crucial role in contrastive learning. They lead to diverse views of the same input image. This forces the learning of a representation space which focuses on the image content, but not on its appearance.

## Pretext Task Vs. Contrastive Learning

Two popular self-supervised pretraining techniques.

| Pretext task                          | Contrastive learning    |                    |
|---------------------------------------|-------------------------|--------------------|
| encoder/backbone                      | yes                     | yes                |
| calculate output ˆy                   | yes                     | no                 |
| decoder/head                          | yes                     | no                 |
| output space (ˆy, y)                  | representation space zi |                    |
| learned representationloss defined in |                         |                    |
| depends on the pretext task           | yes                     | no                 |
| learned to                            | solve a task            | compare (contrast) |

Unsupervised learning was known before deep learning: clustering, principal com-
•
ponent analysis (PCA), . . ., see course DPR.

•
Self-supervised learning more advanced and powerful.

Contrastive learning can be viewed as a special pretext task: Predict 2. view of an
•
input from its 1. view.

# Foundation Model

Foundation **model** (FM) (2022)4 is a data-centric, general-purpose, and pretrained DL model.

pretrained on vast amounts of task-agnostic data using self-supervision
•
can be applied across a wide range of downstream tasks

•
via finetuning and/or prompting
•
is always a huge model

•
Examples:

language: BERT (Google), ChatGPT (OpenAI)

•
image: Florence (Microsoft), DINOv2 (Meta)

•
semantic image segmentation: SAM (Meta)

•
text-image: CLIP (OpenAI)

•
•
 . . .

Foundation model starts a new AI era. But it is still far away from the final goal of general artificial intelligence, see last chapter.

## E10.5: Clip (1) - Pretraining

(2021) from OpenAI5: Contrastive language image pretraining CLIP
a language-image model to learn a joint (text, image) embedding space

- •
pretrained on 400M (text, image) pairs collected from Internet using contrastive learning connecting images with text

→
Pretraining:

one text encoder

•
one image encoder

•
minibatch of N
32768
•
=

```
                             
(text, image) pairs
                    

```

positive: paired (text, im-
•
age)

negative: others

•
contrastive loss based on
- 
dot product similarity After pretraining, each row/column of the similarity matrix represents a classifier.

![303_image_0.png](303_image_0.png)

## E10.5: Clip (2) - Zero-Shot Transfer

Two ways to use CLIP for image classification:

Pretraining: Use CLIP as a backbone for image representation.

•
Finetuning: Use a few labeled images to train a small classification head as in SimCLR. This is often called few-shot **transfer**.

Zero-shot **transfer**: No finetuning, no classification head, no labeled images.

•
Use text as image labels. 

create different texts for

–
images put them into the text

–
encoder put a new image into the image encoder 
–
where is the largest sim-
–
ilarity?

![304_image_0.png](304_image_0.png)

# Comments On Foundation Model

Architecture:

must be expressive to learn everything
•
must scale well with dataset size

- •
transformer is the standard choice today Pretraining: self-supervised contrastive learning, next-tokens prediction, . . .

•
Potential risks:

Privacy: Foundation models use Internet-scale data, running the risk of violating
•
user privacy (e.g., US writers sue ChatGPT developers due to unlicensed use of their books.)

Monopoly "Big company, big model, big business": Only a few big companies

•
are able to train foundation models. They will control whole business areas in the future, raising new risks for economics and society.

# Notations In Transfer Learning

input space, e.g. vector RM,
color image {0, 1, . . . , 255}H×W×3, . . .

Y
output space, e.g. C-class classification (0, 1)C,

| semantic segmentation (0, 1)H×W×C, {bi ∈ R4, yˆi ∈ (0, 1)C}, . . . object detection   |
|---------------------------------------------------------------------------------------|

p(x, y) over X × Y unknown joint distribution

$$\begin{array}{l}{{\mathrm{dSSIIication}\left(\mathbf{0},\mathbf{1}\right)}}\\ {{H{\times}W{\times}C}}\\ {{\mathrm{,}}}\\ {{\equiv(\mathbf{0},\mathbf{1})^{C}\},\ldots}\end{array}$$

available dataset sampled from

$${\mathfrak{o n}},p(x,y)$$

p(x, y)

X

$$\begin{array}{l}{{{\mathcal{D}}=\{\underline{{{x}}}(n),\underline{{{y}}}(n)\}_{n=1}^{n}}}\\ {{{\mathcal{T}}=\{{\mathcal{X}},{\mathcal{Y}},p(\underline{{{x}}},y)\}}}\end{array}$$

{X, Y, p(x, y)} task to learn a function f : X → Y based on p(x, y)

$$\begin{array}{l l}{{{\mathcal{T}}_{s}=\{{\mathcal{X}}_{s},{\mathcal{Y}}_{s},p_{s}({\underline{{{x}}}},{\underline{{{y}}}})\}}}&{{\mathrm{source~task}}}\\ {{{\mathcal{T}}_{t}=\{{\mathcal{X}}_{t},{\mathcal{Y}}_{t},p_{t}({\underline{{{x}}}},{\underline{{{y}}}})\}}}&{{\mathrm{target~task}}}\end{array}$$

## Task Or Distribution Change

A DL model is learned for a source task Ts
{Xs, Ys, ps(x, y)}. It will, however,

=
underperform or fail if the task or distribution changes. This happens frequently.

Three levels of change:

A) New **task**: Xs , Xt, e.g.

speech recognition, image understanding, play Go, compose music

•
B) Related **task**: Xs = Xt, but Ys , Yt, e.g.

speech recognition, speaker identification, dialect classification
•
image classification, semantic image segmentation, object detection

•
C) Distribution shift or domain **shift**: Xs = Xt, Ys = Yt, but ps(x, y) , pt(x, y), e.g.

semantic segmentation of street images collected under different conditions cross daytime: from day to night

•
cross weather: from sunshine to rain, snow, fog
•
cross country: from Germany to USA, China, . . .

•
•
cross sensor: from an old sensor to a new one (new resolution, field of view)

This induces changes on p(x), p(y) or p(x, y).

# How To Deal With Task Or Distribution Change?

A) New task: No solution yet.

Today no DL model is able of this kind of general artificial **intelligence**. This is the final goal of AI to be achieved, see last chapter.

## B) Related Task And C) Domain Shift: Transfer Learning

It transfers a DL model learned on a source task Ts to a new target task Tt.

•
Typically, a new labeled target dataset Dt is required for the transfer learning.

•
For data efficiency, Dt should be as small as possible. Several variants: 
•
Few-shot **learning**: Only a few labeled samples (shots) in Dt. 

–
One-shot **learning**: Only one labeled sample in Dt. 

–
Zero-shot **learning**: No labeled samples for the target task.

–
Transfer learning vs. pretraining:

Actually, pretraining (on a source dataset Ds) and finetuning (on a target dataset Dt)

can be viewed as a kind of transfer learning.

# Domain Adaptation Vs. Continual Learning

## Domain Adaptation

a special case of transfer learning to mitigate C) domain shift:

•
{X, Y, ps(x, y)} → Tt = {X, Y, pt(x, y)}
Ts
=
no change of architecture

•
•
frozen backbone (or finetuning)

head/decoder finetuning on Dt

•
the DL model is required to perform well *only* in the target domain, not in the

•
source domain

## Continual Learning

a much more challenging case of transfer learning
•
adapt a DL model to a sequence of new tasks: Ts → Tt,1 → Tt,2 →
•
. . .

•
the model must perform well in all tasks, i.e. no catastrophical forgetting no training samples of old tasks available

•

![310_image_1.png](310_image_1.png)

# Discriminative Vs. Generative Models

In supervised learning, x is the observation and y the desired output.

## Discriminative Models:

learn the (posterior) conditional distribution p(y|x)

![310_image_0.png](310_image_0.png)

•
useful for discrimination tasks maxy p(y|x) like classification and regression
•
focus on decision boundaries

•
Generative **models**:

learn the joint distribution p(x, y)

•
generate new data according to p(x, y)

•
can also be used for discrimination due to p(y|x) =
•
p(x, y)/p(x), but rarely done

•
more challenging than discriminative models VAE, GAN, diffusion, . . .

•
In unsupervised learning with only, generative models learn the distribution p(x).

x

## E11.1: Discriminative Vs. Generative Models

a) Painting x and painting style y ∈ {oil, ink}

![311_image_1.png](311_image_1.png)

Discriminative model p(y|x): Can distinguish between both painting styles y for a

•
given painting without being able to paint in these styles.

![311_image_0.png](311_image_0.png)

x Generative model p(x, y): Can paint in both styles and, naturally, can also distin-
•
guish between them.

b) Speech x and language y ∈ {English, German, . . .}
I am generative in English, German and Chinese, but I am only discriminative between, say, Japanese and Arabic.

c) ChatGPT is a generative model.

# Which Examples From Ch. 1.3 Are Generative?

| Discriminative (D) or generative (G)?   |                           |                                         |       |
|-----------------------------------------|---------------------------|-----------------------------------------|-------|
| E1.4                                    |                           |                                         |       |
| Example                                 | Task                      | Description                             |       |
|                                         | classification            | estimate the class of an image          | D     |
| E1.5                                    | semantic segmentation     | estimate the class of each pixel/sample | D     |
| E1.6                                    | object detection          | localize & classify objects             | D     |
| E1.7                                    | panoptic segmentation     | semantic and instance segmentation      | D     |
| E1.8                                    | regression                | estimate real-valued parameter          | D     |
| E1.9                                    | classification            | estimate geo location                   | D     |
| E1.10                                   | text-to-image translation | G                                       |       |
| E1.11                                   | translation               | image-to-image translation              | G     |
| E1.12                                   | translation               |                                         |       |
|                                         | generative model          | music composition                       | G     |
| E1.13                                   | various                   | speech and language processing          | D & G |
| E1.14                                   | reinforcement learning    | playing Go                              | D     |
| E1.15                                   | reinforcement learning    | chip design                             |       |
| E1.16                                   | supervised learning       | proof                                   |       |
| E1.17                                   | generative model          | synthetic scene generation              | G     |

# Autoencoder Vs. Variational Autoencoder

![313_image_0.png](313_image_0.png) 

An autoencoder (AE) is not generative (and neither discriminative):

Though we find a suitable latent space z for x, we don't know the distribution of z.

Without the encoded and stored "code" z for x, there is no way to calculate ˆx. An AE

just memories the input x in terms of its code z; it cannot generate new ˆx because we don't know how to create z without x.

A
variational **autoencoder** (VAE) is generative:
It is similar to AE except for the additional requirement that z has a known distribution, say N(0,I)1. After training, the encoder is no longer required. We can draw random samples of z from N(0,I) and use the decoder to translate z to new input samples ˆx without x.

## P(X) (1)⋆ Variational Upper Bound For Negative Log-Likelihood Ln −

and have the joint PDF
p(x,z) with the corresponding marginal PDFs p(x), p(z)
X
Z
and conditional PDFs p(z|x), p(x|z). Often the integral in − ln p(x) = − ln R p(x|*z)p(z)dz* is difficult to calculate due to a missing analytic expression for p(x|z). Using a PDF

approximation q(z|x) for p(z|x), it has been proven: 

$$-\ln p(\underline{{{x}}})\leq\mathrm{E}_{\underline{{{Z}}}\sim q(\underline{{{z}}}|\underline{{{x}}})}\left[-\ln p(\underline{{{x}}}|\underline{{{Z}}})\right]+D_{\mathrm{KL}}\left(q(\underline{{{z}}}|\underline{{{x}}})||p(\underline{{{z}}})\right).$$

The right-hand side is called the variational upper **bound**2 for − ln p(x) and is easier to calculate (see next slide).

q(z|x)||p(z|x) − DKL q(z|x)||p(z) = EZ∼q(z|x) ln q(Z|x) p(Z|x)! − EZ∼q(z|x) ln q(Z|x) p(Z) ! Proof: DKL EZ∼q(z|x) ln p(Z) p(Z|x)! = EZ∼q(z|x) ln p(x) p(x|Z)! = ln p(x) − EZ∼q(z|x) ln p(x|Z). = The Bayes' rule p(z|x) = p(x|z)p(z)/p(x) is used in the last line. Since DKL is always non-negative, 
$$-\ln p(\underline{{{x}}})\leq-\ln p(\underline{{{x}}})+D_{\mathrm{KL}}\left(q(\underline{{{z}}}|\underline{{{x}}})||p(\underline{{{z}}}|\underline{{{x}}})\right)=-\mathrm{E}_{\underline{{{Z}}}\sim q(\underline{{{z}}}|\underline{{{x}}})}\ln p(\underline{{{x}}}|\underline{{{Z}}})+D_{\mathrm{KL}}\left(q(\underline{{{z}}}|\underline{{{x}}})||p(\underline{{{z}}})\right).$$
The equality holds if q(z|x) and p(z|x) are identical.

## P(X) (2)⋆ Variational Upper Bound For Negative Log-Likelihood Ln −

Comments:

The variational upper bound looks like an autoencoder: q(z|x) encodes x into z
•
while p(x|z) decodes x from z.

Since the posterior PDF
q(z|x) is much narrower than the prior PDF
p(z), the ex-
•
pectation EZ∼q(z|x)() can be approximated by the sample mean of by drawing a few samples zi from q(z|x). If q(z|x) ≈ δ(z − z0),

$$\operatorname{E}_{\underline{{{Z}}}\sim q(\underline{{{x}}}|\underline{{{x}}})}\left[-\ln p(\underline{{{x}}}|\underline{{{Z}}})\right]=\int-\ln p(\underline{{{x}}}|\underline{{{z}}})q(\underline{{{z}}}|\underline{{{x}}})\mathrm{d}\underline{{{z}}}\approx-\ln p(\underline{{{x}}}|\underline{{{z}}}_{0}).$$

Assuming Gaussian/Laplace distribution of x|z0 fD(z0), we end up with the with mean l2 or l1 loss of fD(z0) − x.

Z
has a desired known PDF
p(z), e.g., N(0,I). q(z|x) is unknown, but close to Gaus-
•
q(z|x)||p(z). This KL divergence between two Gaussian distributions can be calculated analytically.

sian N(µ(x), C(x)) due to min. DKL
The approximation of the difficult integral in ln p(x) by an easy-to-calculate sample

−
mean and KL divergence is the key benefit of the variational upper bound.

## Application Of The Variational Upper Bound To Vae⋆

with some renamed notations p →
q, q(z|x) →
qE(z|x):

$$I(\underline{{{x}}};\underline{{{\theta}}})=-\ln q(\underline{{{x}}};\underline{{{\theta}}})\leq\underbrace{\mathrm{E}_{\underline{{Z}}\sim q_{E}(\underline{{{x}}};\underline{{{\theta}}}_{E})}[-\ln q(\underline{{{x}}}|\underline{{{Z}}};\underline{{{\theta}}}_{D})]}_{I_{\mathrm{exc}}(\underline{{{x}}};\underline{{{\theta}}}_{E})}+\underbrace{D_{\mathrm{KL}}\left(q_{E}(\underline{{{z}}}|\underline{{{x}}};\underline{{{\theta}}}_{E})||q(\underline{{{z}}})\right)}_{I_{\mathrm{KL}}(\underline{{{z}}};\underline{{{\theta}}}_{E})}$$

 

 
 
 
+
 q(x; θ): The hard-to-calculate parametric model for the unknown true distribution

•
p(x). Instead of minimizing ln q(x; θ), the variational upper bound is minimized.

−
 qE(z|x; θE): The posterior distribution of the latent variable z for a given x. It describes the encoder of VAE with the parameter vector θE.

•
 q(x|z; θD): How can the original input x be reconstructed from the latent variable
•
z? It describes the decoder of VAE with the parameter vector θD.

 q(z): The desired prior distribution of the latent variable z, e.g. Gaussian.

•
minθ lrec: Train the pair of encoder and decoder such to keep maximum information
•
of x in z and to best reconstruct x using z (like in an AE).

minθ to force z to the desired distribution q(z). It is a kind of regularization.

lKL: Train the encoder θE
•

# Vae With Stochastic Encoding

![317_image_0.png](317_image_0.png) 

C1/2 is the square root of the covariance matrix CT/2 · C1/2.

C
•
=
is assumed to be diagonal diag(σ21, . . . , σ2d) with the variances σ2i to simplify the training. In this case, C1/2 =
Mostly, C
•
diag(σ) is diagonal as well with σ
= 
[σ1, . . . , σd]T containing the standard deviations.

•
The forward pass of the above VAE is as usual. Both loss terms lKL and lrec can be calculated and averaged over all samples x(n).

The backpropagation of gradients to learn and can, however, not pass the θE
θD
•
sampling unit. The sampling of z ∼
qE(z|x) =
N(µ(x), C(x)) is a non-continuous operation and has no gradient.

# Vae With Stochastic Encoding And Reparametrization

![318_image_0.png](318_image_0.png) 

Instead of drawing a sample z from N(µ, C), we draw a sample ǫ from N(0,I) and
•
calculate ǫ. Here a diagonal C
diag(σ σ) is assumed.

⊙
⊙
+
z =
µ σ
=
The sampling N(0,I) is now outside the path of backpropagation. z is continu-
•
ǫ ∼
ous in and which are continuous in θE. Backpropagation is thus possible.

µ σ is always non-negative while lnσ can be positive and negative as µ. It is easier
•
σ for the encoder to output lnσ than σ. This makes the exp() function necessary.

## E11.2: Generation Of Mnist Digits

a VAE using a 2D latent space z = [z1,z2]T ∼ N(0,I)
•
trained on the MNIST training set

•
Left (encoder): distribution of the MNIST test set in the latent space

- •
Right (decoder): generated 28x28 MNIST digits in the latent space (z1,z2)

![319_image_0.png](319_image_0.png) 

# (Gan)3 Basic Idea Of Generative Adversarial Network

Illustrated on the example of fake money: A game between two opponents.

forger: Try to make fake money as realistic as possible.

![320_image_0.png](320_image_0.png)

Generator G
•
=
police: Try to distinguish between real and fake money as Discriminator D
•
=
good as possible.

has a spy in the police (backpropagation) and learns why can discriminate.

G
D
•
Based on that, G
improves his faking technique to further fool D.

tries to find new differences between real and fake money to avoid to be fooled.

D
•
A continuous competition between G
and called D
game **theory**.

•
In the end, fake moneys are indistinguishable from real ones.

•

# Structure Of A Gan

![321_image_0.png](321_image_0.png)

![321_image_1.png](321_image_1.png)

latent variable

•
z: noise sample drawn from a known PDF
pnoise(z), e.g. N(0,I) ˆ=
•
G: generator (a DNN) with the parameter vector θG
=ˆ
decoder in VAE
G(z; θG): generated fake sample

- 
ˆ
G(z) =
•
x 
=
x: real sample with the unknown PDF
pdata(x)

D: binary discriminator/classifier (a DNN) with the parameter vector θD. Its output
•
D(x) =
D(x; θD) ∈ [0, 1] is the probability of x being real, i.e. class label 1 for real and 0 for fake. Hence, D
uses a sigmoid activation function in the output layer.

# Training Of Gan (1)

One training step consists of one minibatch for updating L(θG, θD) 
D: max θD
•

 
 
$-$ select $B$ samples $\underline{x}(n)$ from the training set  $-$ draw $B$ samples $\underline{z}(n)$ from $p_{\rm noise}(\underline{z})$  $-$ calculate the gradient $\frac{\partial}{\partial\underline{\theta}_{D}}\frac{1}{B}\sum_{n=1}^{B}\left[\ln D(\underline{x}(n);\underline{\theta}_{D})+\ln(1-D(G(\underline{z}(n);\underline{\theta}_{G});\underline{\theta}_{D}))\right]$  $-$ update $\underline{\theta}_{D}$ by stochastic gradient ascent for a fixed $\underline{\theta}_{G}$: $\underline{\theta}_{D}^{+1}=\underline{\theta}_{D}^{\prime}+\cdots$  and one minibatch for updating $G$: $\min I(\theta_{D},\theta_{D})$


and one minibatch for updating

$$\mathrm{:\,\,\min_{r\in\Omega}L(\underline{{{\theta}}}_{G},\underline{{{\theta}}}_{D})}$$

G: min θG
L(θG, θD) 
•

$-$ draw $B$ samples $\underline{z}(n)$ from $p_{\rm noise}(\underline{z})$  $-$ calculate the gradient $\frac{\partial}{\partial\underline{\theta}_{G}}\frac{1}{B}\sum_{n=1}^{B}\left[\ln(1-D(G(\underline{z}(n);\underline{\theta}_{G});\underline{\theta}_{D}))\right]$  $-$ update $\underline{\theta}_{G}$ by stochastic gradient descent for a fixed $\underline{\theta}_{D}$: $\underline{\theta}_{G}^{+1}=\underline{\theta}_{G}^{\prime}-\ldots$
# Training Of Gan (2)

The training of a GAN consists of a sequence of alternating updates of θD
and θG.

•
Instead of one minibatch for each of D
and G, an alternation between kD
1
≥
•
minibatches for θD
and kG
1 minibatches for θG
is possible.

≥
The minimax optimization is a typical problem in the game theory, a branch of

•
mathematics.4 The solution of minimax is a Nash **equilibrium**, a saddle point.

•

![323_image_0.png](323_image_0.png) 

## E11.3: Image Generation By Gan (1)

a) Generation of new and realistic photos: Which photo is real and which one is generated by a GAN?5

![324_image_0.png](324_image_0.png)

![324_image_1.png](324_image_1.png)

## Comments:

Generation is not copying. There are no real persons like shown. The photos are

•
generated by passing random samples of z to the generator.

The GAN has learned all characteristics of human faces.

•

## E11.3: Image Generation By Gan (2)

b) Generation of MNIST digit images from a latent space R10.

learned from the MNIST training set in an unsupervised way
•
Generator: R10 →
R6272 →
R7×7×128 →
R14×14×64 →
R28×28×32 →
R28×28×1

•
Discriminator: R28×28×1 →
R14×14×64 →
R7×7×128 →
R6272 →
R
•

![325_image_0.png](325_image_0.png)

They look more realistic (i.e. like hand-written) than those in E11.2 with a VAE.

# Conditional Gan (Cgan)

cGAN6:
 G
and D
conditioned on some information, e.g. class label y ∼ plabel

![326_image_0.png](326_image_0.png)

$$\mathrm{\mathbf{fak}}\mathbf{e}$$
 

 

 


 


+
# $\mathrm{r}\mathrm{e}\mathrm{a}\mathrm{l}$? 
$$\operatorname*{min}_{G}\operatorname*{max}_{D}L=\operatorname{E}_{\underline{{{x}}},\underline{{{y}}},\underline{{{z}}}}[\ln D(\underline{{{x}}},\underline{{{y}}})+\ln(1-D(G(\underline{{{z}}},\underline{{{y}}}),\underline{{{y}}}))]$$

the same architecture and training procedure as GAN
•
additional class label input y ∼ plabel 
•
G: generate fake samples ˆx = G(z, y) given class labels y
•
D: distinguish fake and real samples given class labels y 
•
accept (x, y) if x is real AND matches y 
–
reject (x, y) if x is fake OR does not match y
–

# Paired Image Translation

using paired data (x, y), e.g. based on pix2pix7

![327_image_0.png](327_image_0.png)

![327_image_1.png](327_image_1.png)

$$\mathrm{{\bf{fakie}}}$$  or  ... 
 

 

 



$$v)\ \mathrm{to}\ x\ \mathrm{for~pair~}(x,y)$$

 


 


+
real?

 y: source domain image from the unknown distribution psource(y) instead of noise z
•
G(y): generated target domain image

- 
ˆ
•
x 
=
x: true target domain image from the unknown distribution pdata(x)

LcGAN: conventional cGAN loss for adversarial training
- •

L1: additional L1 loss to ensure similarity of ˆx = G(y) to x for paired (x, y)
  **A 17** **183** **to ensure similarity of $\Sigma=\sigma(y)$ to $\Sigma$ for pairs $\min\max L=L_{\rm cGAN}+\lambda L_{1}$**  $L_{\rm cGAN}={\rm E}_{\underline{x},\underline{y}}[\ln D(\underline{x},\underline{y})+\ln(1-D(G(\underline{y}),\underline{y}))]$  $L_{1}={\rm E}_{\underline{x},\underline{y}}[\underline{x}-G(\underline{y})]_{1}$

11.2 Generative adversarial network 11-19

# E11.4: Medgan For Medical Image Translation (1)⋆

a) Motion correction in MR image8: Correct motion blurr

![328_image_0.png](328_image_0.png)

b) Inpainting in MR image9: Autocompletion of damaged image regions

![328_image_1.png](328_image_1.png)

# E11.4: Medgan For Medical Image Translation (2)⋆

![329_Image_0.Png](329_Image_0.Png)

Generator: cascaded U-Net, progressive refinement of the translated output

•
Discriminator: combination of multiple losses for enhanced image quality
•
cGAN loss LcGAN
for adversarial training (fake or real) 
–
perceptional loss Lpercep to capture pixelwise alignment 
–
content loss Lcontent to capture difference in global structures 
–
style loss Lstyle to capture difference in textures and patterns Objective engineering as important as architecture engineering!

–

## E11.5: Image Translation For Radar⋆

Spectrogram of radar signals to monitor the gait (micro-Doppler) of a walking person.

a) Denoising in radar spectrogram10: Reduce noise

![330_image_0.png](330_image_0.png)

b) Super-resolution in radar spectrogram11: Enhance radar resolution artificially

input output target

![330_image_1.png](330_image_1.png)

## E11.6: Cmgan¹² For Speech Enhancement*

- speech enhancement in the spectrogram domain after short-time Fourier transform
- generator based on encoder-decoder architecture and transformer
- metric discriminator

![331_image_0.png](331_image_0.png)

# Cyclegan13 For Unpaired Image-To-Image Translation⋆

![332_image_0.png](332_image_0.png)

![332_image_1.png](332_image_1.png) 

2 generators G1, G2 and 2 discriminators D1, D2

•
•
Goals of training: 
cycle consistency: ˆyˆ ≈
y and ˆxˆ ≈ x, i.e. G2 ≈ G−1 1 
–
adversarial: ˆx as realistic as x, ˆy as realistic as y 
–
but no L1 losses kx − xˆk1, ky − yˆk1 in contrast to supervised translation
–
13J. Y. Zhu et al, Unpaired image-to-image translation using cycle-consistent adversarial networks, 2017, arXiv:1703.10593

## E11.7: Unpaired Semantic Image Synthesis (Usis)⋆

semantic image **synthesis**: inverse process of semantic image segmentation, i.e.

•
semantic map image (not unique)

→
CycleGAN14 unpaired: using unpaired training data based on an improved
•
applications: content creation, semantic manipulation, data augmentation
•

![333_image_0.png](333_image_0.png)

# Unconditional Diffusion Model

Diffusion probabilistic models or diffusion **models**15 are becoming popular as generative models. Several popular diffusion models for image generation are Dall·E 2, Midjourney, and Stable Diffusion, see E1.10.

Forward process (diffusion **process**): Change a given input x0 to a Gaussian noise 1, . . . , T, see course AM for Markov process.

N(0,I) by a sequence of first-order Markov processes p(xt|xt−1), t =
xT
∼

![334_image_0.png](334_image_0.png)

Reverse process (generation **process**): Pass random samples of xT
N(0,I) back-
∼
wards through the Markov process to generate realistic x0 by learning a neural network q(xt−1|xt; θt) for each reverse step.

## Training Of Diffusion Model⋆

The training of a diffusion model is much like that of a VAE using the variational upper bound. The forward process calculating the latent variables x1:T
from is similar to x0 the encoder while the reverse process is similar to the decoder of VAE, respectively.

p(x0) and the approximating distribution The goal is to minimize the KL divergence between the desired distribution q(x0; θ):

$$\operatorname*{min}_{\underline{{\theta}}}D_{\mathrm{KL}}(p(\underline{{x}}_{0})\|q(\underline{{x}}_{0};\underline{{\theta}}))=\mathrm{const}-\operatorname{E}_{\underline{{x}}_{0}\sim p(\underline{{x}}_{0})}\ln q(\underline{{x}}_{0};\underline{{\theta}})\approx\mathrm{const}-\frac{1}{N}\sum_{n=1}^{N}\ln q(\underline{{x}}_{0}(n);\underline{{\theta}})$$  with (no proof)



$$-\ln q(\underline{{{x}}}_{0};\underline{{{\theta}}})\leq\sum_{t=1}^{T}E_{\underline{{{x}}}_{t-1},\underline{{{x}}}_{t}\sim p(\underline{{{x}}}_{t-1},\underline{{{x}}}_{t})}\left[-\ln q(\underline{{{x}}}_{t-1}|\underline{{{x}}}_{t};\underline{{{\theta}}}_{t})\right].$$

Comparison to VAE:

comparable to a hierarchical VAE with multiple levels of latent variables

•
forward/reverse process consists of a sequence of Markov Gaussian transitions

•
forward pass simple and parameterless, reverse process parameterized by T
DNNs
- 
(CNNs, transformers, . . .)

## Conditional Diffusion Model⋆

The unconditional diffusion model can only generate x0 unconditionally from the learned distribution q(x0; θ) ≈
p(x0). A diffusion model trained on ImageNet would generate random ImageNet-like images.

In order to generate specific images x0, e.g., on text prompt like y ="Pope in a white jacket", one considers q(x0|y) ∼
q(y|x0)q(x0) or equivalently 

$$-\ln q(\underline{{{x}}}_{0}|\underline{{{\nu}}})=-\ln q(\underline{{{\nu}}}|\underline{{{x}}}_{0})-\ln q(\underline{{{x}}}_{0})+\mathrm{const.}$$

The 2. term q(x0) is handled like in the unconditional diffusion model, while the 1. term ln
−
q(y|x0) describes the typical negative log-likelihood (NLL) classification loss for the classification "input x0 ln
−
belongs to class y".

# Vae Vs. Gan Vs. Diffusion Model

VAE: 
low quality of generated samples

−

## Gan:

high quality of generated samples 
+
difficult adversarial training (minmax); sometimes mode **collapse**, i.e., generates a limited diversity of samples or even the same sample regardless of the input

−
Diffusion model:

high quality of generated samples

+
avoid difficult adversarial training of GAN
+
large number of steps T, high complexity, long training time, long inference time

−

## Important But Unaddressed Topics

Despite of a comprehensive lecture, not all topics of deep learning could be addressed.

Architecture
•
graph neural network
–
Learning paradigm
•
reinforcement learning
–
Bottlenecks of deep learning
•
data-inefficiency, see ch. 10
–
poor generalization, see ch. 10
–
lack of explainability
–
visualization
*
causal inference 
*
sensitive to adversarial attacks 
–
high complexity
–
model reduction
*
neural architecture search
*

## Graph Neural Network (1)

Up to now simple data types relying on fixed, regular 1D/2D/3D grids are assumed.

well defined ordering
•
well defined spatial locality
•
fixed topology and fixed size

•

 
$$\mathrm{series}$$

 

![339_image_0.png](339_image_0.png)

![339_image_1.png](339_image_1.png)

 


 
1D grid: time series 2D grid: images graph

There are more complex data types described by a **graph**, consisting of nodes and edges, e.g., social networks (who knows whom?), communication networks (network of base stations), chemistry, molecular biology. The edges describe the relationship between the nodes.

no fixed node ordering: {A, B, C, D, . . .} is the same as {C, A, D, B, . . .}
- •
no spatial locality arbitrary topology and arbitrary size
•

## Graph Neural Network (2)

Conventional DNNs like CNN are not applicable to graphs.

Graph neural network
(GNN): A class of deep learning methods tailored to perform inference on data described by graphs.

Typical first step: Node embedding each node i in a graph is characterized by a node feature vector, e.g.,
•
xi = {age, gender, education, country, favorite music/books/sports/foods,, . . .}
for a social graph where each node represents a person.

zi = f(xi) projecting the feature vector xi onto zi in a numerical embedding/representation space.

learn an embedding function
•
the node embedding is learned such that similarity in the embedding space approx-
•
imates the similarity between the node feature vectors.

## Reinforcement Learning

Supervised learning learn from examples, e.g., AlphaGo learns from old Go games of humans.

Reinforcement learning
(RL) is a subarea of ML. It learns from rewards for sequential decision making. For example, AlphaGo Zero learns from games against computer (AlphaGo). To be more precise, based on the current state (game situation)

and reward of an environment (Go game), an agent (AlphaGo Zero) optimizes his next action (which move?) to maximize the long-term reward (win of the game). RL can be interpreted as a data-driven control when there is no system model for the traditional model-based control.

Deep reinforcement learning
(DRL) combines RL with deep neural networks.

![341_image_0.png](341_image_0.png)

## Visualization Visualization Techniques

improve the understanding of DNN models

•
enhances their acceptance by users

•
Two popular visualization techniques for two different purposes:

Gradient-weighted class activation mapping (Grad-CAM) or guided **Grad-CAM**:1 
•
visualize important regions (Grad-CAM) or pixels (guided Grad-CAM) of the input image for a certain target class 
–
applicable to all DNNs without retraining
–
t-distributed stochastic neighbor embedding (**t-SNE**):2 
•
visualize the similarity of high-dimensional data in a low-dimensional space 
–
more powerful than principal component analysis (PCA) due to a nonlinear

–
mapping from high- to low-dimensional space

# Grad-Cam And Guided Grad-Cam (1)

![343_image_0.png](343_image_0.png)

# Grad-Cam And Guided Grad-Cam (2)

Grad-CAM:

Train a DNN for e.g. image classification
•
Select the output Xl ∈ RHl×Wl×Cl of layer l consisting of Cl feature maps Xl,i ∈

RHl×Wl. Typically, choose a deep layer (large l) because a deep layer contains more

•
class information than low-level image features like edges.

Select the activation aL,c (before softmax) of the output neuron corresponding to a

•
wanted class c (e.g. cat).

Calculate gradients ∂aL,c/∂[Xl]hwi and their average αci = 1 HlWl Ph Pw ∂aL,c/∂[Xl]hwi as weighting factors.

•
Calculate Vc
•
=

$$\mathrm{ReLU}\left|\sum\alpha_{i}^{c}\mathbf{X}_{l,i}\right|$$

Xi αciXl,i as the visualization map for the class c.

•
Normally, a deep layer has a much lower spatial resolution than the input image.

Hence upsampling of Vc to the input resolution and overlay with the input image.

guided **backpropagation**3 Guided Grad-CAM: Combination of Grad-CAM with

# T-Sne

An unsupervised and nonlinear dimension reduction technique by representing each high-dimensional sample xi (e.g. a 28x28 digit) by a 2- or 3-dimensional point yi:
xi and xj, e.g. pi j ∼ e−αkxi−x jk2 calculate pairwise similarity probabilities between
•
define pairwise similarity probabilities qi j between yi and yj in a similar way
•
•
choose
{yi} to minimize the KLD between pi j and qi j

![345_image_0.png](345_image_0.png)

t-SNE of MNIST digits

## Causal Inference (1)

Statistical **inference** in ML/DL: Make a prediction ˆy given x based on p(y|x) which is unknown and estimated from data by using a (DNN) model. We say, x and y are
(linearly or nonlinearly) correlated. Correlation is bidirectional: x ↔
y.

Causal inference/**reasoning**: Draw a conclusion about the cause y for an observation x. Causal connection is unidirectional y →
x.

## E12.1: Correlation Vs. Causality

![346_image_0.png](346_image_0.png)

## Causal Inference (2)⋆

Mathematical tools for prediction and causal inference:

Prediction: Observational p(y|x). What is the distribution of y given observation x?

•
Causal inference: Interventional p(y|do(x)) where do is the **do-operator**. What is the distribution of y if I intervened in the natural data generating process by artificially forcing
•
to take value x.

X
Which one do I want? Depending on the application. p(y|x) for observation and prediction, p(y|do(x)) for proactive change and treatment.

## E12.2: Causal Inference "What If"

a) "Would I not get lung cancer had I not smoked?"

b) "Would my blood pressure be lower had I consumed less salt?"

c) x = medical treatment, y = outcome d) software debugging e) how to prevent global warming?

We are at the beginning of research for causal inference with DL.

## Adversarial Attack (1)

DNNs are powerful, but suffer from adversarial **attacks**: A small, imperceivable, but carefully designed perturbation to the input can fool a trained model.6

![348_image_0.png](348_image_0.png)

not all perturbations will fool the model, only carefully calculated ones

•
imperceivable because the attacks are not visible for humans (more dangerous)

•
attacks also to speech, audio, text, . . .

•
How to calculate adversarial examples? It's an optimization problem.

Assume that a model f(·) is trained to make the correct decision f(x) = y. We look for a small perturbation such that f(x + ∆x) = y′ , y. This is formulated as a minimization problem and solved by gradient descent, using
∆x gradients over the input

(by fixing the model), not over the model parameters as during training.

x

# Adversarial Attack (2)

Different types of adversarial attacks:

targeted **attack**: f(x + ∆x) = y′ for a chosen y′ , y
- •
untargeted **attack**: f(x + ∆x) = y′ for any y′ , y
•
white-box **attack**: model f(·) well known and gradient search is applied

•
black-box **attack**: more challenging due to unknown model f(·). No gradient information to guide the search of ∆x. Instead, multiple queries of f(x) are needed.

Intuitions for adversarial attack:

DNNs have a huge-dimensional feature space

- •
most of the training data concentrated in a small region called manifold input perturbation may move the model to a region the model has never seen before,

•
leading to unpredictable results How to defend against adversarial attacks?

some ideas, but no effective countermeasures yet

•
open research problem
•

## Model Reduction

Today there is no theory to predict the right DNN size and capability for a given task.

In order to solve the task with a satisfactory performance, typically DNNs with a larger capacity than required are used.

For an efficient implementation of trained DNNs on low-cost platforms in mobile (e.g.

mobile phone) and embedded applications (e.g. car), it is highly desirable to reduce the computational and memory complexity as well as power consumption of a trained DNN without remarkable performance loss. This process is called model **reduction**.

Three important approaches for model reduction are:

RM×N of a layer Low-rank **factorization**: Factorize a large weight matrix W
∈
•
RM×K, B
RK×N, K
into a product of two low-rank matrices W
AB
with A
∈
∈
<
≈
min(M, N). The number of multiplications changes from MN
to (M
N)K.

+
Pruning: Force approximately zero columns or rows of W
to remove the corre-
•
sponding input or output neurons.

Quantization: Reduce word length of DNN for inference.

•

## Neural Architecture Search

Conventionally, the architecture of a DNN is hand-designed. Training means the optimization of the model parameters.

The focus of neural architecture **search** (NAS) is an automated optimization of the DNN architecture. Three basic issues are:

Search space: It defines the types of basic building blocks and connections to com-
•
pose the DNN. The use of prior knowledge about typical properties of basic building blocks and connections will help to reduce the search space.

Search strategy: It deals with exploration of the huge discrete search space. A
- 
challenge is the compromise between exploration (quick global search) and exploitation (detailed local search).

Performance estimation strategy: It refers to the evaluation of a selected architec-
•
ture. The simplest choice, a standard training and validation of the architecture, is too time-consuming. Methods to reduce the performance estimation cost are highly desirable.

# Is Ai Already Smarter Than Humans?

AI already beats humans in some tasks:

Deep Blue for playing chess, IBM 1996
•
Watson for question-answering (Jeopardy), IBM 2011
•
ResNet for image recognition, Microsoft 2015
•
AlphaGo for playing Go, DeepMind 2015-2017
•
Chip design, Google Research 2021
•
Support in mathematical proof, DeepMind 2021
•
ChatGPT for text-based conversation, OpenAI 2023
•
No, it is still a long way toward human-like intelligence.

DL/AI

![352_image_2.png](352_image_2.png)

![352_image_3.png](352_image_3.png)

![352_image_0.png](352_image_0.png)

![352_image_1.png](352_image_1.png)

| Artificial neural network   | Human brain       |                        |    |              |
|-----------------------------|-------------------|------------------------|----|--------------|
| size                        | large             | small                  |    |              |
| power consumption           | kW–MW             | 20–25 W                |    |              |
| synchronization             | synchronized      | non-synchronized       |    |              |
| signal carrier              | continuous-valued | pulse train            |    |              |
| sampling rate               | GHz               | 1-100Hz firing rate    |    |              |
| precision                   | high              | low (few bits)         |    |              |
| number of neurons           | ≈                 | 100 millions           | ≈  | 100 billions |
| architecture                | layered           | more general           |    |              |
| memory and computation      | separated         | integrated             |    |              |
| network                     | fixed             | evolutionary           |    |              |
| learning                    | gradient-based    | unknown                |    |              |
| signal transport/processing | electrical        | biochemical electrical |    |              |
| intelligence                | single domain     | multi domains          |    |              |

## Artificial Neural Network Vs. Human Brain

Huge gap between artificial neural network and human brain!

# Limitations Of Ai

![354_Image_0.Png](354_Image_0.Png)

Not energy efficient

•
Not sample efficient

•
It is a weak/**narrow** AI: Achieves or outperforms human's
•
capability in some restricted tasks Strong AI or Artificial general intelligence **(AGI**): Achieves or outperforms human's capability in all tasks.

We are far from *that.*

Hard to predict the AI future:

Initial hope: AI should replace low-skilled works (worker, cleaning, service, . . .)
and the humans do the fun/creative jobs.

•
Today's reality: AI does the fun/creative jobs (writer, poet, painter, composer,

•
game-player, movie-making, . . .)
What's next?

# Serious Questions

We are afraid:

AI-enabled **fake**: fake news, fake image, fake speech, fake video, . . .

- •
AI **monopoly**: a few ones control the access to AI

Singularity in AI: machines achieve human-like intelligence (strong AI).

•
We have to answer the questions:

•
Shall we accept it?

•
Shall we shut down all machines before this happens?

•
Can we still shut down the machines?

•
Shall we stop DL/AI research today?

•
Shall I cancel my DL course?

My personal opinions:

There is no way to stop the invention of new *technologies* (e.g. knife, fire, dynamite,
•
car, nuclear energy, Internet, gene technology, AI, . . .).

•
But we must regularize the use of new *technologies.*

March 13, 2024: New "AI Act" of EU

## My Conclusions

Today's AI is still far away from human-like intelligence.

•
But it becomes a very powerful tool in almost all areas.

•
The regularization of AI becomes an important issue.

•
The AI research will remain relevant for a long time.

•
Long-term prediction: Purely electrical AI is not competitive.

•
The future will be *bioelectrical* AI.

But AI is not the only tool. Use AI only when you have no other *choices*. For

•
problems with a good signal model, model-based signal processing is simpler and

```
                                                                                              
better.
      

```
