# Table of Contents

- [E_commerce_hierarchical : Technical Part](#e_commerce_hierarchical--technical-part)
  - [Approach to the Problem](#approach-to-the-problem)
    - [High Level Overview of the Steps Towards Solution](#high-level-overview-of-the-steps-towards-solution)
    - [Understanding Why this Approach Works well for this problem](#understanding-why-this-approach-works-well-for-this-problem)
    - [Possible Enhancements](#possible-enhancements)
  - [More in-Depth Look into The Model Mechanics](#more-in-depth-look-into-the-model-mechanics)
    - [Model Architecture :](#model-architecture-)
      - [What is Dropout and What is DropConnect](#what-is-dropout-and-what-is-dropconnect)
      - [Meaning of Dropout in this Project](#meaning-of-dropout-in-this-project)
      - [Explaining DropConnect](#explaining-dropconnect)
        - [What regularization are used for other connections in the AWD-LSTM:](#what-regularization-are-used-for-other-connections-in-the-awd-lstm)
    - [Explaining the non-monotonically trigged ASGD](#explaining-the-non-monotonically-trigged-asgd)
  - [References](#references)
- [Results](#results)
  - [Getting Started : Build and Run the Docker Container [ver 1 : Conda Container]](#getting-started--build-and-run-the-docker-container-ver-1--conda-container)
    - [Program Content : Demonstration how to use the Trained Model [VERSION 1, 3 FEATURES]:](#program-content--demonstration-how-to-use-the-trained-model-version-1-3-features)
- [DATA ](#data-)
  - [ML Splits Used in Model 1:](#ml-splits-used-in-model-1)
  - [Raw data:](#raw-data)
  - [Full ML Splits:](#full-ml-splits)
    - [Using Virtual Environment Notebook (Fast.AI + Jupyter Notebook) : Conda or PIP ](#using-virtual-environment-notebook-fastai--jupyter-notebook--conda-or-pip-)
    - [Comments](#comments)


# E_commerce_hierarchical : Technical Part
[Client-Service] This is the high-dimensional deep-learning and NLP based intelligent product category remapper --- MODEL [VERSION 1]. The model is fit on 80% of the data and validated on the remaining 20% of the data. The validation error refers to accuracy and weighted F1 score on the 20% of the full data (validation data).

## Approach to the Problem
Nowadays the training of modern [deep learning models](https://en.wikipedia.org/wiki/Deep_learning) can be said to be of semi-supervized nature, where the 
- first phase can be said to be unsupervized pre-training where a model having possibly a complex architecture and a lot of parameters is trained on some publicly available datasets, for example Wikipedia, followed by
- supervized training phase (*supervised finetuning*) in which the pre-trained model from phase 1 is *fine-tuned* to a dataset chosen by the user in a process called **transfer learning**.

The chosen approach in this project was to use Transfer learning in a form of language modeling to improve understanding the language of the underyling dataset better and then build a classification model on top of that language model to fine-tune the language model to perform a concrete task on the dataset.

### High Level Overview of the Steps Towards Solution

Firstly, based on evidence found from research, a baseline [ULMFit -- universal language model](https://arxiv.org/abs/1801.06146) model was chosen for implementation. 
The  modeling with **ULMFit** is done in 2 steps: 
- language modeling  to predict the next word in the input text. That should have the impact of teaching the network how to start to understand the text structure better.
- high dimensional classification modeling on top of the built language model. High dimensional refers to a scenario where the amount of classes is on a similar scale to the amount of samples in the dataset.

### Understanding Why this Approach Works well for this problem

- **String features** 
These indicate the usage of an NLP model. Obvious options are *BERT*, *Transformers* and *ULMFit*.

- According to [1] the performance of BERT and ULMFit is very similar. Thus, we focus on ULMFit.
- Versatile structure of *ULMFit*  overcomes the classical overfitting problems of RNNs. That reduces the generalization error to a minimum, ensuring that the model generalizes to unseen data points
- Pre-trained on Finnish corpus increases performance on a 'Finnish-mostly' dataset
- No additional translation layers forcing all the input to be in one language reduces overhead complexity and enhances prediction speed

### Possible Enhancements
- Extend or modify the Fast.AI library's `batch prediction` functionality to enable the proper use of batch prediction functionality for improved inference speed
- Get multi-lingual data and build a multi-lingual model that is relatively agnostic to input text language.
- Include more text features by for example using [Latent Dirichlet Allocation](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation) and topic modeling

## More in-Depth Look into The Model Mechanics

### Model Architecture :
The ULMFit model uses **AWD-LSTM (average stochastic gradient descent weight-dropped LSTM)**
that is defined through the use of *DropConnect* + *non-monotonically triggered averaged stochastic gradient descent*.

#### What is Dropout and What is DropConnect
To understand more about the model, it's recommended to first understand what is dropout and dropconnect.
A good primer to get started is [this video](https://youtu.be/E1DyJI7L6tI).
For an advanced understanding of how the Dropout operation is actually equivalent to Bayesian variational inference, please refer to this [video](https://www.coursera.org/lecture/bayesian-methods-in-machine-learning/dropout-as-bayesian-procedure-XZKFJ).
The idea of dropout in general is to introduce noise to the network in order to make learning more robust and reduce overfitting.
There are different types of dropouts : 
- Bernoulli dropout (where the noise is binary, the neuron is either on or off) and
- Gaussian dropout (where the noise is multiplicative, with mean of 1 and standard deviation of $\alpha$, from Gaussian distribution).
Dropout from Bayesian viewpoint makes the network more robust to learn the true posterior distribution approximation which in our case corresponds to the set of parameters defining the weights in between the AWD-LSTM layers.

According to [3], the use of dropout (and its variants) in NNs can be interpreted as a Bayesian approximation of a well known probabilistic model: the Gaussian process (GP) (Rasmussen & Williams, 2006).
This can be easily seen from the [this video](https://youtu.be/E1DyJI7L6tI) as well as from 
![this example](https://github.com/Integrify-Finland/e_commerce_predictor/blob/master/img/dropout.png), where we see that **factoring suitable multiplicative Gaussian noise does not change the stochastic gradient descent optimization objective**.
Note: to recall Monte Carlo methods, a [good refresher](https://www.scratchapixel.com/lessons/mathematics-physics-for-computer-graphics/monte-carlo-methods-in-practice/monte-carlo-integration).


#### Meaning of Dropout in this Project
Applying dropout over the input text vector to the language model is equivalent to "placing a distribution over the weight matrix following the input and approximately integrating over it" ([3]).


What is word embedding : This is the one-hot word-vector multiplied by the embedding matrix  VxD (transformed into a new vector space), where V is the number of words in the vocabulary and D is the dimension of the embedding. Since the embedding matrix is optimized, it is desirable to apply dropout to the one-hot-encoded word vectors (~dropping words at random in input sentence).
Since the inputs are short text chunks concatenated from different features, in the first layers, embedding matrices over these texts are computed. 
The embedding dropout then means that a random set of rows in the embedding matrix are set to 0 and since one row corresponds to one word in the vocabulary, these words won’t have an impact on training since the mask is repeated at each time step thus the same words throughout the sequence are dropped.
A distribution is placed over these word embeddings. 

Dropout in this word based model corresponds to randomly dropping word TYPES in the sentence so that the model is forced not to rely on single words for the task.
Explaining Embedding Dropout
Embedding Dropout is also used on at a word level. Dropout in general is an approximate Bayesian inference procedure, so is the embedding dropout.
Remember that frequentist model struggle learning in small data scenario; as well as here in the LSTM the overfitting problem.

#### Explaining DropConnect
The idea of this is to use Dropconnect on RNN, that is basically a dropout procedure extended from activations to weights. The dropouts are initialized once before the forward and backward passes, thus the impact of the training is minimal.
The DropConnect sets a random subset of recurrent connections of the hidden to hidden weight matrices to 0 to prevent the most common problem of RNNs : Overfitting in the recurrent connections.
The type if Dropout used is **Bernoulli dropout**.

##### What regularization are used for other connections in the AWD-LSTM:
While DropConnect is used for the hidden weight matrices, variational dropout (where the dropout mask is sampled only upon the first call in the pass per batch, and then used repeatedly for all connections within a pass on that batch)  is used for all other dropout operations, especially for inputs and outputs of LSTM in a given pass.
The stochasticity arises from the fact that a new mask is generated for each batch, thus hopefully the LSTM will learn something different from each batch of language data.

### Explaining the non-monotonically triggered ASGD
Averaged stochastic gradient descent is triggered only if the validation metric fails to improve for multiple cycles. This algorithm uses a constant learning rate.


## References
[1] [BERT VS UlmFit Faceoff](https://towardsdatascience.com/battle-of-the-heavyweights-bert-vs-ulmfit-faceoff-91a582a7c42b)

[2]  [What makes AWD-LSTM great ?](https://yashuseth.blog/2018/09/12/awd-lstm-explanation-understanding-language-model/)

[3] [Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning](https://arxiv.org/pdf/1506.02142.pdf)



# Results
![Results](https://github.com/Integrify-Finland/e_commerce_predictor/blob/master/cover.PNG)
The 6-textual feature model has achieved performance of **97.6% validation accuracy / weighted F1-score**.

## Getting Started : Build and Run the Docker Container [ver 1 : Conda Container]

[Container link](https://github.com/Integrify-Finland/e_commerce_predictor/tree/master/docker_working )
- `docker build -t ver1 .`
- `docker run -i -t ver1`

### Program Content : Demonstration how to use the Trained Model [VERSION 1, 3 FEATURES]:

1. Navigate to `code/package` folder
2. Run the `main.py` script defined in [this folder](https://github.com/Integrify-Finland/e_commerce_predictor/tree/master/code/package)

-- RUNTIME ARGS:
- running_mode : production/.... . If set to production, then all the 3 features used. If this variable is set to something else, for example "test", then the data processing pipe will not be executed and only provider and brand are chosen from features, thus this execution is quick and good for debugging.
- model_feature_names : Each model used will have different features, these features should be listed in this variable, separated from each other by comma, no space!

3. Read the execution logs in the logs folder to see how the program ran.


### Using Virtual Environment Notebook (Fast.AI + Jupyter Notebook) : Conda or PIP 
`conda create -n <chosen_env_name> --file requirements.txt`

`pip install -r requirements.txt`

`pip install -e git://github.com/Integrify-Finland/e_commerce_predictor/blob/master/virtualenv_cpu_jupyter/requirements.txt`


### Comments

The model is not meant for products (by `mapped_id`) that have only 1 occurrance in the dataset -- they also had to be excluded from the model because for a reason, scikit's train-test split is not meant to handle 1 item per class edge cases. There are synthetic methods to go beyond it (upsampling), but it's more the question  that if there data is so rare, then it shouldn't possibly affect the learning so much as it would do in the case of upsampling

Otherwise as you could see, in the function `calculate_metrics_on_data`  the performance metrics are calculated. One could see that especially the classes with the highest support will have very good performance -- thus, it seems that the model performance can be estimated class-wise is as directly related to the amount of data per that class, which also makes sense.
