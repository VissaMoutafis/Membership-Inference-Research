# Bilbiographical Report 

## Bilbiographic Table
| Paper | Application | Methodology | NN | Linear Classifier |  MLaS | BlackBox | Metric Based |
|-|-|-|-|-|-|-|-|
|MIA against machine learning models, Shokri et al | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |-|:heavy_check_mark:|:heavy_check_mark:|-|
|MIA against aversarially robust DLMs, Song et al|-|:heavy_check_mark:|:heavy_check_mark:|-|-|:heavy_check_mark:|-|
|Demystifying MIA in MLaS, Truex et al|:heavy_check_mark:|:heavy_check_mark:|-|-|:heavy_check_mark:|:heavy_check_mark:|-|
|ML-Leaks Model and Data Independent MIA and Def on Machine Learning Models, Salem et al|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark: |:heavy_check_mark: |:heavy_check_mark:|:heavy_check_mark:|
|Label Only MIA, Choo et al|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|-|:heavy_check_mark:|:heavy_check_mark:|-|
|Deep Models under the GAN, Hitaj et al| :heavy_check_mark: | :heavy_check_mark: |-|-| :heavy_check_mark: |-|-|
|MIA on ML: A survey, HU et al |:heavy_check_mark:||:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|

## Paper Overview

### Membership Inference Attacks on Machine Learning Models - Shokri et al

#### Datasets
<div id="Datasets">

- CIFAR-10: images 
- CIFAR-100: images 
- Purchases: categorical 
- Locations: categorical 
- Texas Hospital Stay: categorical 
- MNIST: images 
- UCI Adult: numerical & categorical 

</div>

#### Target Model Type
<div id="Target Model Types">

- NN 
- MLaS
</div>


#### Summary
<div id="Summary">

- First MIA appearance, exploiting Model overfitting.
- Attacker might has access to dataset ditribution, but not to target dataset
- Attacker will not know anything of the model architecture 
- Proposing ML-Model Queries for attacker dataset composition
- Using confidence intervals to predict membership
- Using shadow models (target model mimes), to train attack model
- Supervised training of binary classifier with confidence vectors queried from shadows, using membership knowledge.
</div>



### Membership Inference Attack against Adversarially Robust Deep Learning Models - Song et al

#### Datasets
<div id="Datasets">

- CIFAR-10: images 
- SVHN: images - street address views 
</div>

#### Target Model Type
<div id="Target Model Types">
- Deep NN 
- MLaS
</div>


#### Summary
<div id="Summary">

- Research on MIA performance in Adversarially robust ML models
- Model Overfitting -> vulnerable to MIA
- Adv. Robust Models ensure that model predictions are unchanged in the space around a training example -> High risk of MIA vulnerabilities
- Experimental Research shows that adv-robust models are **more vulnerable** to MIAs than completely undefended models !!!!
- Use of threshold classifier (logistic regression and other linear classifiers) as attack model.
</div>


### Demystifying Membership Inference Attacks in MLaS - Truex et al

#### Datasets
<div id="Datasets">

- Adult
- Cifar-10
- MNIST
- Purchases
</div>

#### Target Model Type
<div id="Target Model Types">

- Linear Models
- NN
</div>


#### Summary
<div id="Summary">

- Definition of Black/White/Grey attacker knowledge for future reference
- Experiments on different ways of developing a shadow dataset if the API probing is limited with respect to the calls permitted
    - Statistics Based Sampling
    - Active Learning, where the attacker acquires samples via a automated labeling and generation routine which is based on some data probed by the ML API
    - Region Based Generation, which is ultimately an augmentation of the pre-acquired dataset
- Detailed analysis of each phase in a Membership Inference Attack with confidence vector scores.
- How can we choose a shadow model architecture if we are uncertain of the original architecture: **Use an ensemble of different architectures and query each one of them in each datapoint**
- Comparison of attacks with different combs of target, shadow/generation model, attack model types.
- Exploring the attacker knowledge effect on the attack. As attacker knowledge we define the quality and quantity of the shadow dataset and the approximation of the target model by the shadows.
- Finally we consider the attack on a federated learning context:
    - Insider attacks: the attacker is one of the federation parties and has full access to all probability vectors produced by all parties -> easier to infer membership on a party dataset
    - Outside attacks: the attacker is just a user and can only acquire an aggregated form of the proballity vector -> lower attack performance 
- Talks about mitigations:
    - Model hardening: Reduce overfitting with drop out, regularization, dimension reduction, adversarial regularization
    - API hardening: provide a top-k confidence vector where not all probabilities are provided.
    - Differential Privacy: Not quite descriptive. Just setting some questions

</div>

### ML Leaks: Model and Data Independent Membership Inference Attacks and Defenses on ML Models - Salem et al

#### Datasets
<div id="Datasets">

The same as Shokri et al
</div>

#### Target Model Type
<div id="Target Model Types">

Same as Shokri et al
</div>


#### Summary
<div id="Summary">

- Supports that MIAs can be executed with a gradually total black box concept. Proposes 3 concepts
- Model 1: breaks the Shokri et al's assumption that we need many shadow models to get good performance - under certain occasions - and also constructs a method to loose the assumption of knowing the targets ML algorithm. The so called method combining is a aggregated ensemble that MUST contain the target model to succeed, but the attacker does not know which this model is, it's quite trivial to discover it though IMHO. 
- Model 2: Building in the previous context, Salem et al, propose that we do not need same-distr shadow data to execute a successfull attack and create an image dataset different than the targeted one, train their attack on it, execute it and get results quite similar to Shokri et al. Reasoning on why this so called data transfering attack succeeds is that MIAs focus on target META behaviour, on the prediction space and do not care about the dataset distribution (**NOTE THAT THIS IS WHY OUR PURCHASE-100 PERTURBATION ATTACK SUCCEEDS, as noted by chatziko**). 
- Model 3: Final model, focuses on proving that we don't need shadow models at all. The proposition is that we can set a specific threshold and determine membership based on higher posterior. This threshold attack is total black box and is consisted of predicting based on t-percentile membership. Results on this attack are totally emperical and the choice of t is experimental.
- Proposed defences:
    - Dropout
    - Model Stacking - Ensembles

and generally ways of reducing overfitting

</div>



### Label Only Membership Inference Attacks - Choo et all

#### Datasets
<div id="Datasets">

Same or Similar to Shokri et al
</div>

#### Target Model Type

<div id="Target Model Types">

Same as Shokri
</div>


#### Summary

<div id="Summary">

- What happens when we don't have the whole or even a part of confidence vector and we only have the predicted label
- Argument: Training data are robust to small perturbations, while test data aren't
- Propositions: Create "N Perturbations of x" and then pipeline that to acquire the predicted labels.
- Use this predictions to create a binary vector relative to the specific datapoint. vector[i] = 0 if the predicted label for the i-th perturbation is different from the predicted label for original x, otherwise is 1.
- After this procedure, use these binary vectors to train an attack model.
- Results are fairly competitive to the original attack and in many cases are quite better, given that our model takes into consideration the sensitivity of the model that the relation of confidences to the training and non training examples
- The query complexity of those attacks are quite big especially for image data
- This attack can break almost all confidence masking defences, since the latter always do not change the actual predicted label but only add noise to the confidence scores but retain the highest scored label.
- THIS ATTACK WILL BREAK THE DATA AUGMENTATION CURE for overfitting since by definition it exploits the relation of training dataset to its perturbation, given a trained model. 
</div>



### Membership Inference Attacks: A Survey - Hu et al

General overview of papers regarding MIAs 

### Deep Learning with Differential Privacy - Abadi et al

#### Datasets
<div id="Datasets">

- MNIST
- AT & T dataset of faces
</div>

#### Target Model Type
<div id="Target Model Types">

- GANs
- MLaS
- Deep NNs
</div>


#### Summary
<div id="Summary">

</div>



### Do Not Trust Prediction Scores for Membership Inference Attacks - Hintersdorf et al

#### Datasets
<div id="Datasets">

- MNIST
- AT & T dataset of faces
</div>

#### Target Model Type
<div id="Target Model Types">

- GANs
- MLaS
- Deep NNs
</div>


#### Summary
<div id="Summary">

</div>