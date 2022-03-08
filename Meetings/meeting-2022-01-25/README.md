# Meeting Notes

## General
We discussed about the notebooks in this directory and concluded that 
- in label only attack the vulnerability we exploit is that, in an overfitted model the training samples are way more robust in perturbations than the non-training instances.
- the more features we miss the less our attack will score

We also decided to close couple of directions, as well as open some other ones.
## More Research Points
### Differential Privacy with MIAs and other defences
Find and read more papers on differential privacy and MIAs.

## Soon-to-be Closed Direcitions

### Overfit vs Attack Performance :heavy_check_mark:
We will finish our research on overfitting-attack score, by training a model till it reaches its top prediction state with no overfit. After that we will train the model by 10 steps at a time. Every time we will attack the produced model and record the attack scores. At the end we will present the plots that show the **precision**, **recall** and **AUC score** of the attack at each point.

We will do that for CIFAR-10 and (maybe) MNIST dataset.


### Label Only Attack in no-image Datasets

We will explore other datasets so that we will prove experimentaly that the label only attack will score at least as high as the original attack and maybe even better.
We will attack based on datasets that do not contain images.
We will try the purchase dataset, constructed recently.

### MIA with missing values :heavy_check_mark:

We will the `ADULT_mia_v1.ipynb` attack scenario where we want to infer the memebership of instances that might contain missing values-features. In first version we train the attack shadow models and attack model in data that **do not** contain missing values. WHAT IF we used data with missing features in the attack during the training process?  

## New directions proposed. Further Research Needed.

### Generate confidence vectors from perturbation predictions

The main idea here is to try train a model with X,y samples where 
- X : is a vector with the predicted labels for each perturbation 
- y : is a confidence vector 

### Tune perturbations 
The goal is to find a GAN-like architecture that will tune the perturbation in order to achieve highest MIA scores

### Add DP in Label Only Attack
Use DP to defend from perturbation Label only attack

Idea: Use DP to make the model more robust in prediction of perturbed instances => avoid pattern recognition on label vector. 

Here the 'neighbour' paradigm follows these 2 rules
- 2 points, where one is perturbation of the other are neighbours.
- 2 points in the original dataset, that are not explicitly perturbations of each other are not neighbours. 
