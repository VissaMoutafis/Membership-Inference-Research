# Meeting Notes

## Pre-Meeting Links

- [Introduction to TensorFlow Privacy - Presentation](https://www.youtube.com/watch?v=S5j0kKgqbJc)

## General

## TODO for closed directions

### Check MIA with missing values (EASY)
Some strange results in **ADULT_mia_v2.ipynb**, in the final boards, where the $0\%$ MVP datasets (both test and shadow) performed really poorly. This was not the results of the original attack where we attacked with full on shadow dataset on missing valued - attack test dataset.

### Label Only in Categorical data (EASY)
- Create a notebook running both attacks and create comparative graphs

### Create a new metric and run some of the attacks to evaluate on it
Let us define $$A = \frac{Acc(Model(D_{train}))}{Acc(Model(D_{out}))}$$
and $$L = \frac{Loss(Model(D_{test}))}{Loss(Model(D_{train}))}$$

Now we can define a new metric that we will call **Model MIA Vulnerability** defined as:

$$
Vuln(Model) = \log 2\frac{A \cdot L}{A + L}
$$

Given that $A \geq 1, L \geq 1$ we have $Vuln(Model) \geq 0$ where the equality holds for the **optimal model with no overfit at all**.

## Further Research

1. Tensorflow privacy DP-SGD inner workings and tuning.

2. General Citation of MIA-related papers, for both attacks and defences. Specifically:
    - Check the citations from seen papers 
    - Create a board that summarizes 1.Under Attack Dataset Type, 2.Target  
        Model Type, 3. Propositions and Motivations, 4. A small summary of 
        conclusions 

TL;DR: Create a Bibliographical summary table.


## More Demanding Tasks

### Create a more efficient perturbator using a ML model, or an optimization framework
This could be done by creating a model (like an auto-encoder or even a simple FFN) or using another framework for optimization problem solving like python optuna and use as input $x$ and get output $\bar x$, the latter beign the perturbated datapoint. This will work only if we add noise in the output, otherwise the model will produce the input as the output, since it minimizes Cross Entropy Loss.

So how can we add noise to the final output perturbation, $\bar x$?

This might be done by defining a Loss function as 
$$Loss = CE - \frac{||x - \bar x||_1}{C}, C \in R^+$$ 

where $C$ is a **clip norm threshold**. Note that $C$, will be the scale of the noise we add.


**NOTE**: THIS THEORY IS NOT BULLETPROOF AND NEEDS TESTING.


### Create a DP study-notebook
Create a notebok where we will study the attack performance vs model performance, during the addition of the noise, with regard to epsilon $\epsilon$.