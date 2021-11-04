## Meeting Context

- First paper implementation (Shokri's original paper)
- NN-MIA attack based on paper was successful but fairly easy.

## Notes for next session
- [ ] Test it on a more generalized model
- [ ] Try to train 1 classifier in general, __instead of 1 classifier per class__ 
- [ ] Implement the same attack without supplying attacker with the proba-vector
- [ ] Try attack when some features are missing
- [ ] Further investigate data-synthesis algorithm


## Attack Explained (TODO: Elaborate on the attack further)

### Attack Model
- _Target Model_ : f_target() is the victim and we want to infer membership of datapoints in its training dataset 
- _Shadow Model_ : f_shadow() is one of the models that mimic f_target. Helps with acquiring same predictive behaviour as the target model
- _Attack Model_ : f_attack() is the attack model that infers membership of a given datapoint.

### Assumptions
- Target model is overfitted in its training data.
- Target model provides us with probability vectors that give away its confidence in predicting f_target(x), for a given datapoint x.
- Shadow model training set is completely __disjoint__ with target's training set, as in the _worst case senario_. 

### Shadow Models
Shadow models are models with architecture close to target model and mimic its predictive behaviour. They are trained on a part of D_shadow_i, one per shadow model, a dataset disjoint to the target's training dataset.

### Attack Model
We use all the datapoints in shadow datasets to query the models, knowing the ground truth:
- for all x in D_shadow_i that was used to train shadow_i, we acquire a probability vector that we label with class 'in'
- for all x in D_shadow_i not used for training, we label the probability vector with 'out'

These datapoints with format <y-class, prob-vec, 'in'/'out'> will consist the attack model's dataset. The model will use prob-vec as features and based on a supervised learning paradigm will be able to perform membership inference for a given datapoint, specific to the target model. 

### My Set up
- **Dataset** : Cifar-10. samples of size 2000
- **Target Model**: Simple 2 layer CNN with tanh activation function
- **Shadow Model Formula**: Same as the target (ideal scenario - might need to lose it)
- **Attack Model**: Dense NN with 4-5 layers and relu activation
- **# Shadows**: 20
- **Shadow Dataset Size**: 3000, disjoint from target's training data to simulate complete ignorance to D_target from attacker.

### Attack evaluation
Evaluation was based on accuracy on 2 disjoint sets. First one was a subset of target model's training set. Second was a completely disjoint set to D_target_training.  

