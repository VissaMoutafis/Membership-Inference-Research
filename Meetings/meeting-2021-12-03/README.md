# Meeting Notes


## Meeting Context
- We talked about MIA in a less overfitted model and established that we should __check attack's performance in a generalized classifier__ in order to display why the __vulnerability__ to be exploited is _the overfit on training data_ 
- We displayed the basic __label-only__ attack based on perturbation and test-train data behaviour on boundary attacks.

## TODO's
<<<<<<< HEAD
- [x] Apply the perturbations in the target's training dataset and retry attack
- [x] __In given attack__ check if training or test data are more label-sensitive, when perturbed
- [ ] Interesting direction: "_Can we tune the perturbations properly, in order ot achieve max attack accuracy_". This might be done in GAN style.
- [ ] Try the same attack in a state of the art model and display if vulnerable (Optional)
- [ ] Perform better label balancing in the attack dataset in order to achieve best model performance (**DO IN BOTH CONF-VEC AND LABEL-ONLY ATTACK TO COMPARE RESULTS.**

## Important NOTEs
- [ ] __Improve presentation by adding some graphs__
- [ ] Try document more precisely each step of the given algorithms implementation
