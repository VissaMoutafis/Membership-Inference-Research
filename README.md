# Membership-Inference Attacks against Machine Learning Models
Undergraduate Thesis on Membership Inference Attacks. This repo is consisted of 
- Research papers
- External resources
- Programming Framework for attack implementation
- Thesis document describing the research process and findings along with experimental results

## General 
Membership Inference Attacks (MIAs) are a kind of inference attacks that enable a malicious user to decide on the membership status of a data-record, with respect to the training dataset of a target ML model. The main vulnerability exploited in this attack is derived from the fact that ML models make robust and over-confident predictions on data-points that they have previously seen, hence the model's behavrio on data-points seen during training will be noticable. Different attack variations utilize probability thresholds, machine learning algorithms and data perturbations in order to infer the membership of a target data-point. A successful attack means that user privacy is breached and that the ML model is vulnerable to attacks that leak user sensitive data, such as medical conditions (i.e. cancer), financial information and any other non trivial insight into the victim user's privacy. The first paper documented, Shokri et al [[1]](https://github.com/VissaMoutafis/Membership-Inference-Research/blob/main/Papers/MIA%20against%20ML%20Models.pdf) refers to a complete attack pipeline and provides us with detailed insights on why and how this attack works.

In our studies we focused our effort to both alterations of the original attack and defenses against it. MIAs, being data-driven attacks, are widely affected by the quality of the data an adversary posseses pre-attack and the context of the attack scenario (black-box, white-box, etc.). Furthermore, we conducted various experiments that focus on the MIA behavior with respect to the dataset under attack and the settings of both the attack set-up and the target model.

## Source Code
The framework was an attempt to abstract the attack pipeline and it is not considered to be optimized or ideal, in any case. It provides a generic approach to study MIAs and implements some python classes specifics to the conducted experiments, such as the LabelOnlyMIA class.

### TODO
- [ ] Optimize code and shadow models training processing with multithreading
- [ ] Upload working examples on various datasets such as CIFAR-10/100, MNIST and other categorical datasets.

## Contributors
- Supervisor: [Kostas Chatzikokolakis](https://www.chatzi.org/)
- External Consultant: [Nikos Galanis](https://www.linkedin.com/in/nikos-galanis/)
