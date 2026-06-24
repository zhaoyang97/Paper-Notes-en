---
title: >-
  [Paper Note] Crafting Privacy-Preserving Adversarial Examples: A Defense Against Membership Inference
description: >-
  [ACL 2025][AI Safety][Privacy Protection] This paper proposes a method to defend against Membership Inference Attacks (MIA) by constructing privacy-preserving adversarial examples. It injects carefully designed perturbations into the model's prediction outputs, preventing attackers from determining whether a specific data point belongs to the training set, while maintaining service quality for normal users.
tags:
  - "ACL 2025"
  - "AI Safety"
  - "Privacy Protection"
  - "Adversarial Examples"
  - "Membership Inference Attack"
  - "Differential Privacy"
  - "Defense Mechanisms"
date: 2026-05-08
content_hash: ee252d197ac8ef81
---

# Crafting Privacy-Preserving Adversarial Examples: A Defense Against Membership Inference

**Conference**: ACL 2025  
**Area**: AI Safety  
**Keywords**: Privacy Protection, Adversarial Examples, Membership Inference Attack, Differential Privacy, Defense Mechanisms

## TL;DR
This paper proposes a method to defend against Membership Inference Attacks (MIA) by constructing privacy-preserving adversarial examples. It injects carefully designed perturbations into the model's prediction outputs, preventing attackers from determining whether a specific data point belongs to the training set, while maintaining service quality for normal users.

## Background & Motivation

**Background**: Membership Inference Attack (MIA) is a core threat in the field of machine learning privacy. Attackers determine whether a specific data point was used for model training by observing the model's predictive behavior on that data (such as prediction probabilities, confidence distributions). In the NLP field, MIA has been proven to pose severe privacy risks to tasks such as text classification and language models, potentially leaking sensitive information (such as medical records and personal conversations) from the training data.

**Limitations of Prior Work**: Existing MIA defense methods are mainly divided into three categories, each with its own limitations: (1) Differential Privacy Training (DP-SGD)—adds noise during the training process, providing strong privacy guarantees but usually significantly degrading model performance (a 5-15% drop in accuracy); (2) Regularization methods (e.g., Dropout, L2 regularization)—indirectly defend against MIA by reducing overfitting, but have limited effectiveness and do not provide theoretical privacy guarantees; (3) Knowledge Distillation—reduces leakage through a teacher-student framework, but increases training costs and provides insufficient defense against strong attackers.

**Key Challenge**: Defending against MIA requires a trade-off between "privacy protection" and "model utility"—stronger defense leads to greater interference with model outputs and more degradation of service quality. Existing methods perform poorly on this trade-off, either offering insufficient privacy protection or causing excessive performance loss.

**Goal**: To design an MIA defense method at the inference stage (rather than the training stage) that confuses attackers by minimizing perturbations to model outputs while having minimal impact on the experience of normal users.

**Key Insight**: The authors observe that the core signal relying on MIA attackers is the difference in output distributions between training members and non-members—members typically obtain higher confidence and sharper probability distributions. If this difference can be "smoothed" during inference, MIA can be defended against effectively. The key insight is that this smoothing does not need to treat all outputs equally; it only needs targeted perturbation of output features that may expose membership identity.

**Core Idea**: During the model inference phase, adversarial perturbations are dynamically generated based on the "membership exposure risk" of the output probability distribution. High-risk membership features (such as excessively high confidence) are adjusted to a range indistinguishable from non-members, achieving privacy protection at inference time.

## Method

### Overall Architecture
The defense system is deployed at the model inference end, serving as an intermediate layer between the model output and the user/API. The process is: the model receives a query and generates raw outputs $\rightarrow$ the risk evaluator assesses the membership exposure risk of the output $\rightarrow$ if the risk exceeds the threshold, the perturbation generator produces minimal adversarial perturbations $\rightarrow$ the perturbed output is returned to the querier. For low-risk queries, the output remains unmodified.

### Key Designs

1. **Membership Exposure Risk Evaluator**:

    - **Function**: Determines whether the model output for a given input is likely to expose its training membership identity.
    - **Mechanism**: A lightweight binary classifier (shadow model approach) is trained to simulate the attacker's perspective. It is trained using model outputs of a set of known members and non-members, and its output probability is used as the "exposure risk score". Specific features include: the maximum value of the predicted probability (max confidence), the entropy of the probability distribution, the correctness of the prediction, and the deviation from the average output of the same class. When the risk score $r \in [0, 1]$ exceeds the threshold $\tau$, a perturbation is triggered.
    - **Design Motivation**: Not all outputs require perturbation—for outputs that inherently fail to expose membership information, perturbation merely adds unnecessary noise. Risk evaluation makes the defense more precise and minimizes the impact on normal usage.

2. **Targeted Adversarial Perturbation Generator**:

    - **Function**: Generates output perturbations that can effectively confuse attackers while minimizing the impact on model utility.
    - **Mechanism**: The perturbation is modeled as a transformation of the probability distribution. A perturbation $\delta$ is added to the model output's logits vector $z$, so that the perturbed output $\text{softmax}(z + \delta)$ satisfies two optimization objectives: (a) maximize the attacker's prediction error (adversarial loss), i.e., adding perturbation to make the risk evaluator judge the output as a non-member; (b) minimize the perturbation magnitude (utility loss), keeping $|\delta|$ as small as possible to preserve the predicted label. The constrained optimization problem is solved using the method of Lagrange multipliers: $\min |\delta|$ s.t. $r(z+\delta) < \tau$. In the specific implementation, Projected Gradient Descent (PGD) is used to iteratively optimize within the perturbation space.
    - **Design Motivation**: The adversarial approach guarantees the effectiveness of the perturbation (directly opposing the attacker's discriminative model), while the minimization constraint ensures utility. PGD searching finds the minimal perturbation within the constrained space that keeps the label unchanged.

3. **Adaptive Perturbation Intensity Control**:

    - **Function**: Dynamically adjusts perturbation intensity according to different attack strategies and threat levels.
    - **Mechanism**: Maintains a perturbation intensity parameter $\epsilon$, whose initial value is determined based on statistical analysis of a calibration set. During deployment, $\epsilon$ is dynamically adjusted by detecting query patterns (e.g., a large number of similar queries in a short time may indicate an attacker probing). In addition, the perturbation intensity varies across different output dimensions—smaller perturbations on dimensions with high impact on the predicted label (preserving utility), and larger perturbations on dimensions containing significant information for the attacker (protecting privacy). Dimensional importance is determined through gradient analysis.
    - **Design Motivation**: Fixed perturbation intensity cannot adapt to different attack strategies. The adaptive mechanism gives the defense better generalization capabilities against unknown attacks.

### Loss & Training
The risk evaluator is trained on shadow model data using the binary cross-entropy loss. The perturbation generator is optimized in real-time during inference without requiring pre-training. PGD iterations for each perturbation typically converge in just 3-5 steps, adding approximately 10-15% of inference latency.

## Key Experimental Results

### Main Results

| Dataset / Model | Method | Model Acc↑ | MIA Attack Acc↓ | Privacy Leakage Rate↓ |
|------------|------|---------|-------------|-----------|
| SST-2 / BERT | No Defense | 92.3% | 73.5% | 47.0% |
| SST-2 / BERT | DP-SGD (ε=8) | 85.6% | 55.2% | 10.4% |
| SST-2 / BERT | Regularization | 91.1% | 68.4% | 36.8% |
| SST-2 / BERT | **Ours** | **91.8%** | **53.1%** | **6.2%** |
| AG News / RoBERTa | No Defense | 94.7% | 71.2% | 42.4% |
| AG News / RoBERTa | DP-SGD (ε=8) | 88.3% | 54.8% | 9.6% |
| AG News / RoBERTa | **Ours** | **94.1%** | **52.6%** | **5.1%** |
| MNLI / DeBERTa | No Defense | 89.5% | 69.8% | 39.6% |
| MNLI / DeBERTa | **Ours** | **89.0%** | **54.3%** | **8.7%** |

### Ablation Study

| Configuration | Model Acc | MIA Attack Acc | Description |
|------|--------|-----------|------|
| Full method | 91.8% | 53.1% | Complete method |
| w/o Risk Evaluator (Perturb All) | 90.4% | 52.8% | Perturbing all outputs, Acc decreases |
| w/o Adaptive Intensity | 91.6% | 56.7% | Fixed perturbation intensity, slightly weaker defense |
| w/o Targeted Perturbation (Random Noise) | 91.2% | 63.2% | Random noise defense is less effective |
| Temperature Scaling Only | 91.9% | 65.8% | Simple temperature adjustment is insufficient |

### Key Findings
- Targeted adversarial perturbations perform 10 percentage points better than random noise (53.1% vs 63.2%), demonstrating the necessity of the "adversarial" design.
- Selective perturbation guided by risk evaluation drops model accuracy by only 0.5% (91.8% vs 92.3%), far superior to the 6.7% drop of DP-SGD.
- Across three datasets and three models, MIA attack accuracy is successfully suppressed to near random guessing levels (~53%).
- The core advantage of inference-time defense is that it does not require model retraining and can be directly deployed on pre-trained models.

## Highlights & Insights
- The paradigm shift to inference-time defense is the core contribution—it does not change the model training process, applies minimal perturbations solely at the inference side, is compatible with any pre-trained model, and has extremely low deployment costs.
- The selective perturbation design of the risk evaluator is exquisite—it only processes "dangerous" outputs, leaving most outputs completely unaffected, thereby minimizing utility loss.
- Targeted adversarial perturbations directly counter the attacker's discriminative capability, making them much more targeted than heuristic defenses.

## Limitations & Future Work
- The defense effectiveness depends on the risk evaluator's accurate simulation of attacker behaviors; if the attacker uses a strategy completely different from the shadow model, the defense might fail.
- Currently, only text classification scenarios are validated; the effectiveness of the defense on generative models (such as MIA for LLMs) has not been verified.
- Inference latency increases by 10-15%, which may not be ideal in scenarios with low-latency requirements.
- Formal privacy guarantees (such as the $(\epsilon, \delta)$-guarantee of differential privacy) are not provided, and the theoretical analysis is not deep enough.
- Future research can explore combining inference-time defense with training-time defense to achieve stronger composite protection.

## Related Work & Insights
- **vs DP-SGD (Abadi et al.)**: DP-SGD adds noise during training, providing strict privacy guarantees but with significant performance loss; this work perturbs during inference, yielding minimal performance loss but lacking theoretical guarantees. The two are complementary.
- **vs MemGuard (Jia et al.)**: MemGuard is also an inference-time defense, but uses a fixed noise addition strategy; the targeted adversarial and risk evaluation mechanisms in this work make the defense more precise.
- **vs Knowledge Distillation Defenses**: Distillation methods require extra training processes and teacher models, which are costly to deploy; the plug-and-play design in this work is much lighter.

## Rating
- Novelty: ⭐⭐⭐⭐ The idea of adversarial defense against MIA during inference is relatively novel
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated across multiple datasets, models, and attack methods
- Writing Quality: ⭐⭐⭐⭐ Clear problem definition and rigorous methodology description
- Value: ⭐⭐⭐⭐ Possesses practical engineering value for NLP model privacy protection

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] CENTAUR: Bridging the Impossible Trinity of Privacy, Efficiency, and Performance in Privacy-Preserving Transformer Inference](centaur_bridging_the_impossible_trinity_of.md)
- [\[ICCV 2025\] Find a Scapegoat: Poisoning Membership Inference Attack and Defense to Federated Learning](../../ICCV2025/ai_safety/find_a_scapegoat_poisoning_membership_inference_attack_and_defense_to_federated_.md)
- [\[NeurIPS 2025\] Impact of Dataset Properties on Membership Inference Vulnerability of Deep Transfer Learning](../../NeurIPS2025/ai_safety/impact_of_dataset_properties_on_membership_inference_vulnerability_of_deep_trans.md)
- [\[AAAI 2026\] Reference Recommendation based Membership Inference Attack against Hybrid-based Recommender Systems](../../AAAI2026/ai_safety/reference_recommendation_based_membership_inference_attack_against_hybrid-based_.md)
- [\[ICLR 2026\] Curation Leaks: Membership Inference Attacks against Data Curation for Machine Learning](../../ICLR2026/ai_safety/curation_leaks_membership_inference_attacks_against_data_curation_for_machine_le.md)

</div>

<!-- RELATED:END -->
