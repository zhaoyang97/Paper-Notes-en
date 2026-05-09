---
title: >-
  [Paper Note] Mind the Cost of Scaffold! Benign Clients May Even Become Accomplices of Backdoor Attack
description: >-
  [ICCV 2025][AI Safety][federated learning] This paper proposes BadSFL, the first backdoor attack tailored to the Scaffold federated learning algorithm. By manipulating control variates, BadSFL turns benign clients into unwitting accomplices. Combined with GAN-based data augmentation and an optimization strategy that predicts the global model's future convergence direction, BadSFL achieves backdoor persistence lasting 60+ rounds after the attack ceases in non-IID settings—three times longer than baseline methods.
tags:
  - ICCV 2025
  - AI Safety
  - federated learning
  - backdoor attack
  - Scaffold
  - control variate
  - Non-IID
date: 2026-05-08
content_hash: 34326b03633f333f
---

# Mind the Cost of Scaffold! Benign Clients May Even Become Accomplices of Backdoor Attack

**Conference**: ICCV 2025
**arXiv**: [2411.16167](https://arxiv.org/abs/2411.16167)
**Code**: None
**Area**: AI Security
**Keywords**: federated learning, backdoor attack, Scaffold, control variate, Non-IID

## TL;DR

This paper proposes BadSFL, the first backdoor attack tailored to the Scaffold federated learning algorithm. By manipulating control variates, BadSFL turns benign clients into unwitting accomplices. Combined with GAN-based data augmentation and an optimization strategy that predicts the global model's future convergence direction, BadSFL achieves backdoor persistence lasting 60+ rounds after the attack ceases in non-IID settings—three times longer than baseline methods.

## Background & Motivation

### State of the Field

Federated learning (FL) protects client data privacy through distributed training. Under non-IID data distributions, local optima diverge from the global optimum (client drift), causing poor convergence in standard methods such as FedAvg. Scaffold mitigates client drift by introducing control variates to correct each client's local gradient update direction, making it the de facto approach for non-IID FL.

### Limitations of Prior Work

**Security of Scaffold has not been studied**: Although backdoor attacks in FL have been extensively investigated, virtually all existing work targets standard aggregation algorithms such as FedAvg. The control variate mechanism introduced by Scaffold creates an entirely new attack surface that has been completely overlooked.

**Non-IID settings are unfavorable to attackers**: Without knowledge of the global data distribution, naively injecting a backdoor causes severe degradation in benign-task performance (e.g., PTA drops from 85% to 25% in prior approaches), making the attack easily detectable.

**Poor backdoor persistence**: Once the attacker stops participating, benign updates gradually wash out the backdoor—existing baselines see backdoor accuracy fall below 50% within 20 rounds after the attack stops.

### Root Cause

Scaffold's control variate $c$ is used by all clients to correct local gradient updates: $w_i \leftarrow w_i - \eta_l(g_i(w_i) - c_i + c)$. If an attacker manipulates the uploaded control variate $\Delta c_p$, they can **indirectly steer the gradient directions of all benign clients**, causing those clients to unknowingly update in a direction that favors backdoor persistence. This constitutes a strategy of "turning benign clients into accomplices."

## Method

### Overall Architecture

BadSFL consists of four steps:
1. **Initialization**: Download the global model $w_g$ and control variate $c$.
2. **GAN-based data augmentation**: Generate samples from classes absent in the attacker's non-IID dataset.
3. **Backdoor trigger injection**: Select backdoor samples from the augmented dataset and manipulate their labels.
4. **Backdoor training with control variate constraint**: Optimize the backdoor model to align with the predicted future state of the global model.

### Key Designs

#### 1. **Dataset Supplementation via GAN**
- **Function**: In non-IID settings, the attacker holds data for only a subset of classes. A GAN is used to generate samples for the missing classes.
- **Mechanism**: Generator $G$ uses a series of deconvolutional layers to transform noise into images; Discriminator $D$ reuses the feature extraction layers of the current global model. At each round, the latest global model is downloaded to update $D$, after which $G$ is optimized to produce more realistic synthetic samples. The generated samples $D_f$ are merged with the attacker's original data $D^i$ to form the augmented dataset $D_c$.
- **Design Motivation**: Without augmentation, local training in a non-IID setting causes the local model to deviate severely from the global optimum, collapsing main-task accuracy. Augmentation brings the attacker's local optimum closer to the global optimum, preventing the backdoor from being diluted after aggregation.

#### 2. **Trigger Injection Strategy**
- **Function**: Inject a backdoor into the augmented dataset.
- **Three variants**:
    - **Label Flipping**: Relabel an entire class (e.g., dog→bird).
    - **Pattern Trigger**: Add a small triangular pattern to an image corner.
    - **Feature-based Trigger**: Use naturally occurring intra-class features as triggers (e.g., green cars, red boats).
- **Design Motivation**: Feature-based triggers are the most stealthy because they exploit natural data variation without modifying images, minimizing conflict with benign updates.

#### 3. **Control-Variate-Constrained Backdoor Optimization**
- **Function**: Leverage the global control variate $c$ to predict the global model's future convergence direction, optimizing the backdoor to remain effective in subsequent rounds.
- **Core formulas**:
    - Standard backdoor objective: $w_p^* = \arg\min_{w_p} L(D_p, w_p)$
    - Predicted global model after $j$ future rounds: $P_j(w_p, c) = \frac{w_p + w_g \cdot (n-1)}{n} - \eta_l \cdot c \cdot j$
    - BadSFL objective: $w_p^* = \arg\min_{w_p} [L(D_p, w_p) + L(D_p, P_j(w_p, c))]$
- **Manipulated control variate**: $\Delta c_p = \frac{1}{K \cdot \eta_l}(w_g - w_p) - c$
- **Design Motivation**: The global control variate $c$ represents the global convergence direction. By simulating future aggregation and applying the backdoor objective to the predicted future model, the method ensures the backdoor is not erased by subsequent benign updates. Meanwhile, the manipulated control variate propagates through server aggregation to all benign clients, covertly steering their gradient directions.

### Attack Protocol

- The attacker joins at round 10 and withdraws at round 40.
- 20 clients in total; 50% are selected per round.
- Data are sorted by label into 200 shards and randomly assigned to clients (non-IID).
- Future prediction horizon: $j = 10$.

## Key Experimental Results

### Main Results (CIFAR-10, Feature-based Trigger: Plane in Sunset)

| Method | BTA During Attack | BTA After Attack Stops (rounds 40→100) | Persistence (rounds) |
|---|---|---|---|
| Black-box Attack | ~70% | Drops below 50% (~round 60) | ~20 |
| Neurotoxin | ~75% | Drops below 50% (~round 65) | ~25 |
| IBA | ~65% | Drops below 40% (~round 60) | ~20 |
| 3DFed | ~70% | Drops below 50% (~round 60) | ~20 |
| **BadSFL** | **>90%** | **>90% (sustained to round 100)** | **>60** |

### Effect of GAN Data Augmentation

| Dataset | Method | PTA During Attack |
|---|---|---|
| CIFAR-10 | Without augmentation | <25% |
| CIFAR-10 | **With augmentation** | **~55%** |
| MNIST | Without augmentation | <75% |
| MNIST | **With augmentation** | **>90%** |

### Defense Experiments (CIFAR-10, Feature-based Trigger)

| Defense | BadSFL BTA | Baseline BTA | Note |
|---|---|---|---|
| Differential Privacy + Model Pruning | >80% | <50% | BadSFL remains effective |
| FLAME | >70% | <40% | BadSFL remains effective |
| SparseFed | >75% | <40% | BadSFL remains effective |

### Key Findings

- **BadSFL's persistence is more than 3× that of baselines**: After the attack stops, baseline BTA falls below 50% within ~20 rounds, whereas BadSFL sustains above 90% for over 60 rounds.
- **GAN data augmentation is critical for preserving main-task accuracy**: Without augmentation, PTA collapses to 25% on CIFAR-10; with augmentation, it is maintained at 55%.
- **Feature-based triggers outperform label flipping and pattern triggers**: Their natural character minimizes conflict between backdoor and benign updates.
- **Applying all four evaluated defenses simultaneously still fails to mitigate BadSFL effectively.**
- **Neurotoxin does not exhibit its expected persistence advantage in SFL**: The control variate mechanism of Scaffold likely alters the dynamics of parameter updates.

## Highlights & Insights

1. **Identification of a new attack surface**: The control variate in Scaffold was introduced to improve convergence, but simultaneously provides attackers with a channel to "manipulate the global direction and thereby indirectly control benign clients"—a textbook case of "feature as vulnerability."
2. **Elegance of the accomplice mechanism**: Rather than directly combating benign updates, the attacker manipulates control variates so that benign clients unknowingly update in a backdoor-favorable direction. This indirect manipulation is more stealthy and persistent than conventional model replacement attacks.
3. **Future-aware optimization via control variates**: Using $c$ as a proxy for the global convergence direction to simulate future aggregation and optimize the backdoor accordingly is a "forward-looking" strategy generalizable to other FL attack scenarios.
4. **Implication for FL security research**: Algorithmically superior FL methods are not necessarily more secure—Scaffold's correction mechanism turns out to amplify attack effectiveness.

## Limitations & Future Work

1. **PTA still degrades**: A main-task accuracy of 55% on CIFAR-10, while better than the unaugmented baseline, remains far below normal training (~80%) and may raise suspicion.
2. **Multi-round participation assumed**: The attacker must participate continuously from round 10 to round 40, which may be detectable in real systems.
3. **GAN quality dependency**: The effectiveness of data augmentation depends on GAN generation quality under the constrained conditions of FL.
4. **Only classification tasks evaluated**: Generalization to more complex tasks such as detection and segmentation is not verified.
5. **Adaptive defenses not discussed**: Defense methods specifically targeting anomalous control variates are not explored.
6. **Low PTA on CIFAR-10**: A PTA of 55%, reflecting an inherent challenge of SFL under non-IID settings, may limit applicability in practical scenarios.

## Related Work & Insights

- Scaffold (Karimireddy et al., 2020) is the standard solution for non-IID FL—this paper reveals the security cost of its control variate mechanism.
- Neurotoxin (Zhang et al.) enhances backdoor persistence by locking low-frequency parameters, but fails in SFL where Scaffold's control variates alter parameter update dynamics.
- 3DFed's multi-layer backdoor framework is effective under FedAvg but likewise cannot match BadSFL's persistence in SFL.
- GAN-based data augmentation from transfer learning is successfully adapted to the FL attack setting.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — First to expose the security vulnerability of Scaffold's control variate; the concept of turning benign clients into accomplices is highly original.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Three datasets, three trigger types, four defenses, and ablation studies provide broad coverage.
- **Writing Quality**: ⭐⭐⭐ — Method descriptions are clear, but some experimental results require consulting the appendix; quantitative results in the main text rely heavily on figures rather than tables.
- **Value**: ⭐⭐⭐⭐⭐ — Carries significant warning value for FL security research, highlighting the deeper issue that algorithmic improvements may introduce new security risks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Taught Well, Learned Ill: Towards Distillation-Conditional Backdoor Attack](../../NeurIPS2025/ai_safety/taught_well_learned_ill_towards_distillation-conditional_backdoor_attack.md)
- [\[ICCV 2025\] Backdoor Attacks on Neural Networks via One-Bit Flip](backdoor_attacks_on_neural_networks_via_one_bit_flip.md)
- [\[ICCV 2025\] Find a Scapegoat: Poisoning Membership Inference Attack and Defense to Federated Learning](find_a_scapegoat_poisoning_membership_inference_attack_and_defense_to_federated_.md)
- [\[ICCV 2025\] Backdoor Mitigation by Distance-Driven Detoxification](backdoor_mitigation_by_distance-driven_detoxification.md)
- [\[NeurIPS 2025\] Cost Efficient Fairness Audit Under Partial Feedback](../../NeurIPS2025/ai_safety/cost_efficient_fairness_audit_under_partial_feedback.md)

</div>

<!-- RELATED:END -->
