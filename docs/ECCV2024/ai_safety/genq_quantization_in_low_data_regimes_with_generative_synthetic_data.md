---
title: >-
  [Paper Note] Event Trojan: Asynchronous Event-based Backdoor Attacks
description: >-
  [ECCV2024][AI Safety][backdoor attack] This paper proposes the Event Trojan framework, which, for the first time, designs backdoor attack methods specifically for asynchronous event data streams. It includes two modes, namely immutable triggers and mutable triggers, directly injecting malicious events at the event stream level to achieve stealthy and efficient backdoor attacks.
tags:
  - "ECCV2024"
  - "AI Safety"
  - "backdoor attack"
  - "event camera"
  - "event data"
  - "immutable trigger"
  - "mutable trigger"
date: 2026-05-08
content_hash: 15a6b289ea92d60b
---

# Event Trojan: Asynchronous Event-based Backdoor Attacks

**Conference**: ECCV2024  
**arXiv**: [2407.06838](https://arxiv.org/abs/2407.06838)  
**Code**: To be confirmed  
**Area**: AI safety  
**Keywords**: backdoor attack, event camera, event data, immutable trigger, mutable trigger, AI safety

## TL;DR

This paper proposes the Event Trojan framework, which, for the first time, designs backdoor attack methods specifically for asynchronous event data streams. It includes two modes, namely immutable triggers and mutable triggers, directly injecting malicious events at the event stream level to achieve stealthy and efficient backdoor attacks.

## Background & Motivation

Event cameras are widely used in vision tasks such as autonomous driving, object tracking, surveillance, and gesture recognition due to their high dynamic range, low latency, and high temporal resolution. Asynchronous event data typically needs to be converted into image-like representations (e.g., Event Spike Tensor, EST) before being fed into deep learning models.

Existing image backdoor attack methods (such as BadNets, FIBA) can only inject triggers at the event representation level. However, in practical scenarios, attackers usually cannot access the intermediate output of the representation module. Additionally, if the representation is reconstructed from the untampered original event stream, representation-layer-based backdoors will fail. Therefore, directly injecting triggers into the original event data stream possesses higher practical threat value.

Since event data consists of a large number of asynchronous event points that are difficult for the human eye to perceive directly, it provides natural conditions for highly stealthy backdoor attacks. However, prior to this work, almost no study had explored the backdoor attack risks of event data, leaving relevant tasks facing potential security threats.

## Core Problem

1. How to directly inject backdoor triggers into asynchronous event data streams, bypassing the dependency on event representation modules?
2. How to design triggers that achieve both a high attack success rate (ASR) and high stealthiness?
3. How to make the triggers adapt to the distribution characteristics of different event data to enhance generalization capability?

## Method

### Threat Model

- **Attacker's Capability**: The attacker cannot control model training details (architecture, loss function, etc.) but can access a portion of the training data. During the inference phase, the attacker can only access raw event data and cannot manipulate the inference process or acquire information about representation methods.
- **Attacker's Goal**: To create an event model with a stealthy backdoor that outputs the attacker's preset label when specific trigger patterns are injected, while maintaining correct predictions on clean data.

### Event Data Basics

Each event $e_k = (x_k, y_k, t_k, p_k)$ contains spatial coordinates $(x, y)$, a timestamp $t$, and polarity $p$. An event is triggered when the pixel intensity change exceeds a threshold $\sigma$. Event streams are converted into grid representations $V_{\pm}(x_w, y_h, t_n)$ using methods like EST before being input to downstream models.

### Immutable Trigger

The core idea is to place malicious events at specific spatial locations and timestamps in the event stream:

- Spatial coordinates $(x, y)$ are sampled from a predefined area.
- Timestamps are fixed to $\alpha$ (default $10^{-2}$).
- Polarity is fixed to $\beta$ (default $1.0$).
- Once injected, they present a consistent pattern across different image-like representations of event streams.

The advantage is simple implementation, which effectively attacks most classifiers; the disadvantage is that the fixed settings result in limited generalization ability, and performance drops in scenarios with high background noise or uneven data distribution.

### Mutable Trigger

To overcome the limitations of the immutable trigger, an adaptive learning mechanism is proposed:

1. **Timestamp Adaptation**: Malicious events maintain identical spatial coordinates in different event streams (ensuring consistent trigger shape) but adopt adaptive timestamps, causing the trigger to present different pixel values in the image-like representations.
2. **Trigger Generator** $T_\xi(\cdot)$: Constructed based on a 5-layer MLP (64 channels per layer), taking $m$ timestamps randomly sampled from the original event as input and outputting malicious timestamps.
3. **Joint Training**: The classifier and the trigger generator are optimized jointly, with the classifier guiding the generator to learn the optimal trigger pattern.

### Loss & Training

$$\mathcal{L}_T = \lambda_1 \frac{T_\xi(\mathbf{t}) \cdot \mathbf{t}}{\|T_\xi(\mathbf{t})\| \times \|\mathbf{t}\|} + \lambda_2 \psi(T_\xi(\mathbf{t}), \mathbf{t})$$

- **Cosine Similarity Term** ($\lambda_1=1$): Minimizes the cosine similarity between malicious timestamps and original timestamps, increasing the difference between the trigger and clean data to enhance attacking capability.
- **Expectation-Variance Constraint Term** $\psi(\cdot)$ ($\lambda_2=2$): Constrains the expectation and variance of malicious timestamps to be close to the original data, preventing the generated events from deviating too far from the normal data distribution and maintaining classification performance on clean data.

The two terms work synergistically: the cosine term is responsible for attack effectiveness, while the $\psi$ term ensures stealthiness.

### Training Process

Following Algorithm 1, in each iteration:
1. Sample a mini-batch of event data.
2. Generate poisoned event streams using the mutable trigger generator.
3. Jointly optimize classifier parameters $\theta$, representation module parameters $\omega$, and trigger generator parameters $\xi$.

## Key Experimental Results

### Datasets

- **N-Caltech101**: An event-based version of Caltech101, containing 101 classes, with 4356/2612/1741 (train/validation/test) splits. The data volume varies significantly across classes.
- **N-Cars**: An event-based car detection dataset, with 2 classes, and 8392/2462/8608 splits.

### Main Results (ResNet-18 on N-Caltech101)

| Method | CDA↑ | ASR↑ |
|------|------|------|
| Representation Trigger BadNets | 57.24 | 0.0 |
| Representation Trigger FIBA | 82.47 | 43.39 |
| Immutable Trigger | 85.61 | 96.73 |
| **Mutable Trigger** | **86.21** | **99.71** |

### Main Results (Multi-model on N-Cars)

The mutable trigger achieves an ASR close to or equal to 100% on N-Cars across ResNet-18, VGG-16, Swin-S, and ViT-S, while maintaining the highest CDA (92.72, 92.93, 94.76, 87.17).

### Defense Experiment (Neural Polarizer)

| Method | CDA↑ | ASR↑ |
|------|------|------|
| BadNets | 60.00 | 1.03 |
| FIBA | 15.63 | 0.0 |
| Immutable Trigger | 66.84 | 22.01 |
| **Mutable Trigger** | **83.03** | **64.11** |

The mutable trigger maintains the highest attack success rate even under defense because the trigger is directly injected into the event data itself, where benign features and poisoned features are tightly intertwined and difficult to separate via polarization.

### Ablation Study

- Without the cosine similarity term (w/o cos.): ASR drops to 11.93%, significantly reducing attack capability.
- Without the $\psi$ constraint term: ASR is 100%, but CDA drops to 85.67%, introducing confusion to clean data.
- Full loss function: CDA 86.21%, ASR 99.71%, achieving the optimal balance.

## Highlights & Insights

1. **Pioneering Work**: First to systematically study backdoor attacks on asynchronous event data, revealing security vulnerabilities in event-based vision tasks.
2. **Two-level Trigger Design**: Transitioning from simple fixed to adaptive learning, the immutable trigger serves as a baseline, while the mutable trigger achieves comprehensive improvements.
3. **Exquisite Loss Function Design**: Synergizing the cosine term and the expectation-variance term, a good balance is struck between attack effectiveness and stealthiness.
4. **Large-scale Experiments**: Comprehensive evaluation across 22 classifiers (CNN + Transformer) $\times$ 2 datasets.
5. **Robustness Against Defenses**: Maintains a high attack success rate even when facing defenses such as Neural Polarizer.

## Limitations & Future Work

1. **Dependency on Data Scale**: The attack success rate of some models is sub-optimal on small-scale, high-noise datasets (N-Caltech101), e.g., ViT-S achieves only 87.73%, and Inception-V3 performs even worse.
2. **Limited Defense Evaluations**: Only one defense, Neural Polarizer, was tested, lacking evaluation against classic defenses like Spectral Signatures, Fine-pruning, and STRIP.
3. **Only Classification Tasks Considered**: Not extended to more complex downstream tasks like detection and segmentation.
4. **Fixed Spatial Dimensions of Trigger**: The mutable trigger is only adaptive in the timestamp dimension, while spatial coordinates remain fixed, leaving room to explore fully adaptive triggers.
5. **Lack of Validation in Real Deployment Scenarios**: All experiments were conducted on public datasets without considering real-world event camera scenarios.

## Related Work & Insights

### Comparison with Related Work

| Dimension | Image Backdoor Attack (BadNets/FIBA) | Event Trojan |
|------|------------------------------|--------------|
| Attack Level | Image/Representation layer | Raw event data stream |
| Attack Feasibility | Requires access to representation module output | Only requires access to raw event stream |
| Stealthiness | Visible at the representation layer | Difficult for the human eye to perceive in event streams |
| Robustness | Fails once representation is reconstructed | Triggers are embedded in the data itself |
| Defense Resistance | Susceptible to polarization separation | Features are tightly intertwined and hard to separate |

Similar to 3D point cloud backdoor attacks (PointBA), this work exploits the asynchronous, sparse nature of the data itself to hide triggers, while event data additionally contains temporal dimension information to be leveraged.

### Insights & Connections

- Reveals the vulnerability of event-based vision systems in safety-critical scenarios (autonomous driving, surveillance), serving as an important warning for AI safety research.
- The adaptive learning concept of the mutable trigger can be extended to adversarial attack research on other non-standard data modalities (e.g., point clouds, radar signals).
- The trade-off pattern of "effectiveness vs. stealthiness" in the trigger optimization loss possesses universal reference value.
- Points out the direction for research on defense methods for event data: it is necessary to design purification and detection mechanisms specifically at the data stream level.

## Rating
- Novelty: ⭐⭐⭐⭐ — First systematic study of backdoor attacks on event data, offering a novel topic.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive evaluation across 22 models, although defense experiments are relatively sparse.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure and well-elaborated motivation.
- Value: ⭐⭐⭐⭐ — Highly alarming for event vision security, opening a new research direction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Time Is All It Takes: Spike-Retiming Attacks on Event-Driven Spiking Neural Networks](../../ICLR2026/ai_safety/time_is_all_it_takes_spike-retiming_attacks_on_event-driven_spiking_neural_netwo.md)
- [\[ECCV 2024\] Fisher Calibration for Backdoor-Robust Heterogeneous Federated Learning](fisher_calibration_for_backdoor-robust_heterogeneous_federated_learning.md)
- [\[ICLR 2026\] Defending against Backdoor Attacks via Module Switching](../../ICLR2026/ai_safety/defending_against_backdoor_attacks_via_module_switching.md)
- [\[ICCV 2025\] Backdoor Attacks on Neural Networks via One-Bit Flip](../../ICCV2025/ai_safety/backdoor_attacks_on_neural_networks_via_one_bit_flip.md)
- [\[ICML 2025\] Adversarial Inception Backdoor Attacks against Reinforcement Learning](../../ICML2025/ai_safety/adversarial_inception_backdoor_attacks_against_reinforcement_learning.md)

</div>

<!-- RELATED:END -->
